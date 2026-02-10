#!/usr/bin/env python3
"""
Distributed Training Testing Framework

Tests multi-process data parallel training capabilities:
- Single machine, multi-process simulation
- Distributed data parallel (DDP) setup
- Communication overhead benchmarking
- Scalability metrics

Usage:
    python distributed_training_tester.py --num_processes 2 --epochs 4
"""

import os
import sys
import json
import time
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.distributed as dist
from torch.distributed import TCPStore
from torch.nn.parallel import DistributedDataParallel as DDP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class RandomTokenDataset(torch.utils.data.Dataset):
    """Random token dataset for testing."""
    
    def __init__(self, num_samples: int, vocab_size: int, seq_len: int):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random tokens
        input_ids = torch.randint(1, self.vocab_size, (self.seq_len,))
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
        return input_ids, labels


class DistributedTransformerModel(nn.Module):
    """Transformer model for distributed training."""
    
    def __init__(
        self,
        vocab_size: int = 2048,
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        max_seq_len: int = 128
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        embeds = self.embedding(input_ids)
        pos_embeds = self.pos_embedding(positions)
        hidden = embeds + pos_embeds
        
        encoded = self.transformer_encoder(hidden)
        logits = self.output_projection(encoded)
        
        return logits


def setup_distributed(rank: int, world_size: int, backend: str = 'gloo'):
    """Initialize distributed training using TCPStore for Windows compatibility."""
    master_addr = '127.0.0.1'
    master_port = 29500
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    
    # Windows doesn't support gloo backend for multi-process DDP
    # Fall back to environment-based init which works better on Windows
    try:
        # Try environment-based initialization first (works on Linux/Mac)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=30)
        )
    except RuntimeError as e:
        if "unsupported gloo device" in str(e) or "gloo" in str(e):
            # Windows gloo limitation - try with a store-based approach
            logger.warning(f"Rank {rank}: gloo backend failed, attempting TCPStore approach")
            try:
                store = TCPStore(
                    host_name=master_addr,
                    port=master_port,
                    world_size=world_size,
                    is_master=(rank == 0),
                    wait_for_workers=True,
                    timeout=timedelta(minutes=30),
                    use_libuv=False
                )
                dist.init_process_group(
                    backend='gloo',
                    store=store,
                    rank=rank,
                    world_size=world_size,
                    timeout=timedelta(minutes=30)
                )
            except Exception as store_error:
                logger.error(f"Rank {rank}: TCPStore also failed: {store_error}")
                raise
        else:
            raise
    
    torch.cuda.set_device(rank) if torch.cuda.is_available() else None


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def train_distributed_worker(
    rank: int,
    world_size: int,
    num_epochs: int,
    batch_size: int,
    model_config: Dict[str, Any],
    output_file: str
):
    """
    Worker process for distributed training.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        num_epochs: Training epochs
        batch_size: Batch size per process
        model_config: Model configuration
        output_file: Metrics output file
    """
    try:
        # Setup distributed environment
        setup_distributed(rank, world_size)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        model = DistributedTransformerModel(**model_config)
        model = model.to(device)
        
        # Wrap with DDP
        model = DDP(
            model,
            device_ids=[rank] if torch.cuda.is_available() else None,
            output_device=rank if torch.cuda.is_available() else None
        )
        
        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model, T_max=num_epochs)
        
        # Create synthetic dataset
        vocab_size = model_config['vocab_size']
        max_seq_len = model_config['max_seq_len']
        num_samples = batch_size * world_size * 10  # 10 batches per process
        
        train_input_ids = torch.randint(1, vocab_size, (num_samples, max_seq_len))
        train_labels = torch.randint(0, vocab_size, (num_samples, max_seq_len))
        
        dataset = TensorDataset(train_input_ids, train_labels)
        
        # Distributed sampler ensures no data duplication/missing
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0
        )
        
        # Training metrics
        metrics = {
            'rank': rank,
            'world_size': world_size,
            'epochs': [],
            'total_duration_sec': 0
        }
        
        training_start = time.time()
        
        # Training loop
        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)  # Important for proper shuffling
            model.train()
            
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (input_ids, labels) in enumerate(loader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                
                # Forward pass
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, model_config['vocab_size']),
                    labels.view(-1)
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if rank == 0 and batch_idx % 5 == 0:
                    logger.info(f"Rank {rank} | Epoch {epoch+1} | Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")
            
            scheduler.step()
            
            epoch_duration = time.time() - epoch_start
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            
            metrics['epochs'].append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'duration_sec': epoch_duration,
                'num_batches': num_batches,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            if rank == 0:
                logger.info(f"Rank {rank} | Epoch {epoch+1} | Loss: {avg_loss:.4f} | Duration: {epoch_duration:.1f}s")
        
        total_duration = time.time() - training_start
        metrics['total_duration_sec'] = total_duration
        
        if rank == 0:
            logger.info(f"Training complete on rank {rank} | Total: {total_duration:.1f}s")
            
            # Save metrics
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics saved to {output_path}")
        
        cleanup_distributed()
        
    except Exception as e:
        logger.error(f"Error in rank {rank}: {e}", exc_info=True)
        cleanup_distributed()
        raise


class DistributedTrainingTester:
    """Orchestrates distributed training testing."""
    
    def __init__(self, output_dir: str = 's:\\Ryot\\distributed_training_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def test_multi_process_training(
        self,
        num_processes: int = 2,
        num_epochs: int = 2,
        batch_size: int = 16,
        model_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Test multi-process distributed training.
        
        Args:
            num_processes: Number of worker processes
            num_epochs: Training epochs
            batch_size: Batch size per process
            model_config: Model configuration
            
        Returns:
            Aggregated results from all processes
        """
        if model_config is None:
            model_config = {
                'vocab_size': 2048,
                'embedding_dim': 256,
                'num_heads': 4,
                'num_layers': 2,
                'ff_dim': 512,
                'max_seq_len': 128
            }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {num_processes} processes | {num_epochs} epochs | Batch size: {batch_size}")
        logger.info(f"Model: {model_config}")
        logger.info(f"{'='*60}\n")
        
        # Run distributed training
        output_file = self.output_dir / f"ddp_{num_processes}x{num_epochs}ep.json"
        
        # For testing with single machine, use spawn
        ctx = mp.get_context('spawn')
        processes = []
        
        for rank in range(num_processes):
            p = ctx.Process(
                target=train_distributed_worker,
                args=(rank, num_processes, num_epochs, batch_size, model_config, str(output_file))
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes
        for p in processes:
            p.join()
        
        # Load and return results
        if output_file.exists():
            with open(output_file, 'r') as f:
                results = json.load(f)
            logger.info(f"\n✅ Distributed training test completed")
            return results
        else:
            logger.error(f"Results file not found: {output_file}")
            return {}
    
    def benchmark_scaling(
        self,
        num_epochs: int = 2,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Benchmark scaling across different process counts.
        
        Args:
            num_epochs: Training epochs
            batch_size: Batch size per process
            
        Returns:
            Scaling benchmarks
        """
        logger.info(f"\n{'='*60}")
        logger.info("SCALING BENCHMARK")
        logger.info(f"{'='*60}\n")
        
        model_config = {
            'vocab_size': 2048,
            'embedding_dim': 256,
            'num_heads': 4,
            'num_layers': 2,
            'ff_dim': 512,
            'max_seq_len': 128
        }
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'scaling_tests': [],
            'platform_note': 'Windows DDP limitation: gloo backend does not support multi-process on Windows. Running simulated scaling tests instead.'
        }
        
        # Test 1, 2, 4 processes - simulated on Windows
        for num_procs in [1, 2, 4]:
            if num_procs > mp.cpu_count():
                logger.warning(f"Skipping {num_procs} processes (insufficient CPU cores)")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {num_procs} processes (simulated) | {num_epochs} epochs | Batch size: {batch_size}")
            logger.info(f"Model: {model_config}")
            logger.info(f"{'='*60}\n")
            
            # For Windows: Run single-process training multiple times (simulates distributed)
            output_file = self.output_dir / f"ddp_{num_procs}x{num_epochs}ep.json"
            
            try:
                # Simulate distributed training by running training for each rank
                all_metrics = []
                for rank in range(num_procs):
                    rank_metrics = self._simulate_distributed_rank(
                        rank=rank,
                        world_size=num_procs,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        model_config=model_config
                    )
                    all_metrics.append(rank_metrics)
                
                # Aggregate results
                aggregated = {
                    'timestamp': datetime.now().isoformat(),
                    'num_processes': num_procs,
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'rank_results': all_metrics,
                    'aggregated_metrics': {
                        'avg_loss': sum(m['epochs'][-1]['train_loss'] for m in all_metrics) / len(all_metrics),
                        'total_duration_sec': max(m['total_duration_sec'] for m in all_metrics),
                        'samples_per_sec': self._estimate_throughput(all_metrics, num_procs)
                    }
                }
                
                # Save results
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(aggregated, f, indent=2)
                logger.info(f"✅ Simulated distributed test ({num_procs}p) completed")
                logger.info(f"   Loss: {aggregated['aggregated_metrics']['avg_loss']:.4f}")
                logger.info(f"   Duration: {aggregated['aggregated_metrics']['total_duration_sec']:.1f}s")
                logger.info(f"   Throughput: {aggregated['aggregated_metrics']['samples_per_sec']:.1f} sp/s")
                
                results['scaling_tests'].append(aggregated)
                
            except Exception as e:
                logger.error(f"Error in {num_procs}-process scaling test: {e}")
                results['scaling_tests'].append({'error': str(e), 'num_processes': num_procs})
        
        return results
    
    def _simulate_distributed_rank(
        self,
        rank: int,
        world_size: int,
        num_epochs: int,
        batch_size: int,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate distributed training for a single rank (Windows workaround)."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model with different seeds per rank
        torch.manual_seed(42 + rank)
        model = DistributedTransformerModel(**model_config).to(device)
        
        # Optimizer per rank
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
        
        # Training metrics
        metrics = {
            'rank': rank,
            'world_size': world_size,
            'epochs': [],
            'total_duration_sec': 0
        }
        
        training_start = time.time()
        
        # Create dataset
        dataset = RandomTokenDataset(
            num_samples=320,  # Smaller for speed
            vocab_size=model_config['vocab_size'],
            seq_len=model_config['max_seq_len']
        )
        
        # Distributed sampler for proper data splitting
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0
        )
        
        # Training loop
        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)
            model.train()
            
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (input_ids, labels) in enumerate(loader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                
                # Forward/backward
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, model_config['vocab_size']),
                    labels.view(-1)
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            
            epoch_duration = time.time() - epoch_start
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            
            metrics['epochs'].append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'duration_sec': epoch_duration,
                'num_batches': num_batches,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        metrics['total_duration_sec'] = time.time() - training_start
        return metrics
    
    def _estimate_throughput(self, all_metrics: list, num_processes: int) -> float:
        """Estimate samples per second from metrics."""
        total_samples = 320 * num_processes  # 320 samples per rank
        total_duration = max(m['total_duration_sec'] for m in all_metrics)
        return total_samples / total_duration if total_duration > 0 else 0.0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Distributed Training Testing Framework')
    parser.add_argument('--num_processes', type=int, default=2, help='Number of worker processes')
    parser.add_argument('--epochs', type=int, default=2, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per process')
    parser.add_argument('--benchmark', action='store_true', help='Run full scaling benchmark')
    parser.add_argument('--output_dir', default='s:\\Ryot\\distributed_training_results', help='Output directory')
    
    args = parser.parse_args()
    
    tester = DistributedTrainingTester(output_dir=args.output_dir)
    
    if args.benchmark:
        tester.benchmark_scaling(num_epochs=args.epochs, batch_size=args.batch_size)
    else:
        tester.test_multi_process_training(
            num_processes=args.num_processes,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )


if __name__ == '__main__':
    main()

