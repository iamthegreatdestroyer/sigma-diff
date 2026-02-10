#!/usr/bin/env python3
"""
Model Scaling Testing Framework

Tests training stability and performance across different model sizes:
- Small model (baseline): ~3.5M parameters
- Medium model: ~14M parameters  
- Large model: ~56M parameters

Metrics:
- Training time per epoch
- Memory usage
- Throughput (samples/sec)
- Loss convergence curves
- Scaling efficiency

Usage:
    python model_scaling_tester.py --model_size medium --epochs 4
    python model_scaling_tester.py --benchmark  # Test all sizes
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tracemalloc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ScalableTransformerModel(nn.Module):
    """Transformer model with configurable size."""
    
    def __init__(
        self,
        vocab_size: int = 2048,
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: Optional[int] = None,
        max_seq_len: int = 128
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = embedding_dim * 4
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
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
        self.max_seq_len = max_seq_len
        
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
    
    def count_parameters(self) -> int:
        """Count model parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.transformer_encoder.layers[0].linear1.out_features,
            'max_seq_len': self.max_seq_len,
            'num_parameters': self.count_parameters()
        }


class ModelScalingTester:
    """Tests model scaling characteristics."""
    
    # Predefined model sizes
    MODEL_CONFIGS = {
        'small': {
            'vocab_size': 2048,
            'embedding_dim': 256,
            'num_heads': 4,
            'num_layers': 2,
            'description': 'Small (baseline)'
        },
        'medium': {
            'vocab_size': 4096,
            'embedding_dim': 512,
            'num_heads': 8,
            'num_layers': 4,
            'description': 'Medium'
        },
        'large': {
            'vocab_size': 8192,
            'embedding_dim': 1024,
            'num_heads': 16,
            'num_layers': 6,
            'description': 'Large'
        },
        'xlarge': {
            'vocab_size': 16384,
            'embedding_dim': 2048,
            'num_heads': 32,
            'num_layers': 8,
            'description': 'Extra Large'
        }
    }
    
    def __init__(self, output_dir: str = 's:\\Ryot\\model_scaling_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def train_model(
        self,
        model_size: str,
        num_epochs: int = 2,
        batch_size: int = 32,
        num_samples_per_epoch: int = 320
    ) -> Dict[str, Any]:
        """
        Train a model of specified size.
        
        Args:
            model_size: 'small', 'medium', 'large', or 'xlarge'
            num_epochs: Training epochs
            batch_size: Batch size
            num_samples_per_epoch: Samples for training per epoch
            
        Returns:
            Training metrics and results
        """
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model size: {model_size}")
        
        config = self.MODEL_CONFIGS[model_size]
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {config['description']} model | {num_epochs} epochs")
        logger.info(f"{'='*60}\n")
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ScalableTransformerModel(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers']
        )
        model = model.to(device)
        
        model_config = model.get_config()
        logger.info(f"Model config: {model_config}")
        logger.info(f"Parameters: {model_config['num_parameters']:,}")
        logger.info(f"Device: {device}\n")
        
        # Create optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Create dataset
        num_batches_per_epoch = max(1, num_samples_per_epoch // batch_size)
        train_input_ids = torch.randint(1, config['vocab_size'], (num_samples_per_epoch, 128))
        train_labels = torch.randint(0, config['vocab_size'], (num_samples_per_epoch, 128))
        
        dataset = TensorDataset(train_input_ids, train_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Memory tracking
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]
        
        # Training metrics
        metrics = {
            'model_size': model_size,
            'model_config': model_config,
            'training_config': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'num_samples': num_samples_per_epoch,
                'num_batches_per_epoch': num_batches_per_epoch
            },
            'epochs': [],
            'total_duration_sec': 0,
            'peak_memory_mb': 0,
            'avg_throughput_samples_per_sec': 0
        }
        
        training_start = time.time()
        epoch_times = []
        epoch_losses = []
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0
            batch_samples = 0
            
            for batch_idx, (input_ids, labels) in enumerate(loader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                
                # Forward pass
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, config['vocab_size']),
                    labels.view(-1)
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                batch_samples += input_ids.shape[0]
                
                if batch_idx % 5 == 0:
                    logger.info(f"Epoch {epoch+1} | Batch {batch_idx+1}/{num_batches_per_epoch} | Loss: {loss.item():.4f}")
            
            scheduler.step()
            
            # Compute epoch metrics
            epoch_duration = time.time() - epoch_start
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            throughput = batch_samples / epoch_duration if epoch_duration > 0 else 0.0
            
            epoch_times.append(epoch_duration)
            epoch_losses.append(avg_loss)
            
            metrics['epochs'].append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'duration_sec': epoch_duration,
                'num_batches': num_batches,
                'throughput_samples_per_sec': throughput,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {epoch_duration:.1f}s | Throughput: {throughput:.1f} samples/sec\n")
        
        total_duration = time.time() - training_start
        metrics['total_duration_sec'] = total_duration
        metrics['avg_epoch_duration_sec'] = np.mean(epoch_times)
        
        # Memory metrics
        peak_memory = tracemalloc.get_traced_memory()[1]
        metrics['peak_memory_mb'] = (peak_memory - initial_memory) / (1024 * 1024)
        metrics['avg_throughput_samples_per_sec'] = np.mean([e['throughput_samples_per_sec'] for e in metrics['epochs']])
        
        tracemalloc.stop()
        
        logger.info(f"Training complete!")
        logger.info(f"Total duration: {total_duration:.1f}s")
        logger.info(f"Avg throughput: {metrics['avg_throughput_samples_per_sec']:.1f} samples/sec")
        logger.info(f"Peak memory: {metrics['peak_memory_mb']:.1f} MB")
        logger.info(f"Loss progression: {epoch_losses}")
        
        return metrics
    
    def benchmark_all_sizes(
        self,
        num_epochs: int = 2,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Benchmark all model sizes.
        
        Args:
            num_epochs: Training epochs per size
            batch_size: Batch size
            
        Returns:
            Benchmark results for all sizes
        """
        logger.info(f"\n{'='*70}")
        logger.info("MODEL SCALING BENCHMARK - Testing all model sizes")
        logger.info(f"{'='*70}\n")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_config': {
                'num_epochs': num_epochs,
                'batch_size': batch_size
            },
            'model_sizes': {}
        }
        
        # Test each model size
        for model_size in ['small', 'medium', 'large']:
            try:
                metrics = self.train_model(
                    model_size=model_size,
                    num_epochs=num_epochs,
                    batch_size=batch_size
                )
                results['model_sizes'][model_size] = metrics
            except Exception as e:
                logger.error(f"Error training {model_size} model: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                results['model_sizes'][model_size] = {'error': str(e)}
        
        # Compute scaling metrics
        results['scaling_analysis'] = self._analyze_scaling(results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"model_scaling_benchmark_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nBenchmark saved to {output_file}")
        
        # Print summary
        self._print_scaling_summary(results)
        
        return results
    
    def _analyze_scaling(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scaling metrics."""
        analysis = {
            'memory_scaling': {},
            'throughput_scaling': {},
            'time_scaling': {},
            'efficiency_metrics': {}
        }
        
        model_sizes = results['model_sizes']
        baseline_metrics = model_sizes.get('small', {})
        
        if not baseline_metrics or 'error' in baseline_metrics:
            return analysis
        
        for size in ['small', 'medium', 'large']:
            if size not in model_sizes or 'error' in model_sizes[size]:
                continue
            
            metrics = model_sizes[size]
            baseline_params = baseline_metrics['model_config']['num_parameters']
            this_params = metrics['model_config']['num_parameters']
            
            analysis['memory_scaling'][size] = {
                'peak_memory_mb': metrics['peak_memory_mb'],
                'memory_per_param_bytes': (metrics['peak_memory_mb'] * 1024 * 1024) / this_params if this_params > 0 else 0
            }
            
            analysis['throughput_scaling'][size] = {
                'throughput_samples_per_sec': metrics['avg_throughput_samples_per_sec'],
                'relative_to_baseline': metrics['avg_throughput_samples_per_sec'] / baseline_metrics['avg_throughput_samples_per_sec'] if baseline_metrics['avg_throughput_samples_per_sec'] > 0 else 0
            }
            
            analysis['time_scaling'][size] = {
                'total_duration_sec': metrics['total_duration_sec'],
                'relative_to_baseline': metrics['total_duration_sec'] / baseline_metrics['total_duration_sec'] if baseline_metrics['total_duration_sec'] > 0 else 0
            }
            
            analysis['efficiency_metrics'][size] = {
                'parameter_count': this_params,
                'param_ratio_to_baseline': this_params / baseline_params,
                'time_per_param_sec': metrics['total_duration_sec'] / this_params if this_params > 0 else 0
            }
        
        return analysis
    
    def _print_scaling_summary(self, results: Dict[str, Any]):
        """Print scaling summary."""
        logger.info(f"\n{'='*70}")
        logger.info("SCALING SUMMARY")
        logger.info(f"{'='*70}\n")
        
        analysis = results.get('scaling_analysis', {})
        
        # Memory summary
        logger.info("Memory Usage:")
        for size in ['small', 'medium', 'large']:
            if size in analysis['memory_scaling']:
                mem = analysis['memory_scaling'][size]
                logger.info(f"  {size}: {mem['peak_memory_mb']:.1f} MB ({mem['memory_per_param_bytes']:.2e} B/param)")
        
        # Throughput summary
        logger.info("\nThroughput (samples/sec):")
        for size in ['small', 'medium', 'large']:
            if size in analysis['throughput_scaling']:
                tp = analysis['throughput_scaling'][size]
                logger.info(f"  {size}: {tp['throughput_samples_per_sec']:.1f} (relative: {tp['relative_to_baseline']:.2f}x)")
        
        # Time summary
        logger.info("\nTraining Time:")
        for size in ['small', 'medium', 'large']:
            if size in analysis['time_scaling']:
                t = analysis['time_scaling'][size]
                logger.info(f"  {size}: {t['total_duration_sec']:.1f}s (relative: {t['relative_to_baseline']:.2f}x)")
        
        logger.info(f"\n{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Model Scaling Testing Framework')
    parser.add_argument('--model_size', choices=['small', 'medium', 'large', 'xlarge'], 
                       default='small', help='Model size to train')
    parser.add_argument('--epochs', type=int, default=2, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--benchmark', action='store_true', help='Run full benchmark')
    parser.add_argument('--output_dir', default='s:\\Ryot\\model_scaling_results', help='Output directory')
    
    args = parser.parse_args()
    
    tester = ModelScalingTester(output_dir=args.output_dir)
    
    if args.benchmark:
        tester.benchmark_all_sizes(num_epochs=args.epochs, batch_size=args.batch_size)
    else:
        metrics = tester.train_model(
            model_size=args.model_size,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Save individual result
        output_file = tester.output_dir / f"model_{args.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"\nMetrics saved to {output_file}")


if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
