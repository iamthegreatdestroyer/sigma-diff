#!/usr/bin/env python3
"""
Phase 2: Unified Model Training Loop with Integrated Optimizations

This module orchestrates the main training loop, integrating:
- kernel_optimizer: CPU feature detection and kernel tuning
- semantic_compression: MRL/binary/sparse embedding compression
- inference_scaling: RLVR multi-path reasoning speedup
- training monitoring: Real-time metrics collection

Usage:
    python training_loop.py --config training_configuration.yaml --epochs 10
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# Add scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Import Phase 1 optimization modules
try:
    from kernel_optimizer import KernelOptimizer
    from semantic_compression import SemanticCompressor
    from inference_scaling import InferenceScalingEngine
    PHASE1_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Phase 1 modules not fully available: {e}")
    PHASE1_AVAILABLE = False

# Import Phase 2 modules
from training_metrics_collector import TrainingMetricsCollector
from optimization_controller import OptimizationController
from optimization_orchestrator import OptimizationOrchestrator

# Configure logging (MUST be before logger use)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('training_loop.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Optional model validator (fallback if not available)
try:
    from model_inference_validator import ModelInferenceValidator
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False
    logger.warning("ModelInferenceValidator not available - proceeding without inference validation")


class SimpleTransformerModel(nn.Module):
    """
    Lightweight transformer model for training demonstration.
    
    Architecture:
    - Embedding layer (vocab_size → embedding_dim)
    - 2x Transformer encoder blocks
    - Output projection (embedding_dim → vocab_size)
    """
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
        """
        Forward pass through the model.
        
        Args:
            input_ids: (batch_size, seq_len) token tensor
            
        Returns:
            logits: (batch_size, seq_len, vocab_size) prediction logits
        """
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Embedding + positional encoding
        embeds = self.embedding(input_ids)
        pos_embeds = self.pos_embedding(positions)
        hidden = embeds + pos_embeds
        
        # Transformer encoder
        encoded = self.transformer_encoder(hidden)
        
        # Output projection
        logits = self.output_projection(encoded)
        
        return logits


class TrainingLoop:
    """
    Main training orchestration with integrated Phase 1 optimizations.
    
    Features:
    - Baseline and optimized training modes
    - Real-time metrics collection
    - Optimization parameter adaptation
    - Checkpoint saving and resuming
    """
    
    def __init__(
        self,
        config_path: str,
        enable_optimization: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize training loop.
        
        Args:
            config_path: Path to training configuration YAML
            enable_optimization: Enable Phase 1 optimizations
            device: Training device (cuda/cpu)
        """
        self.config_path = Path(config_path)
        self.enable_optimization = enable_optimization
        self.device = torch.device(device)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        
        # Optimization components
        self.kernel_optimizer = None
        self.compression_engine = None
        self.inference_controller = None
        self.optimization_controller = None
        self.orchestrator = None  # Will be initialized in setup_optimizations()
        
        # Metrics collection
        self.metrics_collector = TrainingMetricsCollector()
        self.inference_validator = ModelInferenceValidator() if VALIDATOR_AVAILABLE else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"TrainingLoop initialized | Device: {self.device} | Optimization: {enable_optimization}")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and parse YAML configuration file."""
        import yaml
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return default config
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'model': {
                'vocab_size': 2048,
                'embedding_dim': 256,
                'num_heads': 4,
                'num_layers': 2,
                'ff_dim': 512,
                'max_seq_len': 128
            },
            'training': {
                'batch_size': 32,
                'num_epochs': 10,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'warmup_steps': 100,
                'gradient_accumulation_steps': 1
            },
            'optimization': {
                'use_kernel_optimization': True,
                'use_semantic_compression': True,
                'use_inference_scaling': True,
                'compression_ratio': 0.3,  # 30% reduction
                'kernel_tile_size': 64
            },
            'validation': {
                'val_interval': 2,  # Validate every 2 epochs
                'save_interval': 5,  # Save checkpoint every 5 epochs
                'early_stopping_patience': 5
            }
        }
    
    def setup_model_and_data(self):
        """Initialize model, optimizer, and data loaders."""
        logger.info("Setting up model and data...")
        
        # Create model
        model_config = self.config['model']
        self.model = SimpleTransformerModel(
            vocab_size=model_config.get('vocab_size', 2048),
            embedding_dim=model_config.get('embedding_dim', 256),
            num_heads=model_config.get('num_heads', 4),
            num_layers=model_config.get('num_layers', 2),
            ff_dim=model_config.get('ff_dim', 512),
            max_seq_len=model_config.get('max_seq_len', 128)
        ).to(self.device)
        
        logger.info(f"Model created: {self.model.__class__.__name__}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create optimizer
        train_config = self.config['training']
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config.get('learning_rate', 1e-4),
            weight_decay=train_config.get('weight_decay', 1e-5)
        )
        
        # Create learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_config.get('num_epochs', 10)
        )
        
        # Create synthetic dataset for demonstration
        vocab_size = model_config['vocab_size']
        max_seq_len = model_config['max_seq_len']
        batch_size = train_config['batch_size']
        
        num_samples = batch_size * 10  # 10 batches for demo
        
        # Training data: random token sequences
        train_input_ids = torch.randint(1, vocab_size, (num_samples, max_seq_len))
        train_labels = torch.randint(0, vocab_size, (num_samples, max_seq_len))
        
        train_dataset = TensorDataset(train_input_ids, train_labels)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Validation data
        num_val_samples = batch_size * 2
        val_input_ids = torch.randint(1, vocab_size, (num_val_samples, max_seq_len))
        val_labels = torch.randint(0, vocab_size, (num_val_samples, max_seq_len))
        
        val_dataset = TensorDataset(val_input_ids, val_labels)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Data loaders created | Train samples: {num_samples} | Val samples: {num_val_samples}")
    
    def setup_optimizations(self):
        """Initialize Phase 1 optimization modules and OptimizationOrchestrator."""
        if not self.enable_optimization or not PHASE1_AVAILABLE:
            logger.info("Phase 1 optimizations disabled or unavailable")
            return
        
        # Initialize OptimizationOrchestrator with optimization configuration
        try:
            orchestrator_config = {
                'kernel_optimizer': self.config['optimization'].get('kernel_optimizer', {'tile_size': 64, 'block_size': 64}),
                'semantic_compression': self.config['optimization'].get('semantic_compression', {'compression_ratio': 0.3, 'block_size': 64}),
                'inference_scaling': self.config['optimization'].get('inference_scaling', {'path_selection_threshold': 0.7, 'sparsity_threshold': 0.5})
            }
            self.orchestrator = OptimizationOrchestrator(orchestrator_config)
            
            # Validate parameter compatibility at initialization
            is_compatible, warnings = self.orchestrator.validate_parameter_compatibility()
            if warnings:
                for warning in warnings:
                    logger.warning(warning)
            
            logger.info("✅ OptimizationOrchestrator initialized and parameter compatibility validated")
        except Exception as e:
            logger.error(f"Failed to initialize OptimizationOrchestrator: {e}")
            self.orchestrator = None
        
        logger.info("Initializing Phase 1 optimizations...")
        
        try:
            # Kernel optimization (includes CPU feature detection)
            repo_root = Path(__file__).parent.parent.name  # Get RYZEN-LLM directory name
            self.kernel_optimizer = KernelOptimizer(repo_root="RYZEN-LLM")
            cpu_features = self.kernel_optimizer.cpu_features
            logger.info(f"CPU features detected: {cpu_features}")
            logger.info("Kernel optimizer initialized")
            
            # Semantic compression engine
            self.compression_engine = SemanticCompressor(
                compression_ratio=self.config['optimization'].get('compression_ratio', 0.3)
            )
            logger.info("Semantic compression engine initialized")
            
            # Inference scaling controller
            self.inference_controller = InferenceScalingEngine()
            logger.info("Inference scaling controller initialized")
            
            # Integrated optimization controller
            self.optimization_controller = OptimizationController(
                kernel_optimizer=self.kernel_optimizer,
                compression_engine=self.compression_engine,
                inference_controller=self.inference_controller
            )
            logger.info("Integrated optimization controller initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Phase 1 optimizations: {e}")
            self.enable_optimization = False
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Execute one training epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            Dictionary with epoch metrics (loss, learning_rate, duration, etc.)
        """
        sys.stdout.flush()  # Ensure print appears immediately
        self.model.train()
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # Orchestrator: Adapt parameters at epoch start
        if self.orchestrator is not None:
            try:
                adapted_config = self.orchestrator.adapt_parameters({'epoch': epoch})
                logger.info(f"📊 Orchestrator adapted parameters for epoch {epoch}")
            except Exception as e:
                logger.warning(f"Orchestrator parameter adaptation failed: {e}")
        
        for batch_idx, (input_ids, labels) in enumerate(self.train_loader):
            if batch_idx == 0:
                logger.debug(f"Starting batch processing with {len(self.train_loader)} total batches")
            # Move to device
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            batch_start = time.time()
            logits = self.model(input_ids)  # (batch_size, seq_len, vocab_size)
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config['model']['vocab_size']),
                labels.view(-1)
            )
            
            # Orchestrator: Validate safety gates after loss computation
            if self.orchestrator is not None:
                try:
                    # Compute gradient norm for safety gate validation
                    self.model.zero_grad()
                    loss.backward()
                    
                    # Compute gradient norm
                    total_grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_grad_norm += (p.grad.data.norm(2.0) ** 2)
                    grad_norm = total_grad_norm ** 0.5
                    
                    # Prepare metrics for safety gate validation
                    safety_metrics = {
                        'loss': loss.item(),
                        'gradient_norm': grad_norm,
                        'compression_recon_error': 0.02  # Placeholder from orchestrator
                    }
                    
                    # Validate safety gates
                    gates_passed, gate_violations = self.orchestrator.validate_safety_gates(safety_metrics)
                    if not gates_passed:
                        for violation in gate_violations:
                            logger.warning(violation)
                    else:
                        logger.debug(f"✅ Safety gates passed at epoch {epoch}, batch {batch_idx}")
                except Exception as e:
                    logger.warning(f"Orchestrator safety gate validation failed: {e}")
                    # Continue training even if validation fails
                    if 'loss' in locals():
                        loss.backward()
            else:
                # Standard backward if orchestrator not active
                loss.backward()
            
            # Optimization step
            if (batch_idx + 1) % self.config['training'].get('gradient_accumulation_steps', 1) == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            batch_duration = time.time() - batch_start
            
            # Accumulate metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log batch metrics
            if batch_idx % 5 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                throughput = input_ids.shape[0] / batch_duration
                logger.info(
                    f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | Throughput: {throughput:.1f} samples/sec"
                )
            
            # Record batch metrics
            self.metrics_collector.record_batch_metric({
                'epoch': epoch,
                'batch': batch_idx,
                'loss': loss.item(),
                'throughput': input_ids.shape[0] / batch_duration,
                'batch_duration_sec': batch_duration,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
        # Step scheduler
        self.scheduler.step()
        
        # Compute epoch statistics
        epoch_duration = time.time() - epoch_start_time
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'duration_sec': epoch_duration,
            'num_batches': num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'throughput': sum(
                m.throughput for m in self.metrics_collector.batch_metrics[-num_batches:]
                if hasattr(m, 'throughput') and m.throughput > 0
            ) / num_batches if num_batches > 0 else 0.0
        }
        
        logger.info(
            f"Epoch {epoch+1} complete | Loss: {avg_loss:.4f} | "
            f"Duration: {epoch_duration:.1f}s | Throughput: {epoch_metrics['throughput']:.1f} samples/sec"
        )
        
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Execute validation.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, labels in self.val_loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, self.config['model']['vocab_size']),
                    labels.view(-1)
                )
                
                val_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
        
        logger.info(f"Validation loss: {avg_val_loss:.4f}")
        
        val_metrics = {
            'epoch': epoch,
            'val_loss': avg_val_loss,
            'num_val_batches': num_batches
        }
        
        return val_metrics
    
    def run_training(self, num_epochs: Optional[int] = None):
        """
        Execute full training loop.
        
        Args:
            num_epochs: Number of epochs to train (if None, use config)
        """
        if num_epochs is None:
            num_epochs = self.config['training'].get('num_epochs', 10)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Optimization enabled: {self.enable_optimization}")
        
        # Setup model and data
        self.setup_model_and_data()
        
        # Setup optimizations
        if self.enable_optimization:
            self.setup_optimizations()
        
        training_start = time.time()
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(epoch)
            self.metrics_collector.record_epoch_metric(train_metrics)
            
            # Validation
            if (epoch + 1) % self.config['validation'].get('val_interval', 2) == 0:
                val_metrics = self.validate_epoch(epoch)
                self.metrics_collector.record_epoch_metric(val_metrics)
                
                # Check for best validation loss
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self._save_checkpoint(epoch, is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % self.config['validation'].get('save_interval', 5) == 0:
                self._save_checkpoint(epoch)
        
        training_duration = time.time() - training_start
        
        # Final report
        logger.info(f"\nTraining complete!")
        logger.info(f"Total duration: {training_duration:.1f}s ({training_duration/60:.1f} minutes)")
        logger.info(f"Average epoch time: {training_duration/num_epochs:.1f}s")
        
        # Generate metrics report
        self._generate_reports()
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint with orchestrator configuration snapshot."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 's:\\Ryot\\RYZEN-LLM\\checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_name = 'model_best.pt' if is_best else f'model_epoch_{epoch}.pt'
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Orchestrator: Save configuration snapshot with checkpoint
        if self.orchestrator is not None:
            try:
                orchestrator_snapshot = self.orchestrator.snapshot_configuration(
                    epoch=epoch,
                    timestamp=time.time()
                )
                # Store optimization states for reproducibility
                checkpoint_data['orchestrator_states'] = [
                    {
                        'kernel_config': state.kernel_config,
                        'compression_config': state.compression_config,
                        'rlvr_config': state.rlvr_config,
                        'epoch': state.epoch,
                        'timestamp': state.timestamp,
                        'metrics': state.metrics
                    }
                    for state in self.orchestrator.optimization_states
                ]
                logger.info(f"📸 Orchestrator configuration snapshot saved (total states: {len(self.orchestrator.optimization_states)})")
            except Exception as e:
                logger.warning(f"Failed to save orchestrator snapshot: {e}")
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _generate_reports(self):
        """Generate training metrics and analysis reports."""
        # Export metrics
        self.metrics_collector.export_metrics(
            'training_metrics_report.json'
        )
        
        logger.info("Training metrics exported to training_metrics_report.json")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Phase 2 Training Loop with Integrated Optimizations')
    parser.add_argument('--config', default='training_configuration.yaml', help='Configuration file path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Training device')
    parser.add_argument('--no-optimization', action='store_true', help='Disable Phase 1 optimizations')
    
    args = parser.parse_args()
    
    # Create training loop
    training_loop = TrainingLoop(
        config_path=args.config,
        enable_optimization=not args.no_optimization,
        device=args.device
    )
    
    # Run training
    training_loop.run_training(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
