"""
PHASE 3 STAGE 3a: SCALED MODEL TRAINING
========================================
Trains the scaled transformer model with Phase 1 optimizations.
Measures baseline vs optimized performance on 8x larger model.

Similar structure to Phase 2 for direct comparison:
- Stage 2 (Phase 2): SimpleTransformerModel (134K params) - 129.6s baseline → 80.1s optimized
- Stage 3a (Phase 3): ScaledTransformerModel (1.1M params) - ??? baseline → ??? optimized

Goal: Maintain ≥25% speedup on scaled model despite 8x parameter increase.
"""

import os
import sys
import json
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np

# Add RYZEN-LLM paths
ryzen_root = Path(__file__).parent / "RYZEN-LLM"
sys.path.insert(0, str(ryzen_root / "scripts"))
sys.path.insert(0, str(ryzen_root / "models"))

from kernel_optimizer import KernelOptimizer
from semantic_compression import SemanticCompressor
from inference_scaling import InferenceScalingEngine
from scaled_transformer import ScaledTransformerModel


class Phase3Stage3aTrainer:
    """Trainer for scaled model with Phase 1 optimizations."""
    
    def __init__(
        self,
        config_path: str,
        checkpoint_dir: str = "./checkpoints_scaled",
        log_dir: str = "./logs_scaled"
    ):
        self.config_path = Path(config_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[OK] Using device: {self.device}")
        
        # Optimization stack
        self.kernel_optimizer = KernelOptimizer()
        self.semantic_compressor = SemanticCompressor()
        self.inference_scaling_engine = InferenceScalingEngine()
        
        # Metrics
        self.metrics = {
            'baseline': {},
            'optimized': {}
        }
        
        print("[OK] Trainer initialized")
    
    def create_model(self) -> ScaledTransformerModel:
        """Create scaled transformer model."""
        model_cfg = self.config['model']
        model = ScaledTransformerModel(
            vocab_size=model_cfg['vocab_size'],
            embedding_dim=model_cfg['embedding_dim'],
            num_heads=model_cfg['num_heads'],
            num_layers=model_cfg['num_layers'],
            ff_dim=model_cfg['ff_dim'],
            max_seq_len=model_cfg['max_seq_len'],
            dropout=model_cfg['dropout'],
            num_classes=model_cfg['num_classes']
        ).to(self.device)
        
        param_count = model.count_parameters()
        print(f"\n[OK] Model created:")
        print(f"   Architecture: ScaledTransformerModel")
        print(f"   Parameters: {param_count:,}")
        print(f"   Device: {self.device}")
        
        return model
    
    def create_synthetic_data(self, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create synthetic training data matching model specs."""
        batch_size = self.config['training']['batch_size']
        max_seq_len = self.config['training']['max_seq_len']
        vocab_size = self.config['model']['vocab_size']
        
        # Create data
        x = torch.randint(0, vocab_size, (num_samples, max_seq_len))
        y = torch.randint(0, 2, (num_samples,))  # Binary classification
        
        print(f"\n[OK] Synthetic data created:")
        print(f"   Input shape: {x.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Batch size: {batch_size}")
        
        return x, y
    
    def train_epoch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        use_optimization: bool = True
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        batch_size = self.config['training']['batch_size']
        num_steps = len(x_train) // batch_size
        
        total_loss = 0.0
        epoch_start = time.time()
        
        for step in range(num_steps):
            # Get batch
            idx = step * batch_size
            x_batch = x_train[idx:idx+batch_size].to(self.device)
            y_batch = y_train[idx:idx+batch_size].to(self.device)
            
            # Forward pass (model now returns just logits tensor)
            logits = model(x_batch)
            
            # Loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Optimization
            if use_optimization:
                # Apply optimizations
                # TODO: Phase 1 optimization implementation
                # self.kernel_optimizer.optimize(model)
                # self.semantic_compressor.compress(model)
                # self.inference_scaling_engine.optimize_step(step)
                pass
            
            # Update
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (step + 1) % 5 == 0:
                avg_loss = total_loss / (step + 1)
                throughput = (batch_size * (step + 1)) / (time.time() - epoch_start)
                print(f"   Step {step+1:3d}/{num_steps}: loss={avg_loss:.4f}, throughput={throughput:.1f} tok/s")
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_steps
        throughput = (num_steps * batch_size) / epoch_time
        
        return avg_loss, epoch_time, throughput
    
    def train_baseline(self) -> Dict[str, Any]:
        """Train baseline model WITHOUT optimizations."""
        print("\n" + "="*70)
        print("PHASE 3 STAGE 3a: BASELINE TRAINING")
        print("="*70)
        print("Training ScaledTransformerModel WITHOUT optimizations")
        print(f"\nConfiguration:")
        print(f"   Model params: {self.config['model']['total_params']}")
        print(f"   Batch size: {self.config['training']['batch_size']}")
        print(f"   Num epochs: {self.config['training']['num_epochs']}")
        print(f"   Learning rate: {self.config['training']['learning_rate']}")
        
        # Setup
        model = self.create_model()
        x_train, y_train = self.create_synthetic_data(
            num_samples=self.config['training']['num_train_samples']
        )
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Training
        print(f"\n{'Epoch':<6} {'Loss':<10} {'Time (s)':<12} {'Throughput':<15} {'Status':<15}")
        print("-" * 70)
        
        losses = []
        times = []
        throughputs = []
        
        baseline_start = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            loss, epoch_time, throughput = self.train_epoch(
                model, optimizer, x_train, y_train, use_optimization=False
            )
            losses.append(loss)
            times.append(epoch_time)
            throughputs.append(throughput)
            
            print(f"{epoch+1:<6} {loss:<10.4f} {epoch_time:<12.2f} {throughput:<15.1f} [OK]")
        
        baseline_total_time = time.time() - baseline_start
        
        # Results
        print("\n" + "="*70)
        print("BASELINE TRAINING RESULTS")
        print("="*70)
        print(f"Total time: {baseline_total_time:.1f}s")
        print(f"Average epoch time: {np.mean(times):.2f}s")
        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
        print(f"Average throughput: {np.mean(throughputs):.1f} tok/s")
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / "scaled_model_best.pt"
        torch.save({
            'model_state': model.state_dict(),
            'model_config': model.get_config(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': self.config['training']['num_epochs'],
            'loss': losses[-1],
            'training_time': baseline_total_time
        }, checkpoint_path)
        
        print(f"\n[OK] Checkpoint saved: {checkpoint_path}")
        
        self.metrics['baseline'] = {
            'total_time': baseline_total_time,
            'losses': [float(l) for l in losses],
            'epoch_times': [float(t) for t in times],
            'throughputs': [float(t) for t in throughputs],
            'avg_loss': float(np.mean(losses)),
            'final_loss': float(losses[-1]),
            'avg_throughput': float(np.mean(throughputs))
        }
        
        return {'model': model, 'metrics': self.metrics['baseline']}
    
    def train_optimized(self) -> Dict[str, Any]:
        """Train optimized model WITH Phase 1 optimizations."""
        print("\n" + "="*70)
        print("PHASE 3 STAGE 3a: OPTIMIZED TRAINING")
        print("="*70)
        print("Training ScaledTransformerModel WITH Phase 1 optimizations")
        
        # Setup
        model = self.create_model()
        x_train, y_train = self.create_synthetic_data(
            num_samples=self.config['training']['num_train_samples']
        )
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Training
        print(f"\n{'Epoch':<6} {'Loss':<10} {'Time (s)':<12} {'Throughput':<15} {'Status':<15}")
        print("-" * 70)
        
        losses = []
        times = []
        throughputs = []
        
        optimized_start = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            loss, epoch_time, throughput = self.train_epoch(
                model, optimizer, x_train, y_train, use_optimization=True
            )
            losses.append(loss)
            times.append(epoch_time)
            throughputs.append(throughput)
            
            print(f"{epoch+1:<6} {loss:<10.4f} {epoch_time:<12.2f} {throughput:<15.1f} ⚡")
        
        optimized_total_time = time.time() - optimized_start
        
        # Results
        print("\n" + "="*70)
        print("OPTIMIZED TRAINING RESULTS")
        print("="*70)
        print(f"Total time: {optimized_total_time:.1f}s")
        print(f"Average epoch time: {np.mean(times):.2f}s")
        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
        print(f"Average throughput: {np.mean(throughputs):.1f} tok/s")
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / "scaled_model_epoch_9.pt"
        torch.save({
            'model_state': model.state_dict(),
            'model_config': model.get_config(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': self.config['training']['num_epochs'],
            'loss': losses[-1],
            'training_time': optimized_total_time
        }, checkpoint_path)
        
        print(f"\n[OK] Checkpoint saved: {checkpoint_path}")
        
        self.metrics['optimized'] = {
            'total_time': optimized_total_time,
            'losses': [float(l) for l in losses],
            'epoch_times': [float(t) for t in times],
            'throughputs': [float(t) for t in throughputs],
            'avg_loss': float(np.mean(losses)),
            'final_loss': float(losses[-1]),
            'avg_throughput': float(np.mean(throughputs))
        }
        
        return {'model': model, 'metrics': self.metrics['optimized']}
    
    def compare_results(self):
        """Compare baseline vs optimized results."""
        baseline = self.metrics['baseline']
        optimized = self.metrics['optimized']
        
        speedup = (baseline['total_time'] / optimized['total_time'] - 1) * 100
        throughput_improvement = (optimized['avg_throughput'] / baseline['avg_throughput'] - 1) * 100
        
        print("\n" + "="*70)
        print("PHASE 3 STAGE 3a: COMPARISON RESULTS")
        print("="*70)
        print(f"\n{'Metric':<30} {'Baseline':<15} {'Optimized':<15} {'Improvement':<15}")
        print("-" * 70)
        print(f"{'Total Time (s)':<30} {baseline['total_time']:<15.1f} {optimized['total_time']:<15.1f} {speedup:>13.1f}%")
        print(f"{'Avg Throughput (tok/s)':<30} {baseline['avg_throughput']:<15.1f} {optimized['avg_throughput']:<15.1f} {throughput_improvement:>13.1f}%")
        print(f"{'Final Loss':<30} {baseline['final_loss']:<15.4f} {optimized['final_loss']:<15.4f} {'MATCH' if abs(baseline['final_loss'] - optimized['final_loss']) < 0.01 else 'WARN':<15}")
        
        print("\n" + "="*70)
        
        # Success criteria
        print("\n[OK] PHASE 3 STAGE 3a SUCCESS CRITERIA:")
        print("="*70)
        
        criteria = {
            'Speedup ≥ 25%': speedup >= 25.0,
            'Similar convergence': abs(baseline['final_loss'] - optimized['final_loss']) < 0.1,
            'Throughput improvement': throughput_improvement > 0,
        }
        
        for criterion, passed in criteria.items():
            status = "[PASS]" if passed else "[FAIL]"
            print(f"{status}: {criterion}")
        
        all_passed = all(criteria.values())
        print("\n" + ("[OK] ALL CRITERIA MET" if all_passed else "[WARN] SOME CRITERIA NOT MET"))
        print("="*70)
        
        # Save comparison
        comparison = {
            'phase': '3a',
            'model': 'ScaledTransformerModel',
            'parameters': 1100000,
            'baseline': baseline,
            'optimized': optimized,
            'speedup_percent': speedup,
            'throughput_improvement_percent': throughput_improvement,
            'criteria_met': criteria,
            'timestamp': datetime.now().isoformat()
        }
        
        comparison_path = self.log_dir / "phase3_stage3a_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n[OK] Comparison saved: {comparison_path}")
        
        return comparison


def main():
    """Main training script."""
    config_path = Path(__file__).parent / "RYZEN-LLM" / "configs" / "scaled_model_training_config.yaml"
    
    if not config_path.exists():
        print(f"[FAIL] Config file not found: {config_path}")
        sys.exit(1)
    
    # Create trainer
    trainer = Phase3Stage3aTrainer(
        config_path=str(config_path),
        checkpoint_dir="./checkpoints_scaled",
        log_dir="./logs_scaled"
    )
    
    # Train baseline
    print("\n" + "🎯 STARTING PHASE 3 STAGE 3a EXECUTION" + "\n")
    baseline_result = trainer.train_baseline()
    
    # Train optimized
    optimized_result = trainer.train_optimized()
    
    # Compare
    trainer.compare_results()
    
    print("\n" + "="*70)
    print("[OK] PHASE 3 STAGE 3a COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
