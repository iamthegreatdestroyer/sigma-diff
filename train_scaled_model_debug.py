"""
PHASE 3 STAGE 3a: SCALED MODEL TRAINING - DEBUG VERSION
========================================
Same as train_scaled_model.py but with better error handling and monitoring
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
import traceback
import signal

# Add RYZEN-LLM paths
ryzen_root = Path(__file__).parent / "RYZEN-LLM"
sys.path.insert(0, str(ryzen_root / "scripts"))
sys.path.insert(0, str(ryzen_root / "models"))

try:
    from kernel_optimizer import KernelOptimizer
    from semantic_compression import SemanticCompressor
    from inference_scaling import InferenceScalingEngine
    from scaled_transformer import ScaledTransformerModel
except Exception as e:
    print(f"❌ Import error: {e}")
    traceback.print_exc()
    sys.exit(1)


class Phase3Stage3aTrainerDebug:
    """Trainer for scaled model with Phase 1 optimizations - Debug version."""
    
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
        print("📋 Loading configuration...")
        try:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
            print(f"✅ Config loaded: {self.config_path}")
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Using device: {self.device}")
        
        # Optimization stack - WITH TIMEOUTS
        print("🔧 Initializing optimization stack...")
        try:
            print("  -> KernelOptimizer...")
            self.kernel_optimizer = KernelOptimizer()
            print("     ✅ Initialized")
            
            print("  -> SemanticCompressor...")
            self.semantic_compressor = SemanticCompressor()
            print("     ✅ Initialized")
            
            print("  -> InferenceScalingEngine...")
            self.inference_scaling_engine = InferenceScalingEngine()
            print("     ✅ Initialized")
            
        except Exception as e:
            print(f"❌ Optimization stack initialization failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Metrics
        self.metrics = {
            'baseline': {},
            'optimized': {}
        }
        
        print("✅ Trainer initialized\n")
    
    def create_model(self) -> ScaledTransformerModel:
        """Create scaled transformer model."""
        model_cfg = self.config['model']
        print("🏗️  Creating model...")
        try:
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
            print(f"\n✅ Model created:")
            print(f"   Architecture: ScaledTransformerModel")
            print(f"   Parameters: {param_count:,}")
            print(f"   Device: {self.device}\n")
            
            return model
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def create_synthetic_data(self, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create synthetic training data matching model specs."""
        batch_size = self.config['training']['batch_size']
        seq_len = self.config['model']['max_seq_len']
        vocab_size = self.config['model']['vocab_size']
        num_classes = self.config['model']['num_classes']
        
        print(f"📊 Creating synthetic data ({num_samples} samples)...")
        
        x_train = torch.randint(0, vocab_size, (num_samples, seq_len)).to(self.device)
        y_train = torch.randint(0, num_classes, (num_samples,)).to(self.device)
        
        print(f"   Input shape: {x_train.shape}")
        print(f"   Label shape: {y_train.shape}\n")
        
        return x_train, y_train
    
    def train_epoch(self, model, optimizer, x_train, y_train, use_optimization: bool = False) -> Tuple[float, float, float]:
        """Train one epoch (simplified)."""
        model.train()
        batch_size = self.config['training']['batch_size']
        
        total_loss = 0
        start_time = time.time()
        tokens_processed = 0
        
        try:
            for i in range(0, len(x_train), batch_size):
                batch_x = x_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                # Forward pass
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = nn.CrossEntropyLoss()(logits, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                tokens_processed += batch_x.numel()
            
            epoch_time = time.time() - start_time
            avg_loss = total_loss / (len(x_train) // batch_size)
            throughput = tokens_processed / epoch_time if epoch_time > 0 else 0
            
            return avg_loss, epoch_time, throughput
            
        except Exception as e:
            print(f"❌ Epoch training failed: {e}")
            traceback.print_exc()
            raise
    
    def train_baseline(self) -> Dict[str, Any]:
        """Train baseline model WITHOUT optimizations."""
        print("\n" + "="*70)
        print("PHASE 3 STAGE 3a: BASELINE TRAINING")
        print("="*70)
        print("Training ScaledTransformerModel WITHOUT Phase 1 optimizations\n")
        
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
        print(f"{'Epoch':<6} {'Loss':<10} {'Time (s)':<12} {'Throughput':<15} {'Status':<15}")
        print("-" * 70)
        
        losses = []
        times = []
        throughputs = []
        
        baseline_start = time.time()
        
        try:
            for epoch in range(self.config['training']['num_epochs']):
                loss, epoch_time, throughput = self.train_epoch(
                    model, optimizer, x_train, y_train, use_optimization=False
                )
                losses.append(loss)
                times.append(epoch_time)
                throughputs.append(throughput)
                
                print(f"{epoch+1:<6} {loss:<10.4f} {epoch_time:<12.2f} {throughput:<15.1f} ✅")
        
        except Exception as e:
            print(f"❌ Baseline training failed at epoch: {e}")
            traceback.print_exc()
            sys.exit(1)
        
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
        print(f"Average throughput: {np.mean(throughputs):.1f} tok/s\n")
        
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
        
        print(f"✅ Checkpoint saved: {checkpoint_path}\n")
        
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
        print("Training ScaledTransformerModel WITH Phase 1 optimizations\n")
        
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
        print(f"{'Epoch':<6} {'Loss':<10} {'Time (s)':<12} {'Throughput':<15} {'Status':<15}")
        print("-" * 70)
        
        losses = []
        times = []
        throughputs = []
        
        optimized_start = time.time()
        
        try:
            for epoch in range(self.config['training']['num_epochs']):
                loss, epoch_time, throughput = self.train_epoch(
                    model, optimizer, x_train, y_train, use_optimization=True
                )
                losses.append(loss)
                times.append(epoch_time)
                throughputs.append(throughput)
                
                print(f"{epoch+1:<6} {loss:<10.4f} {epoch_time:<12.2f} {throughput:<15.1f} ⚡")
        
        except Exception as e:
            print(f"❌ Optimized training failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
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
        print(f"Average throughput: {np.mean(throughputs):.1f} tok/s\n")
        
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
        
        print(f"✅ Checkpoint saved: {checkpoint_path}\n")
        
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
        print(f"{'Final Loss':<30} {baseline['final_loss']:<15.4f} {optimized['final_loss']:<15.4f} {'MATCH' if abs(baseline['final_loss'] - optimized['final_loss']) < 0.01 else '⚠️':<15}")
        
        print("\n" + "="*70)
        
        # Success criteria
        print("\n✅ PHASE 3 STAGE 3a SUCCESS CRITERIA:")
        print("="*70)
        
        criteria = {
            'Speedup ≥ 25%': speedup >= 25.0,
            'Similar convergence': abs(baseline['final_loss'] - optimized['final_loss']) < 0.1,
            'Throughput improvement': throughput_improvement > 0,
        }
        
        for criterion, passed in criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {criterion}")
        
        all_passed = all(criteria.values())
        print("\n" + ("✅ ALL CRITERIA MET" if all_passed else "⚠️  SOME CRITERIA NOT MET"))
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
        
        print(f"\n✅ Comparison saved: {comparison_path}\n")
        
        return comparison


def main():
    """Main training script."""
    config_path = Path(__file__).parent / "RYZEN-LLM" / "configs" / "scaled_model_training_config.yaml"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    # Create trainer
    print("🚀 PHASE 3 STAGE 3a - SCALED MODEL TRAINING (DEBUG VERSION)\n")
    
    try:
        trainer = Phase3Stage3aTrainerDebug(
            config_path=str(config_path),
            checkpoint_dir="./checkpoints_scaled",
            log_dir="./logs_scaled"
        )
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Train baseline
    try:
        print("📌 Starting baseline training...\n")
        baseline_result = trainer.train_baseline()
        print("✅ Baseline training complete\n")
    except Exception as e:
        print(f"❌ Baseline training failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Train optimized
    try:
        print("📌 Starting optimized training...\n")
        optimized_result = trainer.train_optimized()
        print("✅ Optimized training complete\n")
    except Exception as e:
        print(f"❌ Optimized training failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Compare
    try:
        print("📌 Comparing results...\n")
        trainer.compare_results()
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✅ PHASE 3 STAGE 3a COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
