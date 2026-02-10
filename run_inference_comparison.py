#!/usr/bin/env python3
"""
Phase 2 Stage 2c: Comparative Inference Analysis
Compares baseline vs optimized model inference performance.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "RYZEN-LLM" / "scripts"))

import torch
import torch.nn as nn
from model_inference_validator import ModelInferenceValidator

# Import model architecture
from training_loop import SimpleTransformerModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint_model(checkpoint_path: str, device: str = "cpu") -> Optional[nn.Module]:
    """Load model from checkpoint with state dict."""
    try:
        # Load checkpoint dictionary
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Checkpoint keys: {checkpoint.keys()}")
        
        # Extract config from checkpoint
        config = checkpoint.get('config', {})
        
        # Map checkpoint config to SimpleTransformerModel constructor parameters
        # Checkpoint uses: vocab_size, embedding_dim, num_heads, num_layers, ff_dim, max_seq_len (from YAML)
        # OR potentially: vocab_size, hidden_size, num_heads, num_layers, intermediate_size, max_sequence_length
        
        # Try to get model section if it exists, otherwise use config root
        model_section = config.get('model', config)
        
        # Create model architecture with proper parameter mapping
        model_config = {
            'vocab_size': model_section.get('vocab_size', 2048),
            'embedding_dim': model_section.get('embedding_dim', model_section.get('hidden_size', 256)),
            'num_heads': model_section.get('num_heads', 4),
            'num_layers': model_section.get('num_layers', 2),
            'ff_dim': model_section.get('ff_dim', model_section.get('intermediate_size', 512)),
            'max_seq_len': model_section.get('max_seq_len', model_section.get('max_sequence_length', 128))
        }
        
        logger.info(f"Creating model with config: {model_config}")
        model = SimpleTransformerModel(**model_config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model state dict loaded successfully")
        
        model.eval()
        model.to(device)
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_best_checkpoint(checkpoints_dir: Path, pattern: str = "model_best.pt") -> Optional[Path]:
    """Find best checkpoint in directory."""
    checkpoint_file = checkpoints_dir / pattern
    if checkpoint_file.exists():
        logger.info(f"Found checkpoint: {checkpoint_file}")
        return checkpoint_file
    
    logger.warning(f"Checkpoint not found: {checkpoint_file}")
    return None


def main():
    """Run comparative inference analysis."""
    
    logger.info("=" * 70)
    logger.info("PHASE 2 STAGE 2c: COMPARATIVE INFERENCE ANALYSIS")
    logger.info("=" * 70)
    
    # Find checkpoints
    checkpoints_dir = Path(__file__).parent / "RYZEN-LLM" / "checkpoints"
    
    # Try to find baseline checkpoint (model_best.pt from baseline training)
    baseline_ckpt = find_best_checkpoint(checkpoints_dir, "model_best.pt")
    if not baseline_ckpt:
        logger.error("Baseline checkpoint not found!")
        return 1
    
    # Try to find optimized checkpoint (model_epoch_9.pt from optimized training)
    optimized_ckpt = find_best_checkpoint(checkpoints_dir, "model_epoch_9.pt")
    if not optimized_ckpt:
        logger.warning("Optimized checkpoint not found, using model_best.pt")
        optimized_ckpt = baseline_ckpt
    
    logger.info(f"Baseline checkpoint: {baseline_ckpt}")
    logger.info(f"Optimized checkpoint: {optimized_ckpt}")
    
    # Initialize validator
    validator = ModelInferenceValidator(output_dir="reports")
    
    # Load models
    logger.info("\nLoading models...")
    baseline_model = load_checkpoint_model(str(baseline_ckpt), device="cpu")
    optimized_model = load_checkpoint_model(str(optimized_ckpt), device="cpu")
    
    if baseline_model is None or optimized_model is None:
        logger.error("Failed to load models")
        return 1
    
    # Create test data
    logger.info("Creating test data...")
    # Use correct model dimensions: vocab_size=2048, max_seq_len=128
    test_data = torch.randint(0, 2048, (16, 128))  # 16 samples, max 128 seq length
    
    # Run validations
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE INFERENCE VALIDATION (5 runs)")
    logger.info("=" * 70)
    validator.validate_baseline(baseline_model, test_data, batch_size=1, device="cpu", num_runs=5)
    
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZED INFERENCE VALIDATION (5 runs)")
    logger.info("=" * 70)
    validator.validate_optimized(optimized_model, test_data, batch_size=1, device="cpu", num_runs=5)
    
    # Generate and save report
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING COMPARATIVE REPORT")
    logger.info("=" * 70)
    
    report = validator.generate_report()
    report_path = validator.save_report("inference_validation_report.json")
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("INFERENCE VALIDATION SUMMARY")
    logger.info("=" * 70)
    
    baseline_ttft = report['baseline']['ttft_mean_ms']
    baseline_throughput = report['baseline']['throughput_mean_tokens_sec']
    optimized_ttft = report['optimized']['ttft_mean_ms']
    optimized_throughput = report['optimized']['throughput_mean_tokens_sec']
    ttft_speedup = report['speedups']['ttft_speedup']
    throughput_improvement = report['speedups']['throughput_improvement']
    
    logger.info(f"\nBASELINE:")
    logger.info(f"  TTFT: {baseline_ttft:.2f}ms")
    logger.info(f"  Throughput: {baseline_throughput:.2f} tokens/sec")
    logger.info(f"  Memory: {report['baseline']['peak_memory_mb']:.2f}MB")
    logger.info(f"  Success Rate: {report['baseline']['success_rate']*100:.1f}%")
    
    logger.info(f"\nOPTIMIZED:")
    logger.info(f"  TTFT: {optimized_ttft:.2f}ms")
    logger.info(f"  Throughput: {optimized_throughput:.2f} tokens/sec")
    logger.info(f"  Memory: {report['optimized']['peak_memory_mb']:.2f}MB")
    logger.info(f"  Success Rate: {report['optimized']['success_rate']*100:.1f}%")
    
    logger.info(f"\nSPEEDUPS:")
    logger.info(f"  TTFT Speedup: {ttft_speedup:.2f}x {'✓ EXCEEDS TARGET (2.5-3.5x)' if 2.5 <= ttft_speedup <= 3.5 else '⚠ BELOW TARGET' if ttft_speedup < 2.5 else '✓ EXCEEDS RANGE'}")
    logger.info(f"  Throughput Improvement: {throughput_improvement:.2f}x {'✓ IN TARGET (1.6-2.4x)' if 1.6 <= throughput_improvement <= 2.4 else '⚠ BELOW TARGET' if throughput_improvement < 1.6 else '✓ EXCEEDS RANGE'}")
    logger.info(f"  Memory Reduction: {report['speedups']['memory_reduction_percent']:.1f}%")
    
    logger.info(f"\nPHASE 1 TARGETS:")
    logger.info(f"  TTFT Target: {report['phase1_comparison']['ttft_target_ms']:.2f}ms → Achieved: {optimized_ttft:.2f}ms ({report['phase1_comparison']['ttft_vs_target']:.2f}x target)")
    logger.info(f"  Throughput Target: {report['phase1_comparison']['throughput_target_tokens_sec']:.2f} tok/s → Achieved: {optimized_throughput:.2f} tok/s ({report['phase1_comparison']['throughput_vs_target']:.2f}x target)")
    
    logger.info(f"\nReport saved to: {report_path}")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
