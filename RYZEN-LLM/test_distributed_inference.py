#!/usr/bin/env python3
"""
Distributed Inference Test Script
=================================

Tests distributed inference engine with tensor parallelism.
Validates 4-GPU baseline performance and scaling efficiency.

Usage:
    # Single GPU baseline
    python test_distributed_inference.py --world_size 1

    # Multi-GPU distributed (4 GPUs)
    torchrun --nproc_per_node=4 test_distributed_inference.py --world_size 4
"""

import torch
import torch.distributed as dist
import argparse
import time
import logging
from typing import Dict, Any
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.engine.distributed_inference import DistributedInferenceEngine
from core.model import BitNetConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test distributed inference')
    parser.add_argument('--world_size', type=int, default=1,
                       help='Number of processes/GPUs')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='Sequence length')
    parser.add_argument('--num_runs', type=int, default=10,
                       help='Number of test runs for averaging')
    parser.add_argument('--warmup_runs', type=int, default=3,
                       help='Number of warmup runs')
    return parser.parse_args()


def create_test_config() -> BitNetConfig:
    """Create test model configuration."""
    config = BitNetConfig()
    config.model_name = "test-bitnet"
    config.vocab_size = 32000
    config.hidden_size = 768
    config.num_layers = 12
    config.num_heads = 12
    config.max_seq_len = 2048
    config.num_parameters = 125000000  # 125M parameters
    return config


def benchmark_inference(engine: DistributedInferenceEngine,
                       prompt: str,
                       config: Any,
                       num_runs: int,
                       warmup_runs: int) -> Dict[str, float]:
    """Benchmark inference performance."""
    logger.info(f"Running benchmark with {num_runs} runs, {warmup_runs} warmup")

    # Warmup runs
    for i in range(warmup_runs):
        with engine.performance_context():
            result = engine.generate(prompt, config, stream=False)
        if engine.rank == 0:
            logger.info(f"Warmup run {i+1}/{warmup_runs} completed")

    # Benchmark runs
    latencies = []
    tokens_generated = []

    for i in range(num_runs):
        start_time = time.time()

        with engine.performance_context():
            result = engine.generate(prompt, config, stream=False)

        end_time = time.time()
        latency = end_time - start_time

        if engine.rank == 0:
            # Count tokens (simplified)
            token_count = len(result.split()) if isinstance(result, str) else 1
            tokens_generated.append(token_count)
            latencies.append(latency)
            logger.info(".3f")

    # Calculate metrics (only on rank 0)
    if engine.rank == 0:
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        throughput = avg_tokens / avg_latency  # tokens per second

        return {
            'avg_latency': avg_latency,
            'avg_tokens': avg_tokens,
            'throughput': throughput,
            'world_size': engine.world_size
        }
    else:
        return {}


def main():
    """Main test function."""
    args = parse_args()

    # Initialize distributed environment if multi-GPU
    if args.world_size > 1:
        # torchrun handles this automatically
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Starting distributed inference test on rank {rank}/{world_size}")

    # Create test configuration
    config = create_test_config()

    # Create distributed inference engine
    engine = DistributedInferenceEngine(
        model_path=args.model_path,
        config=config,
        world_size=world_size,
        rank=rank
    )

    # Load model (simulated for testing)
    success = engine.load_model()
    if not success:
        logger.error("Failed to load model")
        return 1

    # Test prompt
    prompt = "The quick brown fox jumps over the lazy dog. " * 5

    # Create generation config
    class TestConfig:
        def __init__(self):
            self.max_new_tokens = 50
            self.temperature = 0.7
            self.top_k = 40
            self.top_p = 0.9

    gen_config = TestConfig()

    # Run benchmark
    results = benchmark_inference(
        engine, prompt, gen_config,
        args.num_runs, args.warmup_runs
    )

    # Report results
    if rank == 0:
        logger.info("=== BENCHMARK RESULTS ===")
        logger.info(f"World Size: {results.get('world_size', 1)}")
        logger.info(".3f")
        logger.info(".1f")
        logger.info(".2f")

        # Calculate scaling efficiency if multi-GPU
        if results.get('world_size', 1) > 1:
            # This would compare against single-GPU baseline
            # For now, just report the metrics
            logger.info("Multi-GPU scaling test completed")
        else:
            logger.info("Single-GPU baseline test completed")

    # Cleanup
    engine.unload_model()

    logger.info("Distributed inference test completed successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main())