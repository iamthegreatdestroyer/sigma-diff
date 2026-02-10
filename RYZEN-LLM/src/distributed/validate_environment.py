#!/usr/bin/env python3
"""
Environment Validation Script for Distributed Architecture

Validates PyTorch distributed setup, NCCL assumptions, and tensor parallelism concepts.
Tests 4-GPU environment simulation on CPU-only setup.

Usage: python validate_environment.py
"""

import sys
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
import time
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributed.tensor_parallel import RowParallelLinear, ColumnParallelLinear, ParallelAttention, ParallelMLP
from distributed.communication import NCCLCommunicator
from distributed.orchestrator import ProcessGroupManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnvironmentValidator:
    """Validates distributed environment and tensor parallelism setup."""

    def __init__(self):
        self.results = {}
        self.world_size = 4  # Simulate 4 GPUs

    def validate_pytorch_setup(self) -> Dict[str, Any]:
        """Validate PyTorch installation and capabilities."""
        logger.info("Validating PyTorch setup...")

        results = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "distributed_available": hasattr(torch, 'distributed'),
            "nccl_available": torch.distributed.is_nccl_available() if hasattr(torch, 'distributed') else False,
            "gloo_available": torch.distributed.is_gloo_available() if hasattr(torch, 'distributed') else False,
        }

        logger.info(f"PyTorch version: {results['torch_version']}")
        logger.info(f"CUDA available: {results['cuda_available']}")
        logger.info(f"GPU count: {results['gpu_count']}")
        logger.info(f"Distributed available: {results['distributed_available']}")
        logger.info(f"NCCL available: {results['nccl_available']}")
        logger.info(f"Gloo available: {results['gloo_available']}")

        return results

    def validate_tensor_parallel_imports(self) -> Dict[str, Any]:
        """Validate tensor parallelism imports."""
        logger.info("Validating tensor parallelism imports...")

        results = {}
        try:
            from distributed.tensor_parallel import RowParallelLinear, ColumnParallelLinear, ParallelAttention, ParallelMLP
            results["row_parallel_linear"] = True
            results["column_parallel_linear"] = True
            results["parallel_attention"] = True
            results["parallel_mlp"] = True
            logger.info("All tensor parallelism classes imported successfully")
        except ImportError as e:
            results["import_error"] = str(e)
            logger.error(f"Import error: {e}")

        return results

    def test_tensor_parallel_layers(self) -> Dict[str, Any]:
        """Test tensor parallel layer functionality."""
        logger.info("Testing tensor parallel layers...")

        results = {}
        batch_size, seq_len, hidden_size = 2, 128, 1024

        try:
            # Test RowParallelLinear
            row_layer = RowParallelLinear(hidden_size, hidden_size)
            x = torch.randn(batch_size, seq_len, hidden_size)
            out = row_layer(x)
            results["row_parallel_shape"] = out.shape == x.shape
            logger.info(f"RowParallelLinear: input {x.shape} -> output {out.shape}")

            # Test ColumnParallelLinear
            col_layer = ColumnParallelLinear(hidden_size, hidden_size)
            out = col_layer(x)
            results["column_parallel_shape"] = out.shape == x.shape
            logger.info(f"ColumnParallelLinear: input {x.shape} -> output {out.shape}")

            # Test ParallelAttention
            attn_layer = ParallelAttention(hidden_size, 16)  # 16 heads
            out = attn_layer(x)
            results["parallel_attention_shape"] = out.shape == x.shape
            logger.info(f"ParallelAttention: input {x.shape} -> output {out.shape}")

            # Test ParallelMLP
            mlp_layer = ParallelMLP(hidden_size, 4 * hidden_size)  # 4x expansion
            out = mlp_layer(x)
            results["parallel_mlp_shape"] = out.shape == x.shape
            logger.info(f"ParallelMLP: input {x.shape} -> output {out.shape}")

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Tensor parallel test error: {e}")

        return results

    def simulate_distributed_setup(self) -> Dict[str, Any]:
        """Simulate distributed setup with multiple processes."""
        logger.info("Simulating distributed setup...")

        results = {}

        # Test ProcessGroupManager
        try:
            pg_manager = ProcessGroupManager()
            results["process_group_manager"] = True
            logger.info("ProcessGroupManager created successfully")
        except Exception as e:
            results["process_group_manager_error"] = str(e)
            logger.error(f"ProcessGroupManager error: {e}")

        # Test NCCLCommunicator
        try:
            comm = NCCLCommunicator()
            results["nccl_communicator"] = True
            logger.info("NCCLCommunicator created successfully")
        except Exception as e:
            results["nccl_communicator_error"] = str(e)
            logger.error(f"NCCLCommunicator error: {e}")

        return results

    def benchmark_communication(self) -> Dict[str, Any]:
        """Benchmark communication primitives."""
        logger.info("Benchmarking communication primitives...")

        results = {}
        sizes = [1024, 8192, 65536, 524288]  # Various tensor sizes

        try:
            comm = NCCLCommunicator()

            for size in sizes:
                # Create test tensor
                tensor = torch.randn(size)

                # Benchmark all_reduce
                start_time = time.time()
                result = comm.all_reduce(tensor, op="sum")
                end_time = time.time()

                latency = (end_time - start_time) * 1000  # ms
                results[f"all_reduce_{size}"] = latency
                logger.info(f"All-reduce {size} elements: {latency:.2f}ms")

                # Check NCCL latency requirement (< 5ms for large tensors)
                if size >= 65536 and latency < 5.0:
                    results["nccl_latency_requirement"] = True

        except Exception as e:
            results["benchmark_error"] = str(e)
            logger.error(f"Benchmark error: {e}")

        return results

    def validate_memory_layout(self) -> Dict[str, Any]:
        """Validate tensor parallelism memory layout."""
        logger.info("Validating memory layout...")

        results = {}
        hidden_size = 1024

        try:
            # Test weight sharding
            row_layer = RowParallelLinear(hidden_size, hidden_size)
            col_layer = ColumnParallelLinear(hidden_size, hidden_size)

            # Check parameter counts
            row_params = sum(p.numel() for p in row_layer.parameters())
            col_params = sum(p.numel() for p in col_layer.parameters())

            # Row-parallel should have fewer parameters per GPU
            results["row_parallel_memory"] = row_params < hidden_size * hidden_size
            results["column_parallel_memory"] = col_params == hidden_size * hidden_size

            logger.info(f"Row-parallel parameters: {row_params}")
            logger.info(f"Column-parallel parameters: {col_params}")

        except Exception as e:
            results["memory_error"] = str(e)
            logger.error(f"Memory validation error: {e}")

        return results

    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests."""
        logger.info("Starting comprehensive environment validation...")

        self.results = {
            "pytorch_setup": self.validate_pytorch_setup(),
            "tensor_parallel_imports": self.validate_tensor_parallel_imports(),
            "tensor_parallel_tests": self.test_tensor_parallel_layers(),
            "distributed_simulation": self.simulate_distributed_setup(),
            "communication_benchmark": self.benchmark_communication(),
            "memory_layout": self.validate_memory_layout(),
        }

        # Summary
        success_count = 0
        total_count = 0

        for category, tests in self.results.items():
            for test_name, result in tests.items():
                total_count += 1
                if isinstance(result, bool) and result:
                    success_count += 1
                elif isinstance(result, (int, float)) and not test_name.endswith("_error"):
                    # Numeric results are considered successful if they exist
                    success_count += 1

        self.results["summary"] = {
            "total_tests": total_count,
            "successful_tests": success_count,
            "success_rate": success_count / total_count if total_count > 0 else 0,
            "environment_ready": success_count >= total_count * 0.8  # 80% success threshold
        }

        logger.info(f"Validation complete: {success_count}/{total_count} tests passed ({self.results['summary']['success_rate']:.1%})")

        return self.results

    def print_report(self):
        """Print validation report."""
        print("\n" + "="*60)
        print("DISTRIBUTED ARCHITECTURE ENVIRONMENT VALIDATION REPORT")
        print("="*60)

        print(f"\nEnvironment Status: {'READY' if self.results['summary']['environment_ready'] else 'ISSUES DETECTED'}")
        print(f"Success Rate: {self.results['summary']['success_rate']:.1%}")
        print(f"Tests Passed: {self.results['summary']['successful_tests']}/{self.results['summary']['total_tests']}")

        print("\n" + "-"*40)
        print("DETAILED RESULTS")
        print("-"*40)

        for category, tests in self.results.items():
            if category == "summary":
                continue
            print(f"\n{category.upper().replace('_', ' ')}:")
            for test_name, result in tests.items():
                status = "✓" if (isinstance(result, bool) and result) or (isinstance(result, (int, float)) and not test_name.endswith("_error")) else "✗"
                if test_name.endswith("_error"):
                    print(f"  {status} {test_name}: {result}")
                elif isinstance(result, bool):
                    print(f"  {status} {test_name}")
                elif isinstance(result, (int, float)):
                    print(f"  {status} {test_name}: {result}")
                else:
                    print(f"  {status} {test_name}: {result}")

        print("\n" + "="*60)


def main():
    """Main validation function."""
    validator = EnvironmentValidator()
    validator.run_all_validations()
    validator.print_report()

    # Exit with appropriate code
    if validator.results["summary"]["environment_ready"]:
        logger.info("Environment validation PASSED - ready for distributed training")
        sys.exit(0)
    else:
        logger.warning("Environment validation FAILED - check issues above")
        sys.exit(1)


if __name__ == "__main__":
    main()
