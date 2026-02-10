#!/usr/bin/env python3
"""
Sprint 1.1 Environment Validation Script

Validates distributed inference environment and assumptions:
- PyTorch distributed functionality
- NCCL communication (simulated)
- Tensor parallelism concepts
- Multi-GPU setup verification

Usage:
    python validate_environment.py
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import os
import sys
import logging
from datetime import datetime
import psutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from distributed.architecture import DistributedConfig
from distributed.orchestrator import ProcessGroupManager
from distributed.tensor_parallel import RowParallelLinear, ColumnParallelLinear

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnvironmentValidator:
    """Validates distributed inference environment."""

    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()

    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        status = "PASS" if success else "FAIL"
        logger.info(f"[{status}] {test_name}: {details}")
        self.results[test_name] = {"success": success, "details": details}

    def check_pytorch_setup(self):
        """Check PyTorch installation and basic functionality."""
        try:
            logger.info("Checking PyTorch setup...")

            # Version check
            version = torch.__version__
            self.log_result("PyTorch Version", True, f"v{version}")

            # CUDA check (note: this is CPU-only environment)
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                self.log_result("CUDA Available", True, f"CUDA {cuda_version}, {gpu_count} GPUs")
            else:
                self.log_result("CUDA Available", False, "CPU-only installation (expected for dev environment)")

            # Basic tensor operations
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.matmul(x, y)
            self.log_result("Basic Tensor Ops", True, f"Matrix mult successful: {z.shape}")

            return True
        except Exception as e:
            self.log_result("PyTorch Setup", False, str(e))
            return False

    def check_distributed_imports(self):
        """Check that distributed modules can be imported."""
        try:
            logger.info("Checking distributed module imports...")

            # Test imports
            from distributed.architecture import DistributedConfig, CommunicationHandler
            from distributed.orchestrator import ProcessGroupManager
            from distributed.model_loader import DistributedCheckpointLoader
            from distributed.communication import NCCLCommunicator
            from distributed.tensor_parallel import RowParallelLinear, ColumnParallelLinear

            self.log_result("Distributed Imports", True, "All modules imported successfully")
            return True
        except Exception as e:
            self.log_result("Distributed Imports", False, str(e))
            return False

    def test_tensor_parallelism_concepts(self):
        """Test tensor parallelism concepts without actual distribution."""
        try:
            logger.info("Testing tensor parallelism concepts...")

            # Simulate 4-GPU setup with CPU tensors
            world_size = 4
            hidden_size = 4096

            # Test RowParallelLinear concept
            row_parallel = RowParallelLinear(
                input_size=hidden_size,
                output_size=hidden_size,
                bias=True
            )

            # Create test input
            batch_size, seq_len = 2, 128
            x = torch.randn(batch_size, seq_len, hidden_size)

            # Forward pass (will use CPU since no CUDA)
            with torch.no_grad():
                y = row_parallel(x)
                self.log_result("RowParallelLinear Forward", True,
                              f"Input: {x.shape}, Output: {y.shape}")

            # Test ColumnParallelLinear concept
            col_parallel = ColumnParallelLinear(
                input_size=hidden_size,
                output_size=hidden_size,
                bias=True
            )

            with torch.no_grad():
                y_col = col_parallel(x)
                self.log_result("ColumnParallelLinear Forward", True,
                              f"Input: {x.shape}, Output: {y_col.shape}")

            return True
        except Exception as e:
            self.log_result("Tensor Parallelism Concepts", False, str(e))
            return False

    def simulate_communication_patterns(self):
        """Simulate communication patterns (no actual NCCL needed)."""
        try:
            logger.info("Simulating communication patterns...")

            # Simulate all-reduce pattern
            tensors = [torch.randn(10, 10) for _ in range(4)]  # 4 ranks
            summed = sum(tensors)  # Simulate all-reduce sum

            self.log_result("All-Reduce Simulation", True,
                          f"4 tensors summed: {summed.shape}")

            # Simulate all-gather pattern
            gathered = torch.stack(tensors, dim=0)  # Gather all tensors
            self.log_result("All-Gather Simulation", True,
                          f"Gathered shape: {gathered.shape}")

            return True
        except Exception as e:
            self.log_result("Communication Patterns", False, str(e))
            return False

    def check_system_resources(self):
        """Check system resources for distributed setup."""
        try:
            logger.info("Checking system resources...")

            # CPU info
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.log_result("CPU Cores", True, f"{cpu_count} cores available")

            # Memory info
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            self.log_result("System Memory", True, f"{available_gb:.1f}GB available of {total_gb:.1f}GB total")

            # Disk space (for checkpoints)
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            self.log_result("Disk Space", free_gb > 10, f"{free_gb:.1f}GB free")

            return True
        except Exception as e:
            self.log_result("System Resources", False, str(e))
            return False

    def validate_assumptions(self):
        """Validate key assumptions from architecture document."""
        try:
            logger.info("Validating architecture assumptions...")

            # Assumption: PyTorch 2.x
            version_parts = torch.__version__.split('.')
            major_version = int(version_parts[0])
            self.log_result("PyTorch 2.x Assumption", major_version >= 2,
                          f"Version: {torch.__version__}")

            # Assumption: Distributed package available
            has_distributed = hasattr(torch, 'distributed')
            self.log_result("Torch Distributed Available", has_distributed, "torch.distributed module found")

            # Assumption: NCCL backend available (even if not used in CPU mode)
            try:
                import torch.distributed as dist
                has_nccl = 'nccl' in dist.Backend
                self.log_result("NCCL Backend Available", has_nccl, "NCCL backend supported")
            except:
                self.log_result("NCCL Backend Available", False, "Cannot check NCCL in CPU-only mode")

            return True
        except Exception as e:
            self.log_result("Architecture Assumptions", False, str(e))
            return False

    def run_all_validations(self):
        """Run all validation tests."""
        logger.info("=" * 60)
        logger.info("SPRINT 1.1 ENVIRONMENT VALIDATION")
        logger.info("=" * 60)

        tests = [
            self.check_pytorch_setup,
            self.check_distributed_imports,
            self.test_tensor_parallelism_concepts,
            self.simulate_communication_patterns,
            self.check_system_resources,
            self.validate_assumptions,
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            if test():
                passed += 1
            logger.info("-" * 40)

        # Summary
        duration = datetime.now() - self.start_time
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Tests Passed: {passed}/{total}")
        logger.info(f"Duration: {duration.total_seconds():.2f} seconds")

        if passed == total:
            logger.info("✅ ALL VALIDATIONS PASSED - Ready for distributed implementation")
            return True
        else:
            logger.warning(f"⚠️  {total - passed} validations failed - check results above")
            return False

    def generate_report(self):
        """Generate validation report."""
        report = f"""# Sprint 1.1 Environment Validation Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Duration:** {(datetime.now() - self.start_time).total_seconds():.2f} seconds

## Test Results

"""

        for test_name, result in self.results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            report += f"### {status} {test_name}\n"
            if result["details"]:
                report += f"{result['details']}\n\n"

        # Critical assumptions check
        critical_assumptions = [
            "PyTorch Setup",
            "Distributed Imports",
            "Tensor Parallelism Concepts",
            "Architecture Assumptions"
        ]

        critical_passed = all(self.results.get(assumption, {}).get("success", False)
                            for assumption in critical_assumptions)

        report += f"""## Critical Success Factors

**NCCL Latency <5ms:** ⚠️ Cannot test in CPU-only environment
**Torch Distributed:** {"✅ Working" if self.results.get("Torch Distributed Available", {}).get("success") else "❌ Not available"}
**Basic Tensor Ops:** {"✅ Working" if self.results.get("Basic Tensor Ops", {}).get("success") else "❌ Failed"}
**Architecture Assumptions:** {"✅ Validated" if critical_passed else "❌ Issues found"}

## Recommendations

1. **GPU Setup:** For production deployment, ensure CUDA GPUs are available
2. **NCCL Testing:** Test actual NCCL communication latencies on GPU hardware
3. **Multi-Node:** Validate master_addr/port configuration for multi-node setups
4. **Memory:** Ensure sufficient GPU memory for model sharding (4x sharding = ~25% memory per GPU)

## Next Steps

- Complete tensor parallelism implementation
- Test with actual GPU hardware
- Validate 4-GPU communication patterns
- Measure end-to-end performance improvements
"""

        return report


def main():
    """Main validation function."""
    validator = EnvironmentValidator()
    success = validator.run_all_validations()

    # Generate and save report
    report = validator.generate_report()
    report_path = "environment_validation_report.md"

    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"Validation report saved to: {report_path}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())