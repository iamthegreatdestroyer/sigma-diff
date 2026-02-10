"""
RYZEN-LLM Test Runner
Automated test execution with build verification
"""

import subprocess
import sys
from pathlib import Path
import argparse
import time

PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: list, cwd: Path = None, env: dict = None) -> int:
    """Run command and return exit code"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        env=env,
        capture_output=False,
    )
    return result.returncode


def build_project(config: str = "Release") -> bool:
    """Build the C++ project"""
    print("\n" + "=" * 80)
    print(f"Building project ({config} configuration)")
    print("=" * 80 + "\n")
    
    build_dir = PROJECT_ROOT / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Configure
    configure_cmd = [
        "cmake",
        "-S", str(PROJECT_ROOT),
        "-B", str(build_dir),
        "-G", "Ninja",
        f"-DCMAKE_BUILD_TYPE={config}",
    ]
    
    if run_command(configure_cmd) != 0:
        print("❌ CMake configuration failed")
        return False
    
    # Build
    build_cmd = [
        "cmake",
        "--build", str(build_dir),
        "--config", config,
        "--parallel",
    ]
    
    if run_command(build_cmd) != 0:
        print("❌ Build failed")
        return False
    
    print("✅ Build successful\n")
    return True


def run_unit_tests() -> bool:
    """Run C++ unit tests (if available)"""
    print("\n" + "=" * 80)
    print("Running C++ Unit Tests")
    print("=" * 80 + "\n")
    
    # TODO: Add C++ test executable path when available
    print("⚠️  C++ unit tests not yet implemented")
    return True


def run_integration_tests(markers: str = None) -> bool:
    """Run Python integration tests"""
    print("\n" + "=" * 80)
    print("Running Python Integration Tests")
    print("=" * 80 + "\n")
    
    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(PROJECT_ROOT / "tests" / "integration"),
        "-v",
        "--tb=short",
        "-s",
        "--color=yes",
    ]
    
    if markers:
        pytest_cmd.extend(["-m", markers])
    
    exit_code = run_command(pytest_cmd)
    
    if exit_code == 0:
        print("\n✅ All integration tests passed")
        return True
    else:
        print(f"\n❌ Integration tests failed with exit code {exit_code}")
        return False


def run_benchmarks() -> bool:
    """Run performance benchmarks"""
    print("\n" + "=" * 80)
    print("Running Performance Benchmarks")
    print("=" * 80 + "\n")
    
    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(PROJECT_ROOT / "tests" / "integration"),
        "-v",
        "-m", "benchmark",
        "--tb=short",
    ]
    
    try:
        exit_code = run_command(pytest_cmd)
        return exit_code == 0
    except Exception as e:
        print(f"⚠️  Benchmark execution failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="RYZEN-LLM Test Runner")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build project before running tests",
    )
    parser.add_argument(
        "--config",
        choices=["Debug", "Release"],
        default="Release",
        help="Build configuration",
    )
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmarks only",
    )
    parser.add_argument(
        "--markers",
        type=str,
        help="Pytest markers to filter tests",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests (default)",
    )
    
    args = parser.parse_args()
    
    # Default to running all tests
    if not (args.unit or args.integration or args.benchmark):
        args.all = True
    
    start_time = time.time()
    success = True
    
    # Build if requested
    if args.build:
        if not build_project(args.config):
            return 1
    
    # Run tests
    if args.all or args.unit:
        if not run_unit_tests():
            success = False
    
    if args.all or args.integration:
        if not run_integration_tests(args.markers):
            success = False
    
    if args.all or args.benchmark:
        if not run_benchmarks():
            success = False
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Test suite completed in {elapsed:.2f}s")
    
    if success:
        print("✅ All tests passed")
        print("=" * 80 + "\n")
        return 0
    else:
        print("❌ Some tests failed")
        print("=" * 80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
