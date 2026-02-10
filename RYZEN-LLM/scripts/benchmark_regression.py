#!/usr/bin/env python3
"""
Benchmark Regression Detection for Ryzanstein LLM CI/CD Pipeline.

Compares current TL2_0 kernel performance against a stored baseline.
Exits non-zero if any kernel regresses beyond the configurable threshold.

Usage:
    python benchmark_regression.py --threshold 0.10
    python benchmark_regression.py --baseline-branch main --output report.json
    python benchmark_regression.py --update-baseline   # Store current results

Reports JSON with per-kernel metrics for CI annotation.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


# ─── Configuration ───────────────────────────────────────────────

BASELINE_FILE = Path(__file__).parent.parent / ".benchmark_baseline.json"
BUILD_DIR = Path(__file__).parent.parent / "build"

# Synthetic benchmark parameters matching parallel_kernels.cpp
BENCHMARK_CONFIGS = [
    {"name": "TL2_GEMV_512x512",   "M": 512,  "K": 512,  "N": 1,   "warmup": 3, "iters": 20},
    {"name": "TL2_GEMV_1024x1024", "M": 1024, "K": 1024, "N": 1,   "warmup": 3, "iters": 20},
    {"name": "TL2_GEMV_4096x4096", "M": 4096, "K": 4096, "N": 1,   "warmup": 3, "iters": 15},
    {"name": "TL2_GEMM_512x512x8", "M": 512,  "K": 512,  "N": 8,   "warmup": 3, "iters": 15},
    {"name": "TL2_GEMM_1024x1024x4","M": 1024, "K": 1024, "N": 4,  "warmup": 3, "iters": 10},
    {"name": "TL2_GEMM_2048x2048x1","M": 2048, "K": 2048, "N": 1,  "warmup": 3, "iters": 10},
]


@dataclass
class KernelResult:
    """Single kernel benchmark result."""
    name: str
    mean_us: float = 0.0
    median_us: float = 0.0
    min_us: float = 0.0
    max_us: float = 0.0
    stddev_us: float = 0.0
    gflops: float = 0.0
    kernel_type: str = ""


@dataclass
class RegressionResult:
    """Comparison result for a single kernel."""
    name: str
    baseline: float
    current: float
    change: str
    status: str  # "PASS", "WARN", "FAIL"


@dataclass
class BenchmarkReport:
    """Full benchmark regression report."""
    timestamp: str = ""
    commit: str = ""
    branch: str = ""
    threshold: float = 0.10
    overall_status: str = "PASS"
    results: List[Dict] = field(default_factory=list)
    cpu_info: str = ""


def get_git_info() -> Dict[str, str]:
    """Get current git commit and branch."""
    info = {"commit": "unknown", "branch": "unknown"}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return info


def run_native_benchmark() -> List[KernelResult]:
    """
    Run the C++ benchmark executable if available.
    Falls back to synthetic Python benchmarks.
    """
    # Try to find native benchmark binary
    possible_paths = [
        BUILD_DIR / "benchmark_tl2",
        BUILD_DIR / "Release" / "benchmark_tl2",
        BUILD_DIR / "benchmark_tl2.exe",
        BUILD_DIR / "Release" / "benchmark_tl2.exe",
    ]

    for path in possible_paths:
        if path.exists():
            return _run_native(path)

    # Fallback: run synthetic Python benchmarks
    print("[INFO] No native benchmark binary found, using synthetic benchmarks", file=sys.stderr)
    return run_synthetic_benchmarks()


def _run_native(binary_path: Path) -> List[KernelResult]:
    """Execute the native benchmark binary and parse JSON output."""
    results = []
    try:
        result = subprocess.run(
            [str(binary_path), "--json"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for entry in data.get("benchmarks", []):
                results.append(KernelResult(
                    name=entry["name"],
                    mean_us=entry.get("mean_us", 0),
                    median_us=entry.get("median_us", 0),
                    min_us=entry.get("min_us", 0),
                    max_us=entry.get("max_us", 0),
                    gflops=entry.get("gflops", 0),
                    kernel_type=entry.get("kernel_type", "")
                ))
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"[WARN] Native benchmark failed: {e}", file=sys.stderr)

    return results


def run_synthetic_benchmarks() -> List[KernelResult]:
    """
    Run synthetic Python benchmarks that approximate TL2_0 kernel behavior.
    Uses NumPy to simulate ternary GEMV/GEMM operations.
    """
    try:
        import numpy as np
    except ImportError:
        print("[ERROR] NumPy not available for synthetic benchmarks", file=sys.stderr)
        return []

    results = []
    rng = np.random.default_rng(42)

    for config in BENCHMARK_CONFIGS:
        M, K, N = config["M"], config["K"], config["N"]
        name = config["name"]
        warmup = config["warmup"]
        iters = config["iters"]

        # Generate ternary weights {-1, 0, +1}
        weights = rng.choice([-1, 0, 1], size=(M, K)).astype(np.int8)
        scales = rng.uniform(0.5, 2.0, size=M).astype(np.float32)

        # Generate INT8 activations
        activations = rng.integers(-128, 127, size=(K, N), dtype=np.int8)

        # Simulate TL2_0 LUT-based computation
        # Build LUT: for each unique activation value, precompute {-act, 0, +act}
        # Then dispatch based on weight value
        timings = []

        for i in range(warmup + iters):
            start = time.perf_counter_ns()

            # Simulate: output[m, n] = sum_k(weight[m,k] * activation[k,n]) * scale[m]
            # LUT approach: precompute LUT[act_val] = {-act_val, 0, +act_val}
            output = np.zeros((M, N), dtype=np.float32)
            for n in range(N):
                act_col = activations[:, n].astype(np.float32)
                for m in range(M):
                    row = weights[m, :]
                    # LUT simulation: sum positive, subtract negative, skip zero
                    pos_mask = row == 1
                    neg_mask = row == -1
                    acc = np.sum(act_col[pos_mask]) - np.sum(act_col[neg_mask])
                    output[m, n] = acc * scales[m]

            elapsed_ns = time.perf_counter_ns() - start

            if i >= warmup:
                timings.append(elapsed_ns / 1000.0)  # Convert to microseconds

        timings_arr = np.array(timings)
        flops = 2.0 * M * K * N  # multiply-accumulate = 2 ops

        results.append(KernelResult(
            name=name,
            mean_us=float(np.mean(timings_arr)),
            median_us=float(np.median(timings_arr)),
            min_us=float(np.min(timings_arr)),
            max_us=float(np.max(timings_arr)),
            stddev_us=float(np.std(timings_arr)),
            gflops=flops / (float(np.mean(timings_arr)) * 1000.0) if np.mean(timings_arr) > 0 else 0,
            kernel_type="synthetic_python"
        ))

        print(f"  [{name}] mean={np.mean(timings_arr):.1f}μs "
              f"min={np.min(timings_arr):.1f}μs "
              f"gflops={results[-1].gflops:.3f}", file=sys.stderr)

    return results


def load_baseline() -> Optional[Dict[str, float]]:
    """Load baseline benchmark results."""
    if not BASELINE_FILE.exists():
        return None

    try:
        with open(BASELINE_FILE, "r") as f:
            data = json.load(f)
        return {entry["name"]: entry["mean_us"] for entry in data.get("benchmarks", [])}
    except (json.JSONDecodeError, KeyError):
        return None


def save_baseline(results: List[KernelResult]) -> None:
    """Save current results as baseline."""
    git_info = get_git_info()
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git_info["commit"],
        "branch": git_info["branch"],
        "benchmarks": [asdict(r) for r in results]
    }
    with open(BASELINE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Baseline saved to {BASELINE_FILE} with {len(results)} benchmarks")


def compare_results(
    current: List[KernelResult],
    baseline: Dict[str, float],
    threshold: float
) -> BenchmarkReport:
    """Compare current results against baseline, detect regressions."""
    git_info = get_git_info()
    report = BenchmarkReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        commit=git_info["commit"],
        branch=git_info["branch"],
        threshold=threshold,
    )

    has_failure = False

    for result in current:
        if result.name not in baseline:
            report.results.append(asdict(RegressionResult(
                name=result.name,
                baseline=0.0,
                current=result.mean_us,
                change="NEW",
                status="PASS"
            )))
            continue

        base_us = baseline[result.name]
        curr_us = result.mean_us

        if base_us > 0:
            pct_change = (curr_us - base_us) / base_us
            change_str = f"{pct_change:+.1%}"
        else:
            pct_change = 0.0
            change_str = "N/A"

        if pct_change > threshold:
            status = "FAIL"
            has_failure = True
        elif pct_change > threshold * 0.5:
            status = "WARN"
        else:
            status = "PASS"

        report.results.append(asdict(RegressionResult(
            name=result.name,
            baseline=base_us,
            current=curr_us,
            change=change_str,
            status=status,
        )))

    report.overall_status = "FAIL" if has_failure else "PASS"
    return report


def print_report(report: BenchmarkReport) -> None:
    """Print human-readable report to stderr."""
    print("\n═══════════════════════════════════════════════════════", file=sys.stderr)
    print("  Ryzanstein LLM - Benchmark Regression Report", file=sys.stderr)
    print("═══════════════════════════════════════════════════════", file=sys.stderr)
    print(f"  Commit:    {report.commit}", file=sys.stderr)
    print(f"  Branch:    {report.branch}", file=sys.stderr)
    print(f"  Threshold: {report.threshold:.0%} regression allowed", file=sys.stderr)
    print(f"  Status:    {report.overall_status}", file=sys.stderr)
    print("", file=sys.stderr)

    header = f"  {'Kernel':<30} {'Baseline':>12} {'Current':>12} {'Change':>10} {'Status':>8}"
    print(header, file=sys.stderr)
    print("  " + "─" * 74, file=sys.stderr)

    for r in report.results:
        base_str = f"{r['baseline']:.1f}μs" if r['baseline'] > 0 else "N/A"
        curr_str = f"{r['current']:.1f}μs"
        status_icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(r["status"], "?")
        print(f"  {r['name']:<30} {base_str:>12} {curr_str:>12} "
              f"{r['change']:>10} {status_icon:>2} {r['status']}", file=sys.stderr)

    print("═══════════════════════════════════════════════════════\n", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark regression detection for Ryzanstein LLM"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.10,
        help="Maximum allowed regression fraction (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--baseline-branch", type=str, default=None,
        help="Branch to compare against (uses stored baseline if not specified)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON report path"
    )
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="Run benchmarks and save as new baseline"
    )
    args = parser.parse_args()

    print("[INFO] Running benchmarks...", file=sys.stderr)
    current_results = run_native_benchmark()

    if not current_results:
        print("[ERROR] No benchmark results obtained", file=sys.stderr)
        sys.exit(1)

    # Update baseline mode
    if args.update_baseline:
        save_baseline(current_results)
        sys.exit(0)

    # Load baseline
    baseline = load_baseline()
    if baseline is None:
        print("[WARN] No baseline found, creating initial baseline", file=sys.stderr)
        save_baseline(current_results)
        # Generate a pass report
        report = BenchmarkReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            threshold=args.threshold,
            overall_status="PASS",
            results=[asdict(RegressionResult(
                name=r.name, baseline=r.mean_us, current=r.mean_us,
                change="+0.0%", status="PASS"
            )) for r in current_results]
        )
    else:
        # Compare against baseline
        report = compare_results(current_results, baseline, args.threshold)

    # Print human-readable report
    print_report(report)

    # Write JSON output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"[INFO] Report written to {output_path}", file=sys.stderr)

    # Exit with appropriate code
    if report.overall_status == "FAIL":
        print("[FAIL] Benchmark regression detected!", file=sys.stderr)
        sys.exit(1)
    else:
        print("[PASS] No significant regressions", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
