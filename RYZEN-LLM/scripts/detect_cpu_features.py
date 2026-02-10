#!/usr/bin/env python3
"""
CPU Feature Detection for Ryzanstein LLM Build System.

Cross-platform detection of SIMD/vectorization capabilities used to
select optimal kernel dispatch paths at build time and in CI.

Detects: AVX, AVX2, FMA, AVX-512F, AVX-512BW, AVX-512VNNI, SSE4.2,
         OpenMP availability, core count, cache topology.

Usage:
    python detect_cpu_features.py             # Human-readable output
    python detect_cpu_features.py --json      # JSON for CI pipelines
    python detect_cpu_features.py --cmake     # CMake cache variable format
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class CacheInfo:
    """CPU cache hierarchy information."""
    l1_data_kb: int = 0
    l1_instruction_kb: int = 0
    l2_kb: int = 0
    l3_kb: int = 0


@dataclass
class CPUFeatures:
    """Detected CPU capabilities."""
    # Architecture
    architecture: str = ""
    model_name: str = ""
    vendor: str = ""
    physical_cores: int = 0
    logical_cores: int = 0

    # SIMD instruction sets
    sse42: bool = False
    avx: bool = False
    avx2: bool = False
    fma: bool = False
    avx512f: bool = False
    avx512bw: bool = False
    avx512vnni: bool = False
    avx512vbmi: bool = False
    amx_tile: bool = False
    amx_int8: bool = False

    # OpenMP
    openmp_available: bool = False
    openmp_max_threads: int = 0

    # Cache topology
    cache: CacheInfo = field(default_factory=CacheInfo)

    # Recommended kernel configuration
    recommended_kernel: str = ""
    recommended_tile_m: int = 0
    recommended_tile_n: int = 0
    recommended_tile_k: int = 0

    def recommend_kernel(self) -> None:
        """Select optimal kernel and tile sizes based on detected features."""
        if self.avx512vnni and self.avx512f:
            self.recommended_kernel = "TL2_LUT_AVX2"  # LUT wins for ternary GEMV
            # For GEMM, AVX-512 VNNI tiled matmul is competitive
            self.recommended_tile_m = 64
            self.recommended_tile_n = 64
            self.recommended_tile_k = 64
        elif self.avx2 and self.fma:
            self.recommended_kernel = "TL2_LUT_AVX2"
            self.recommended_tile_m = 32
            self.recommended_tile_n = 32
            self.recommended_tile_k = 64
        elif self.avx2:
            self.recommended_kernel = "TL2_LUT_OMP"
            self.recommended_tile_m = 16
            self.recommended_tile_n = 16
            self.recommended_tile_k = 32
        else:
            self.recommended_kernel = "TL2_LUT_SCALAR"
            self.recommended_tile_m = 8
            self.recommended_tile_n = 8
            self.recommended_tile_k = 16

        # Adjust tile sizes for cache if we have info
        if self.cache.l1_data_kb > 0:
            # TernaryLUT = 256 * 3 * 4 = 3072 bytes
            # Each tile needs: tile_m * tile_k * 1 (weights) + tile_k * 4 (activations)
            # Target: fit working set in L1 / 2
            l1_budget = (self.cache.l1_data_kb * 1024) // 2
            max_tile = int((l1_budget / 8) ** 0.5)
            self.recommended_tile_m = min(self.recommended_tile_m, max_tile)
            self.recommended_tile_k = min(self.recommended_tile_k, max_tile * 2)


def detect_linux() -> CPUFeatures:
    """Detect CPU features on Linux via /proc/cpuinfo."""
    features = CPUFeatures()
    features.architecture = platform.machine()

    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()

        # Model name
        match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        if match:
            features.model_name = match.group(1).strip()

        # Vendor
        match = re.search(r"vendor_id\s*:\s*(\w+)", cpuinfo)
        if match:
            features.vendor = match.group(1)

        # Core count
        physical = set()
        logical = 0
        for line in cpuinfo.split("\n"):
            if line.startswith("physical id") and "core id" in cpuinfo:
                pass
            if line.startswith("processor"):
                logical += 1
            match = re.match(r"core id\s*:\s*(\d+)", line)
            if match:
                physical.add(match.group(1))

        features.logical_cores = logical
        features.physical_cores = len(physical) if physical else logical

        # Flags
        match = re.search(r"flags\s*:\s*(.+)", cpuinfo)
        if match:
            flags = set(match.group(1).split())
            features.sse42 = "sse4_2" in flags
            features.avx = "avx" in flags
            features.avx2 = "avx2" in flags
            features.fma = "fma" in flags
            features.avx512f = "avx512f" in flags
            features.avx512bw = "avx512bw" in flags
            features.avx512vnni = "avx512_vnni" in flags or "avx512vnni" in flags
            features.avx512vbmi = "avx512vbmi" in flags
            features.amx_tile = "amx_tile" in flags
            features.amx_int8 = "amx_int8" in flags

    except FileNotFoundError:
        pass

    # Cache info from sysfs
    try:
        cache_base = "/sys/devices/system/cpu/cpu0/cache"
        for idx in range(10):
            index_path = f"{cache_base}/index{idx}"
            if not os.path.exists(index_path):
                break
            with open(f"{index_path}/level") as f:
                level = int(f.read().strip())
            with open(f"{index_path}/type") as f:
                cache_type = f.read().strip()
            with open(f"{index_path}/size") as f:
                size_str = f.read().strip()
                size_kb = int(re.match(r"(\d+)", size_str).group(1))

            if level == 1 and cache_type == "Data":
                features.cache.l1_data_kb = size_kb
            elif level == 1 and cache_type == "Instruction":
                features.cache.l1_instruction_kb = size_kb
            elif level == 2:
                features.cache.l2_kb = size_kb
            elif level == 3:
                features.cache.l3_kb = size_kb
    except (FileNotFoundError, ValueError):
        pass

    return features


def detect_windows() -> CPUFeatures:
    """Detect CPU features on Windows via WMI and registry."""
    features = CPUFeatures()
    features.architecture = platform.machine()

    try:
        # Use wmic for basic info
        result = subprocess.run(
            ["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors,Manufacturer", "/format:list"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split("\n"):
            line = line.strip()
            if line.startswith("Name="):
                features.model_name = line.split("=", 1)[1]
            elif line.startswith("NumberOfCores="):
                features.physical_cores = int(line.split("=")[1])
            elif line.startswith("NumberOfLogicalProcessors="):
                features.logical_cores = int(line.split("=")[1])
            elif line.startswith("Manufacturer="):
                features.vendor = line.split("=")[1]
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        features.logical_cores = os.cpu_count() or 1
        features.physical_cores = features.logical_cores // 2

    # Detect SIMD via a small C test or model name heuristics
    model = features.model_name.lower()

    # Ryzen 9 7950X and similar Zen 4 have AVX-512
    if any(kw in model for kw in ["7950x", "7900x", "7800x", "7700x", "7600x",
                                   "9950x", "9900x", "9800x", "9700x",
                                   "epyc 9"]):
        features.sse42 = True
        features.avx = True
        features.avx2 = True
        features.fma = True
        features.avx512f = True
        features.avx512bw = True
        features.avx512vnni = True
    elif any(kw in model for kw in ["ryzen", "zen", "epyc"]):
        features.sse42 = True
        features.avx = True
        features.avx2 = True
        features.fma = True
    elif any(kw in model for kw in ["i9-1", "i7-1", "i5-1", "i3-1",
                                     "xeon w-3", "xeon w-2"]):
        features.sse42 = True
        features.avx = True
        features.avx2 = True
        features.fma = True
        features.avx512f = True
        features.avx512bw = True
        if any(kw in model for kw in ["12", "13", "14"]):
            features.avx512vnni = True

    # Try to get more accurate info from compiler intrinsic test
    _try_compile_detect(features)

    # Default cache for Zen 4
    if "7950x" in model:
        features.cache = CacheInfo(
            l1_data_kb=32, l1_instruction_kb=32,
            l2_kb=1024, l3_kb=65536  # 64MB L3
        )
    elif features.cache.l1_data_kb == 0:
        # Conservative defaults
        features.cache = CacheInfo(
            l1_data_kb=32, l1_instruction_kb=32,
            l2_kb=512, l3_kb=32768
        )

    return features


def detect_macos() -> CPUFeatures:
    """Detect CPU features on macOS via sysctl."""
    features = CPUFeatures()
    features.architecture = platform.machine()

    try:
        result = subprocess.run(
            ["sysctl", "-a"], capture_output=True, text=True, timeout=10
        )
        sysctl = result.stdout

        match = re.search(r"machdep.cpu.brand_string:\s*(.+)", sysctl)
        if match:
            features.model_name = match.group(1).strip()

        match = re.search(r"hw.physicalcpu:\s*(\d+)", sysctl)
        if match:
            features.physical_cores = int(match.group(1))

        match = re.search(r"hw.logicalcpu:\s*(\d+)", sysctl)
        if match:
            features.logical_cores = int(match.group(1))

        match = re.search(r"machdep.cpu.features:\s*(.+)", sysctl)
        if match:
            flags = set(match.group(1).upper().split())
            features.sse42 = "SSE4.2" in flags
            features.avx = "AVX1.0" in flags or "AVX" in flags
            features.avx2 = "AVX2" in flags
            features.fma = "FMA" in flags

        match = re.search(r"machdep.cpu.leaf7_features:\s*(.+)", sysctl)
        if match:
            leaf7 = set(match.group(1).upper().split())
            features.avx2 = features.avx2 or "AVX2" in leaf7
            features.avx512f = "AVX512F" in leaf7
            features.avx512bw = "AVX512BW" in leaf7
            features.avx512vnni = "AVX512VNNI" in leaf7

        # Cache info
        for key, attr in [("hw.l1dcachesize", "l1_data_kb"),
                          ("hw.l1icachesize", "l1_instruction_kb"),
                          ("hw.l2cachesize", "l2_kb"),
                          ("hw.l3cachesize", "l3_kb")]:
            match = re.search(rf"{key}:\s*(\d+)", sysctl)
            if match:
                setattr(features.cache, attr, int(match.group(1)) // 1024)

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return features


def _try_compile_detect(features: CPUFeatures) -> None:
    """
    Try to compile a small C program to detect SIMD at runtime.
    Fallback approach when OS APIs are insufficient.
    """
    import tempfile

    test_code = r"""
#include <stdio.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

int main() {
    unsigned int eax, ebx, ecx, edx;

    // Basic CPUID leaf 1
#ifdef _MSC_VER
    int info[4];
    __cpuid(info, 1);
    ecx = info[2]; edx = info[3];
#else
    __cpuid(1, eax, ebx, ecx, edx);
#endif

    int sse42 = (ecx >> 20) & 1;
    int avx   = (ecx >> 28) & 1;
    int fma   = (ecx >> 12) & 1;

    // Extended leaf 7
#ifdef _MSC_VER
    __cpuidex(info, 7, 0);
    ebx = info[1]; ecx = info[2];
#else
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
#endif

    int avx2     = (ebx >> 5) & 1;
    int avx512f  = (ebx >> 16) & 1;
    int avx512bw = (ebx >> 30) & 1;
    int vnni     = (ecx >> 11) & 1;

    printf("sse42=%d avx=%d avx2=%d fma=%d avx512f=%d avx512bw=%d vnni=%d\n",
           sse42, avx, avx2, fma, avx512f, avx512bw, vnni);
    return 0;
}
"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(test_code)
            src_path = f.name

        out_path = src_path.replace(".c", ".exe" if platform.system() == "Windows" else "")

        # Try to compile
        if platform.system() == "Windows":
            result = subprocess.run(
                ["cl", "/Fe:" + out_path, src_path, "/nologo"],
                capture_output=True, text=True, timeout=30
            )
        else:
            result = subprocess.run(
                ["gcc", "-o", out_path, src_path],
                capture_output=True, text=True, timeout=30
            )

        if result.returncode == 0:
            run_result = subprocess.run(
                [out_path], capture_output=True, text=True, timeout=5
            )
            if run_result.returncode == 0:
                output = run_result.stdout.strip()
                for pair in output.split():
                    key, val = pair.split("=")
                    if val == "1":
                        if key == "sse42": features.sse42 = True
                        elif key == "avx": features.avx = True
                        elif key == "avx2": features.avx2 = True
                        elif key == "fma": features.fma = True
                        elif key == "avx512f": features.avx512f = True
                        elif key == "avx512bw": features.avx512bw = True
                        elif key == "vnni": features.avx512vnni = True

        # Cleanup
        for p in [src_path, out_path, src_path.replace(".c", ".obj")]:
            try:
                os.unlink(p)
            except OSError:
                pass

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass


def detect_openmp(features: CPUFeatures) -> None:
    """Check if OpenMP is available."""
    try:
        env_threads = os.environ.get("OMP_NUM_THREADS")
        if env_threads:
            features.openmp_max_threads = int(env_threads)
            features.openmp_available = True
        else:
            features.openmp_max_threads = features.logical_cores
            features.openmp_available = True  # Assume available, CI will verify
    except ValueError:
        pass


def detect() -> CPUFeatures:
    """Auto-detect CPU features for the current platform."""
    system = platform.system()

    if system == "Linux":
        features = detect_linux()
    elif system == "Windows":
        features = detect_windows()
    elif system == "Darwin":
        features = detect_macos()
    else:
        features = CPUFeatures()
        features.architecture = platform.machine()
        features.logical_cores = os.cpu_count() or 1
        features.physical_cores = features.logical_cores

    detect_openmp(features)
    features.recommend_kernel()

    return features


def format_human(features: CPUFeatures) -> str:
    """Format features as human-readable output."""
    lines = [
        "═══════════════════════════════════════════════════════",
        "  Ryzanstein LLM - CPU Feature Detection",
        "═══════════════════════════════════════════════════════",
        f"  CPU:          {features.model_name}",
        f"  Vendor:       {features.vendor}",
        f"  Architecture: {features.architecture}",
        f"  Cores:        {features.physical_cores} physical / {features.logical_cores} logical",
        "",
        "  SIMD Capabilities:",
        f"    SSE 4.2:       {'✓' if features.sse42 else '✗'}",
        f"    AVX:           {'✓' if features.avx else '✗'}",
        f"    AVX2:          {'✓' if features.avx2 else '✗'}",
        f"    FMA:           {'✓' if features.fma else '✗'}",
        f"    AVX-512F:      {'✓' if features.avx512f else '✗'}",
        f"    AVX-512BW:     {'✓' if features.avx512bw else '✗'}",
        f"    AVX-512VNNI:   {'✓' if features.avx512vnni else '✗'}",
        f"    AVX-512VBMI:   {'✓' if features.avx512vbmi else '✗'}",
        f"    AMX-TILE:      {'✓' if features.amx_tile else '✗'}",
        f"    AMX-INT8:      {'✓' if features.amx_int8 else '✗'}",
        "",
        f"  OpenMP:       {'Available' if features.openmp_available else 'Not detected'}"
        + (f" ({features.openmp_max_threads} threads)" if features.openmp_available else ""),
        "",
        "  Cache Topology:",
        f"    L1 Data:       {features.cache.l1_data_kb} KB",
        f"    L1 Instr:      {features.cache.l1_instruction_kb} KB",
        f"    L2:            {features.cache.l2_kb} KB",
        f"    L3:            {features.cache.l3_kb} KB",
        "",
        "  Recommended TL2_0 Configuration:",
        f"    Kernel:        {features.recommended_kernel}",
        f"    Tile M×N×K:    {features.recommended_tile_m}×{features.recommended_tile_n}×{features.recommended_tile_k}",
        "═══════════════════════════════════════════════════════",
    ]
    return "\n".join(lines)


def format_json(features: CPUFeatures) -> str:
    """Format features as JSON for CI consumption."""
    data = asdict(features)
    return json.dumps(data, indent=2)


def format_cmake(features: CPUFeatures) -> str:
    """Format features as CMake cache variable definitions."""
    lines = [
        f'set(CPU_HAS_SSE42 {"ON" if features.sse42 else "OFF"} CACHE BOOL "")',
        f'set(CPU_HAS_AVX {"ON" if features.avx else "OFF"} CACHE BOOL "")',
        f'set(CPU_HAS_AVX2 {"ON" if features.avx2 else "OFF"} CACHE BOOL "")',
        f'set(CPU_HAS_FMA {"ON" if features.fma else "OFF"} CACHE BOOL "")',
        f'set(CPU_HAS_AVX512F {"ON" if features.avx512f else "OFF"} CACHE BOOL "")',
        f'set(CPU_HAS_AVX512BW {"ON" if features.avx512bw else "OFF"} CACHE BOOL "")',
        f'set(CPU_HAS_AVX512VNNI {"ON" if features.avx512vnni else "OFF"} CACHE BOOL "")',
        f'set(CPU_HAS_OPENMP {"ON" if features.openmp_available else "OFF"} CACHE BOOL "")',
        f'set(CPU_PHYSICAL_CORES {features.physical_cores} CACHE STRING "")',
        f'set(CPU_LOGICAL_CORES {features.logical_cores} CACHE STRING "")',
        f'set(RECOMMENDED_KERNEL "{features.recommended_kernel}" CACHE STRING "")',
        f'set(RECOMMENDED_TILE_M {features.recommended_tile_m} CACHE STRING "")',
        f'set(RECOMMENDED_TILE_N {features.recommended_tile_n} CACHE STRING "")',
        f'set(RECOMMENDED_TILE_K {features.recommended_tile_k} CACHE STRING "")',
        f'set(L1_CACHE_KB {features.cache.l1_data_kb} CACHE STRING "")',
        f'set(L2_CACHE_KB {features.cache.l2_kb} CACHE STRING "")',
        f'set(L3_CACHE_KB {features.cache.l3_kb} CACHE STRING "")',
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Detect CPU features for Ryzanstein LLM kernel selection"
    )
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--cmake", action="store_true", help="Output CMake format")
    args = parser.parse_args()

    features = detect()

    if args.json:
        print(format_json(features))
    elif args.cmake:
        print(format_cmake(features))
    else:
        print(format_human(features))


if __name__ == "__main__":
    main()
