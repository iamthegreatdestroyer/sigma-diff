#!/usr/bin/env python3
"""
SIMD Detection Test Script
Tests SIMD compilation flags and CPU capabilities without full module build.
"""

import platform
import subprocess
import sys
from pathlib import Path

def check_cpu_capabilities():
    """Check CPU SIMD capabilities."""
    print("=== CPU SIMD CAPABILITY CHECK ===")

    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        flags = cpu.get('flags', [])

        avx2 = 'avx2' in flags
        avx512f = 'avx512f' in flags
        fma = 'fma' in flags

        print(f"AVX2 supported: {avx2}")
        print(f"AVX-512F supported: {avx512f}")
        print(f"FMA supported: {fma}")

        return avx2, avx512f, fma
    except ImportError:
        print("cpuinfo not available, checking via system commands...")
        try:
            result = subprocess.run(['wmic', 'cpu', 'get', 'caption'], capture_output=True, text=True)
            cpu_info = result.stdout.lower()
            avx2 = 'avx2' in cpu_info
            print(f"AVX2 detected in CPU info: {avx2}")
            return avx2, False, False
        except:
            print("Could not detect CPU capabilities")
            return False, False, False

def check_compilation_flags():
    """Check if SIMD flags are properly set in CMake."""
    print("\n=== COMPILATION FLAG CHECK ===")

    cmake_path = Path("RYZEN-LLM/CMakeLists.txt")
    if not cmake_path.exists():
        print("❌ CMakeLists.txt not found")
        return False, False

    content = cmake_path.read_text()

    avx2_flags = "-mavx2" in content or "/arch:AVX2" in content
    avx512_flags = "-mavx512" in content or "/arch:AVX512" in content

    print(f"AVX2 compilation flags: {'✅' if avx2_flags else '❌'}")
    print(f"AVX-512 compilation flags: {'✅' if avx512_flags else '❌'}")

    return avx2_flags, avx512_flags

def check_header_definitions():
    """Check SIMD macro definitions in header."""
    print("\n=== HEADER SIMD DEFINITIONS ===")

    header_path = Path("RYZEN-LLM/src/core/tmac/lut_gemm.h")
    if not header_path.exists():
        print("❌ Header file not found")
        return False, False

    content = header_path.read_text()

    avx2_constructor = "#elif defined(__AVX2__)" in content
    avx512_constructor = "#ifdef __AVX512F__" in content

    print(f"AVX2 constructor logic: {'✅' if avx2_constructor else '❌'}")
    print(f"AVX-512 constructor logic: {'✅' if avx512_constructor else '❌'}")

    return avx2_constructor, avx512_constructor

def main():
    print("SIMD Detection Test Script")
    print("=" * 50)

    # Check CPU capabilities
    cpu_avx2, cpu_avx512, cpu_fma = check_cpu_capabilities()

    # Check compilation setup
    cmake_avx2, cmake_avx512 = check_compilation_flags()

    # Check header definitions
    header_avx2, header_avx512 = check_header_definitions()

    print("\n=== SUMMARY ===")

    if cpu_avx2 and cmake_avx2 and header_avx2:
        print("✅ AVX2 support should be active")
        print("   Expected speedup: 4-6× over scalar")
    elif cpu_avx512 and cmake_avx512 and header_avx512:
        print("✅ AVX-512 support should be active")
        print("   Expected speedup: 6-8× over scalar")
    else:
        print("❌ SIMD support not properly configured")
        if not cpu_avx2:
            print("   Reason: CPU does not support AVX2")
        if not cmake_avx2 and not cmake_avx512:
            print("   Reason: Compilation flags not set")
        if not header_avx2 and not header_avx512:
            print("   Reason: Header logic missing")

    print("\nNext steps:")
    print("1. If AVX2/AVX-512 should work: rebuild the extension")
    print("2. If CPU doesn't support: optimize scalar implementation")
    print("3. Test with: python -c \"import ryzanstein_llm; ryzanstein_llm.check_simd_status()\"")

if __name__ == "__main__":
    main()