#!/usr/bin/env python3
"""
VELOCITY ultra-minimal environment verification
Direct in-process testing, no subprocess contamination
"""
import sys
import os
from pathlib import Path

print("=== VELOCITY MINIMAL ENV TEST ===\n")

# 1. Python environment
print(f"[1] Python Executable: {sys.executable}")
print(f"[2] Python Version: {sys.version}")
print(f"[3] Current Directory: {os.getcwd()}")
print(f"[4] sys.path has {len(sys.path)} entries\n")

# 2. Workspace structure  
ryot_dir = Path("s:\\Ryot")
scripts_subdir = ryot_dir / "RYZEN-LLM" / "scripts"
reports_dir = ryot_dir / "reports"

print(f"[5] Ryot exists: {ryot_dir.exists()}")
print(f"[6] Scripts dir exists: {scripts_subdir.exists()}")
print(f"[7] Reports dir exists: {reports_dir.exists()}\n")

# 3. Module availability
print("[8] Testing imports:")
try:
    import numpy as np
    print(f"    ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"    ✗ NumPy: {e}")

try:
    import torch
    print(f"    ✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"    ✗ PyTorch: {e}")

# 4. Phase 1 modules
print("\n[9] Testing Phase 1 modules:")
sys.path.insert(0, str(scripts_subdir))

for mod_name in ["kernel_optimizer", "semantic_compression", "inference_scaling"]:
    try:
        __import__(mod_name)
        print(f"    ✓ {mod_name}")
    except ImportError as e:
        print(f"    ✗ {mod_name}: {e}")
    except Exception as e:
        print(f"    ? {mod_name}: {type(e).__name__}: {str(e)[:60]}")

# 5. Benchmark script check
benchmark_py = scripts_subdir / "benchmark_overhead.py"
print(f"\n[10] benchmark_overhead.py exists: {benchmark_py.exists()}")
if benchmark_py.exists():
    print(f"     File size: {benchmark_py.stat().st_size} bytes")
    # Try to read first 100 chars
    with open(benchmark_py, 'r') as f:
        header = f.read(200)
        print(f"     Starts with: {header.split(chr(10))[0]}")

# 6. Try to import profiler class
print("\n[11] Attempting to import OverheadProfiler:")
try:
    # This is the key test - can we even load the profiler module?
    from benchmark_overhead import OverheadProfiler
    print("    ✓ OverheadProfiler imported successfully!")
    
    # If we got here, try to instantiate it
    profiler = OverheadProfiler()
    print("    ✓ OverheadProfiler instantiated successfully!")
    
except ImportError as e:
    print(f"    ✗ Import failed: {e}")
except SyntaxError as e:
    print(f"    ✗ Syntax error: {e}")
except Exception as e:
    print(f"    ? {type(e).__name__}: {e}")

print("\n=== END TEST ===")
