#!/usr/bin/env python3
"""
Debug script to diagnose why benchmark_overhead.py is not executing.
Tests: Python env, PyTorch availability, module imports, file paths.
"""

import sys
import os
import subprocess
from pathlib import Path

print("=" * 80)
print("OVERHEAD PROFILER DEBUGGING SCRIPT")
print("=" * 80)

# 1. Python Environment
print("\n[1] PYTHON ENVIRONMENT CHECK")
print(f"  Python executable: {sys.executable}")
print(f"  Python version: {sys.version}")
print(f"  Python path: {sys.path}")

# 2. PyTorch Check
print("\n[2] PYTORCH AVAILABILITY")
try:
    import torch
    print(f"  ✅ PyTorch version: {torch.__version__}")
    print(f"  ✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✅ CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"  ❌ PyTorch import failed: {e}")

# 3. NumPy Check
print("\n[3] NUMPY AVAILABILITY")
try:
    import numpy as np
    print(f"  ✅ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"  ❌ NumPy import failed: {e}")

# 4. Phase 1 Module Imports
print("\n[4] PHASE 1 MODULE IMPORTS")
scripts_dir = Path("S:\\Ryot\\RYZEN-LLM\\scripts")
print(f"  Scripts directory: {scripts_dir}")
print(f"  Exists: {scripts_dir.exists()}")

if scripts_dir.exists():
    # Add to path
    sys.path.insert(0, str(scripts_dir))
    
    modules_to_test = [
        ("kernel_optimizer", "KernelOptimizer"),
        ("semantic_compression", "SemanticCompressor"),
        ("inference_scaling", "InferenceScalingEngine")
    ]
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            cls = getattr(module, class_name, None)
            if cls:
                print(f"  ✅ {module_name}: Imported {class_name}")
            else:
                print(f"  ⚠️  {module_name}: Module exists, but {class_name} not found")
        except ImportError as e:
            print(f"  ❌ {module_name}: Import failed - {e}")
        except Exception as e:
            print(f"  ❌ {module_name}: Unexpected error - {e}")

# 5. Output Directory
print("\n[5] OUTPUT DIRECTORY CHECK")
reports_dir = Path("s:\\Ryot\\reports")
print(f"  Reports directory: {reports_dir}")
print(f"  Exists: {reports_dir.exists()}")
if reports_dir.exists():
    print(f"  Writable: {os.access(str(reports_dir), os.W_OK)}")
    files_in_dir = list(reports_dir.glob("*"))
    print(f"  Files in directory: {len(files_in_dir)}")
    for f in files_in_dir[:5]:
        print(f"    - {f.name}")

# 6. File Paths for Output
print("\n[6] FILE PATHS FOR OUTPUT")
json_path = Path("s:\\Ryot\\reports\\overhead_analysis.json")
md_path = Path("s:\\Ryot\\OVERHEAD_ANALYSIS_REPORT.md")
print(f"  JSON output: {json_path}")
print(f"  JSON exists: {json_path.exists()}")
print(f"  Markdown output: {md_path}")
print(f"  Markdown exists: {md_path.exists()}")

# 7. Try importing benchmark_overhead module
print("\n[7] BENCHMARK_OVERHEAD.PY CHECK")
benchmark_path = scripts_dir / "benchmark_overhead.py"
print(f"  Script path: {benchmark_path}")
print(f"  Exists: {benchmark_path.exists()}")

if benchmark_path.exists():
    try:
        # Check for syntax errors
        with open(benchmark_path, 'r') as f:
            code = f.read()
        compile(code, str(benchmark_path), 'exec')
        print(f"  ✅ Script syntax is valid")
        
        # Try to find OverheadProfiler class
        if "class OverheadProfiler" in code:
            print(f"  ✅ OverheadProfiler class found in script")
        else:
            print(f"  ❌ OverheadProfiler class NOT found in script")
            
    except SyntaxError as e:
        print(f"  ❌ Syntax error in script: {e}")

# 8. Try running a simple test
print("\n[8] SIMPLE EXECUTION TEST")
print("  Attempting to import and run OverheadProfiler...")

os.chdir(str(scripts_dir))
try:
    # Execute the benchmark script in subprocess  to see actual error
    result = subprocess.run(
        [sys.executable, "benchmark_overhead.py"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(scripts_dir)
    )
    
    print(f"  Return code: {result.returncode}")
    
    if result.stdout:
        print(f"\n  STDOUT ({len(result.stdout)} chars):")
        print("  " + "\n  ".join(result.stdout[:500].split("\n")))
        if len(result.stdout) > 500:
            print(f"  ... ({len(result.stdout) - 500} more chars)")
    
    if result.stderr:
        print(f"\n  STDERR ({len(result.stderr)} chars):")
        print("  " + "\n  ".join(result.stderr[:500].split("\n")))
        if len(result.stderr) > 500:
            print(f"  ... ({len(result.stderr) - 500} more chars)")
            
except subprocess.TimeoutExpired:
    print(f"  ⚠️  Script timed out after 30 seconds")
except Exception as e:
    print(f"  ❌ Execution failed: {e}")

# 9. Final check for output files
print("\n[9] FINAL OUTPUT CHECK")
if json_path.exists():
    print(f"  ✅ JSON file created: {json_path.stat().st_size} bytes")
else:
    print(f"  ❌ JSON file NOT created")

if md_path.exists():
    print(f"  ✅ Markdown file created: {md_path.stat().st_size} bytes")
else:
    print(f"  ❌ Markdown file NOT created")

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
