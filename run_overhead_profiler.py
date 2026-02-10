#!/usr/bin/env python3
"""
Simple wrapper to run overhead profiler and capture output cleanly
"""

import subprocess
import sys
from pathlib import Path

# First, let's test basic imports
print("=" * 80)
print("STEP 1: Testing Python Environment")
print("=" * 80)

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Test PyTorch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("❌ PyTorch not available")

# Test NumPy
try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError:
    print("❌ NumPy not available")

# Test Phase 1 modules
print("\nSTEP 2: Testing Phase 1 Module Imports")
print("=" * 80)

scripts_dir = Path("S:\\Ryot\\RYZEN-LLM\\scripts")
sys.path.insert(0, str(scripts_dir))

for module_name in ["kernel_optimizer", "semantic_compression", "inference_scaling"]:
    try:
        __import__(module_name)
        print(f"✅ {module_name} imports successfully")
    except ImportError as e:
        print(f"❌ {module_name} import failed: {e}")

# Check output directories
print("\nSTEP 3: Checking Output Directories")
print("=" * 80)

reports_dir = Path("s:\\Ryot\\reports")
print(f"Reports directory exists: {reports_dir.exists()}")
print(f"Reports writable: {reports_dir.exists() and __import__('os').access(str(reports_dir), __import__('os').W_OK)}")

json_path = Path("s:\\Ryot\\reports\\overhead_analysis.json")
md_path = Path("s:\\Ryot\\OVERHEAD_ANALYSIS_REPORT.md")
print(f"JSON output path: {json_path}")
print(f"Markdown output path: {md_path}")

# Now run the benchmark
print("\nSTEP 4: Running Overhead Profiler")
print("=" * 80)

benchmark_path = scripts_dir / "benchmark_overhead.py"
print(f"Running: {benchmark_path}")
print(f"Script exists: {benchmark_path.exists()}")

result = subprocess.run(
    [sys.executable, str(benchmark_path)],
    cwd=str(scripts_dir),
    capture_output=True,
    text=True,
    timeout=120
)

print(f"\nReturn code: {result.returncode}")

if result.stdout:
    print("\n--- STDOUT ---")
    print(result.stdout)

if result.stderr:
    print("\n--- STDERR ---")
    print(result.stderr)

# Check if files were created
print("\nSTEP 5: Verifying Output Files")
print("=" * 80)

if json_path.exists():
    size = json_path.stat().st_size
    print(f"✅ JSON file created ({size} bytes)")
    # Print first 500 chars of JSON
    with open(json_path, 'r') as f:
        content = f.read()
        print(f"   First 200 chars: {content[:200]}")
else:
    print(f"❌ JSON file NOT created at {json_path}")

if md_path.exists():
    size = md_path.stat().st_size
    print(f"✅ Markdown file created ({size} bytes)")
else:
    print(f"❌ Markdown file NOT created at {md_path}")

print("\n" + "=" * 80)
print("WRAPPER COMPLETE")
print("=" * 80)
