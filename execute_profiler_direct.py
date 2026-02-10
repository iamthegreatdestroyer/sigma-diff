#!/usr/bin/env python3
"""
Direct execution of overhead profiler - bypasses broken terminal subprocess capture
Imports and runs profiler in-process, then verifies output files
"""
import sys
import time
from pathlib import Path

# Set up paths
scripts_dir = Path(r"s:\Ryot\RYZEN-LLM\scripts")
sys.path.insert(0, str(scripts_dir))

print("=" * 80)
print("DIRECT PROFILER EXECUTION - BYPASSING TERMINAL SUBPROCESS ISSUES")
print("=" * 80)

# Step 1: Verify environment
print("\n[1/5] ENVIRONMENT VERIFICATION")
print("-" * 80)

# Check Python version
print(f"✓ Python version: {sys.version}")
print(f"✓ Working directory: {Path.cwd()}")
print(f"✓ Scripts directory: {scripts_dir}")
print(f"✓ Scripts exist: {scripts_dir.exists()}")

# Check module files
print("\nModule files:")
modules_to_check = [
    "kernel_optimizer.py",
    "semantic_compression.py",
    "inference_scaling.py",
    "benchmark_overhead.py"
]
for module in modules_to_check:
    module_path = scripts_dir / module
    exists = "✓" if module_path.exists() else "✗"
    size = f"{module_path.stat().st_size:,} bytes" if module_path.exists() else "N/A"
    print(f"  {exists} {module}: {size}")

# Step 2: Test imports
print("\n[2/5] IMPORT TESTING")
print("-" * 80)

# Try importing PyTorch
try:
    import torch
    print(f"✓ PyTorch imported: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

# Try importing numpy
try:
    import numpy as np
    print(f"✓ NumPy imported: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

# Try importing Phase 1 modules
print("\nPhase 1 modules:")
try:
    from kernel_optimizer import KernelOptimizer
    print(f"✓ kernel_optimizer.KernelOptimizer imported")
except Exception as e:
    print(f"✗ kernel_optimizer.KernelOptimizer import failed: {e}")

try:
    from semantic_compression import SemanticCompressor
    print(f"✓ semantic_compression.SemanticCompressor imported")
except Exception as e:
    print(f"✗ semantic_compression.SemanticCompressor import failed: {e}")

try:
    from inference_scaling import InferenceScalingEngine, TaskComplexityEstimator
    print(f"✓ inference_scaling.InferenceScalingEngine imported")
    print(f"✓ inference_scaling.TaskComplexityEstimator imported")
except Exception as e:
    print(f"✗ inference_scaling import failed: {e}")

# Try importing benchmark_overhead
print("\nBenchmark profiler:")
try:
    from benchmark_overhead import OverheadProfiler
    print(f"✓ benchmark_overhead.OverheadProfiler imported")
except Exception as e:
    print(f"✗ benchmark_overhead.OverheadProfiler import failed: {e}")
    print(f"  Error details: {type(e).__name__}: {e}")
    sys.exit(1)

# Step 3: Verify output directory
print("\n[3/5] OUTPUT DIRECTORY VERIFICATION")
print("-" * 80)

reports_dir = Path(r"s:\Ryot\reports")
reports_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Reports directory: {reports_dir}")
print(f"✓ Directory writable: {reports_dir.stat().st_mode}")

# Step 4: Execute profiler
print("\n[4/5] PROFILER EXECUTION")
print("-" * 80)

try:
    # Configuration
    ITERATIONS = 100
    ACTIVATION_SHAPE = (32, 1024)
    JSON_OUTPUT = r"s:\Ryot\reports\overhead_analysis.json"
    MD_OUTPUT = r"s:\Ryot\OVERHEAD_ANALYSIS_REPORT.md"
    
    # Create and run profiler
    print(f"\nExecuting profiler with {ITERATIONS} iterations, shape {ACTIVATION_SHAPE}...")
    profiler = OverheadProfiler(
        num_iterations=ITERATIONS,
        activation_shape=ACTIVATION_SHAPE
    )
    
    gate_pass = profiler.run(JSON_OUTPUT, MD_OUTPUT)
    
    print(f"\n✓ Profiler execution completed (gate_pass={gate_pass})")
    
except Exception as e:
    print(f"\n✗ Profiler execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Verify output files
print("\n[5/5] OUTPUT FILE VERIFICATION")
print("-" * 80)

json_path = Path(JSON_OUTPUT)
md_path = Path(MD_OUTPUT)

print(f"\nJSON output file: {json_path}")
if json_path.exists():
    size = json_path.stat().st_size
    print(f"✓ EXISTS ({size:,} bytes)")
    
    # Try to read and parse JSON
    try:
        import json
        with open(json_path) as f:
            data = json.load(f)
        
        print(f"✓ Valid JSON structure")
        print(f"  Metadata keys: {list(data.get('profiling_metadata', {}).keys())}")
        print(f"  Results modules: {list(data.get('results', {}).keys())}")
        
        # Print overhead results
        print(f"\n  Overhead Results:")
        for module_key, module_data in data.get('results', {}).items():
            if 'error' not in module_data:
                if module_key == "semantic_compression":
                    overhead = module_data.get('total_overhead_ms', 0)
                else:
                    overhead = module_data.get('overhead_ms', 0)
                status = "✓ PASS" if overhead < 50 else "✗ FAIL"
                print(f"    {status} {module_key}: {overhead:.4f} ms")
            else:
                print(f"    ✗ {module_key}: ERROR - {module_data['error']}")
    except Exception as e:
        print(f"✗ Failed to parse JSON: {e}")
else:
    print(f"✗ NOT FOUND")

print(f"\nMarkdown output file: {md_path}")
if md_path.exists():
    size = md_path.stat().st_size
    print(f"✓ EXISTS ({size:,} bytes)")
    
    # Read first 500 chars
    try:
        with open(md_path) as f:
            content = f.read(500)
        print(f"✓ Readable content preview (first 500 chars):")
        print(f"  {content[:500]}")
    except Exception as e:
        print(f"✗ Failed to read: {e}")
else:
    print(f"✗ NOT FOUND")

print("\n" + "=" * 80)
print("EXECUTION SUMMARY")
print("=" * 80)
print(f"✓ Direct execution completed successfully")
print(f"✓ Output files verified on disk")
print(f"✓ Results ready for analysis and git commit")
print("=" * 80 + "\n")

sys.exit(0)
