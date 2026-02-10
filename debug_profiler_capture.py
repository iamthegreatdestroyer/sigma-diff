#!/usr/bin/env python3
"""
Debug wrapper that captures ALL output to a file
"""
import sys
import os
from pathlib import Path
from io import StringIO


# Redirect all output to file
output_file = open(r"s:\Ryot\profiler_execution.log", "w")
original_stdout = sys.stdout
original_stderr = sys.stderr

# Create tee writer that writes to both file and console
class TeeWriter:
    def __init__(self, *writers):
        self.writers = writers
    
    def write(self, data):
        for w in self.writers:
            w.write(data)
            w.flush()
    
    def flush(self):
        for w in self.writers:
            if hasattr(w, 'flush'):
                w.flush()

sys.stdout = TeeWriter(original_stdout, output_file)
sys.stderr = TeeWriter(original_stderr, output_file)

print("=== PROFILER DEBUG EXECUTION ===")
print(f"CWD: {os.getcwd()}")
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    # Setup paths
    scripts_dir = Path(r"s:\Ryot\RYZEN-LLM\scripts")
    print(f"\nScripts dir: {scripts_dir}")
    print(f"Exists: {scripts_dir.exists()}")
    
    # Check requisite files
    benchmark_file = scripts_dir / "benchmark_overhead.py"
    print(f"benchmark_overhead.py exists: {benchmark_file.exists()}")
    
    if not benchmark_file.exists():
        print("ERROR: benchmark_overhead.py not found!")
        sys.exit(1)
    
    sys.path.insert(0, str(scripts_dir))
    print(f"sys.path[0]: {sys.path[0]}")
    
    print("\n--- IMPORTS ---")
    
    # Test PyTorch
    print("Testing PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch available: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        HAS_TORCH = True
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        HAS_TORCH = False
    
    # Test NumPy
    print("Testing NumPy...")
    try:
        import numpy as np
        print(f"✓ NumPy available: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        sys.exit(1)
    
    # Test Phase 1 modules
    print("\nTesting Phase 1 modules...")
    
    print("  Importing kernel_optimizer...")
    try:
        from kernel_optimizer import KernelOptimizer
        print("  ✓ kernel_optimizer")
    except Exception as e:
        print(f"  ✗ kernel_optimizer: {e}")
    
    print("  Importing semantic_compression...")
    try:
        from semantic_compression import SemanticCompressor
        print("  ✓ semantic_compression")
    except Exception as e:
        print(f"  ✗ semantic_compression: {e}")
    
    print("  Importing inference_scaling...")
    try:
        from inference_scaling import InferenceScalingEngine
        print("  ✓ inference_scaling")
    except Exception as e:
        print(f"  ✗ inference_scaling: {e}")
    
    print("\n--- MAIN EXECUTION ---")
    
    # Import profiler
    print("Importing OverheadProfiler from benchmark_overhead...")
    from benchmark_overhead import OverheadProfiler
    print("✓ OverheadProfiler imported successfully")
    
    # Create profiler
    print("\nCreating OverheadProfiler instance...")
    profiler = OverheadProfiler(
        num_iterations=100,
        activation_shape=(32, 1024)
    )
    print("✓ Profiler instance created")
    
    # Configuration
    JSON_OUTPUT = r"s:\Ryot\reports\overhead_analysis.json"
    MD_OUTPUT = r"s:\Ryot\OVERHEAD_ANALYSIS_REPORT.md"
    
    print(f"\nConfiguration:")
    print(f"  JSON output: {JSON_OUTPUT}")
    print(f"  MD output: {MD_OUTPUT}")
    
    # Create output directory
    output_dir = Path(JSON_OUTPUT).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir exists: {output_dir.exists()}")
    
    # Run profiling
    print("\n--- RUNNING PROFILER ---")
    gate_pass = profiler.run(JSON_OUTPUT, MD_OUTPUT)
    
    print(f"\n✓ Profiler completed")
    print(f"  Gate pass: {gate_pass}")
    
    # Verify files
    print("\n--- VERIFYING OUTPUT FILES ---")
    json_path = Path(JSON_OUTPUT)
    md_path = Path(MD_OUTPUT)
    
    print(f"JSON file: {JSON_OUTPUT}")
    if json_path.exists():
        size = json_path.stat().st_size
        print(f"  ✓ EXISTS ({size} bytes)")
        # Try to read it
        try:
            import json
            with open(json_path) as f:
                data = json.load(f)
            print(f"  ✓ Valid JSON")
            print(f"  Keys: {list(data.keys())}")
        except Exception as e:
            print(f"  ✗ Error reading JSON: {e}")
    else:
        print(f"  ✗ NOT FOUND")
    
    print(f"\nMarkdown file: {MD_OUTPUT}")
    if md_path.exists():
        size = md_path.stat().st_size
        print(f"  ✓ EXISTS ({size} bytes)")
    else:
        print(f"  ✗ NOT FOUND")
    
    print("\n=== EXECUTION COMPLETE ===")

except Exception as e:
    print(f"\n✗ FATAL ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    output_file.close()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    print(f"\nDebug log written to: s:\\Ryot\\profiler_execution.log")
