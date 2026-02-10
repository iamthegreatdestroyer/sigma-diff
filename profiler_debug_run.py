#!/usr/bin/env python3
"""
Debug execution - writes all output to file for analysis
"""
import sys
import time
from pathlib import Path

# Open debug log file FIRST
debug_log = Path(r"s:\Ryot\profiler_debug.log")
debug_file = open(debug_log, 'w', buffering=1)

def log(msg):
    """Write to both stdout and log file"""
    print(msg)
    debug_file.write(msg + '\n')
    debug_file.flush()

log("=" * 80)
log("PROFILER DEBUG EXECUTION")
log("=" * 80)

try:
    log(f"\n[LOG] Python version: {sys.version}")
    log(f"[LOG] Working directory: {Path.cwd()}")
    
    # Step 1: Setup paths
    scripts_dir = Path(r"s:\Ryot\RYZEN-LLM\scripts")
    log(f"[LOG] Scripts directory: {scripts_dir}")
    log(f"[LOG] Scripts exist: {scripts_dir.exists()}")
    
    if not scripts_dir.exists():
        log(f"[ERROR] Scripts directory not found!")
        sys.exit(1)
    
    sys.path.insert(0, str(scripts_dir))
    log(f"[LOG] Added to sys.path")
    
    # Step 2: Try importing PyTorch
    log(f"\n[LOG] Attempting PyTorch import...")
    try:
        import torch
        log(f"[OK] PyTorch: {torch.__version__}")
        log(f"[LOG] CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        log(f"[WARN] PyTorch import failed: {e}")
        torch = None
    
    # Step 3: Try importing NumPy
    log(f"\n[LOG] Attempting NumPy import...")
    try:
        import numpy as np
        log(f"[OK] NumPy: {np.__version__}")
    except ImportError as e:
        log(f"[ERROR] NumPy import failed: {e}")
        sys.exit(1)
    
    # Step 4: Try importing Phase 1 modules
    log(f"\n[LOG] Attempting Phase 1 module imports...")
    
    try:
        from kernel_optimizer import KernelOptimizer
        log(f"[OK] KernelOptimizer imported")
    except Exception as e:
        log(f"[WARN] KernelOptimizer import failed: {type(e).__name__}: {e}")
    
    try:
        from semantic_compression import SemanticCompressor
        log(f"[OK] SemanticCompressor imported")
    except Exception as e:
        log(f"[WARN] SemanticCompressor import failed: {type(e).__name__}: {e}")
    
    try:
        from inference_scaling import InferenceScalingEngine
        log(f"[OK] InferenceScalingEngine imported")
    except Exception as e:
        log(f"[WARN] InferenceScalingEngine import failed: {type(e).__name__}: {e}")
    
    # Step 5: Try importing benchmark_overhead
    log(f"\n[LOG] Attempting benchmark_overhead import...")
    try:
        from benchmark_overhead import OverheadProfiler
        log(f"[OK] OverheadProfiler imported")
    except Exception as e:
        log(f"[ERROR] OverheadProfiler import failed!")
        log(f"[ERROR] Type: {type(e).__name__}")
        log(f"[ERROR] Message: {e}")
        
        # Try to get more details
        import traceback
        log(f"\n[TRACEBACK]\n{traceback.format_exc()}")
        sys.exit(1)
    
    # Step 6: Check output directory
    log(f"\n[LOG] Checking output directories...")
    reports_dir = Path(r"s:\Ryot\reports")
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)
        log(f"[OK] Reports directory: {reports_dir}")
    except Exception as e:
        log(f"[ERROR] Failed to create reports dir: {e}")
        sys.exit(1)
    
    # Step 7: Execute profiler
    log(f"\n[LOG] Creating OverheadProfiler instance...")
    try:
        profiler = OverheadProfiler(
            num_iterations=100,
            activation_shape=(32, 1024)
        )
        log(f"[OK] Profiler instance created")
    except Exception as e:
        log(f"[ERROR] Failed to create profiler: {type(e).__name__}: {e}")
        import traceback
        log(f"\n[TRACEBACK]\n{traceback.format_exc()}")
        sys.exit(1)
    
    # Step 8: Run profiling
    log(f"\n[LOG] Starting profiler.run()...")
    try:
        JSON_OUTPUT = r"s:\Ryot\reports\overhead_analysis.json"
        MD_OUTPUT = r"s:\Ryot\OVERHEAD_ANALYSIS_REPORT.md"
        
        gate_pass = profiler.run(JSON_OUTPUT, MD_OUTPUT)
        log(f"[OK] Profiler.run() completed (gate_pass={gate_pass})")
    except Exception as e:
        log(f"[ERROR] Profiler.run() failed: {type(e).__name__}: {e}")
        import traceback
        log(f"\n[TRACEBACK]\n{traceback.format_exc()}")
        sys.exit(1)
    
    # Step 9: Verify output files
    log(f"\n[LOG] Verifying output files...")
    
    json_path = Path(JSON_OUTPUT)
    if json_path.exists():
        size = json_path.stat().st_size
        log(f"[OK] JSON file exists: {json_path} ({size} bytes)")
    else:
        log(f"[ERROR] JSON file NOT found: {json_path}")
    
    md_path = Path(MD_OUTPUT)
    if md_path.exists():
        size = md_path.stat().st_size
        log(f"[OK] Markdown file exists: {md_path} ({size} bytes)")
    else:
        log(f"[ERROR] Markdown file NOT found: {md_path}")
    
    log(f"\n[LOG] Profiling completed successfully!")

except Exception as e:
    log(f"\n[EXCEPTION] Unexpected error: {type(e).__name__}: {e}")
    import traceback
    log(f"{traceback.format_exc()}")
    sys.exit(1)

finally:
    debug_file.close()
    print(f"\n[COMPLETE] Debug log written to: {debug_log}")
