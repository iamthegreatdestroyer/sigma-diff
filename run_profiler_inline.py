#!/usr/bin/env python3
"""
Direct in-process execution of profiler - NO subprocess
"""
import sys
from pathlib import Path

# Setup paths
scripts_dir = Path(r"s:\Ryot\RYZEN-LLM\scripts")
sys.path.insert(0, str(scripts_dir))

print("Starting profiler execution...")

try:
    # Import profiler
    from benchmark_overhead import OverheadProfiler
    print("✓ OverheadProfiler imported")
    
    # Create profiler
    profiler = OverheadProfiler(
        num_iterations=100,
        activation_shape=(32, 1024)
    )
    print("✓ Profiler instance created")
    
    # Configuration
    JSON_OUTPUT = r"s:\Ryot\reports\overhead_analysis.json"
    MD_OUTPUT = r"s:\Ryot\OVERHEAD_ANALYSIS_REPORT.md"
    
    # Run profiling
    print("\nRunning profiler...")
    gate_pass = profiler.run(JSON_OUTPUT, MD_OUTPUT)
    
    print(f"\n✓ Profiler completed (gate_pass={gate_pass})")
    
    # Verify files
    print("\nVerifying output files...")
    json_path = Path(JSON_OUTPUT)
    md_path = Path(MD_OUTPUT)
    
    if json_path.exists():
        print(f"✓ JSON file created: {json_path.stat().st_size} bytes")
    else:
        print(f"✗ JSON file NOT created: {JSON_OUTPUT}")
    
    if md_path.exists():
        print(f"✓ Markdown file created: {md_path.stat().st_size} bytes")
    else:
        print(f"✗ Markdown file NOT created: {MD_OUTPUT}")
    
    print("\n✓ EXECUTION COMPLETE")

except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
