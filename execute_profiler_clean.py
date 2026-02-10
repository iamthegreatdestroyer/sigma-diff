#!/usr/bin/env python3
"""
Clean execution wrapper - runs profiler and logs results directly to file
Avoids terminal history contamination by using subprocess isolation
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# Setup paths
script_dir = Path(__file__).parent
profiler_script = script_dir / "RYZEN-LLM" / "scripts" / "benchmark_overhead.py"
output_log = script_dir / "profiler_execution_log.txt"
json_report = script_dir / "reports" / "overhead_analysis.json"
md_report = script_dir / "OVERHEAD_ANALYSIS_REPORT.md"

# Write all output to log file
with open(output_log, 'w') as log:
    timestamp = datetime.now().isoformat()
    log.write(f"=== PROFILER EXECUTION LOG ===\n")
    log.write(f"Timestamp: {timestamp}\n")
    log.write(f"Script: {profiler_script}\n")
    log.write(f"Working Directory: {script_dir / 'RYZEN-LLM'}\n")
    log.write(f"Python: {sys.executable}\n")
    log.write(f"Python Version: {sys.version}\n\n")
    
    # Test 1: Check profiler exists
    log.write(f"[TEST 1] Profiler script exists: {profiler_script.exists()}\n")
    if profiler_script.exists():
        log.write(f"         File size: {profiler_script.stat().st_size} bytes\n")
    log.write("\n")
    
    # Test 2: Check script directory
    scripts_dir = script_dir / "RYZEN-LLM" / "scripts"
    log.write(f"[TEST 2] Scripts directory exists: {scripts_dir.exists()}\n")
    if scripts_dir.exists():
        log.write(f"         Contents: {list(scripts_dir.glob('*.py'))}\n")
    log.write("\n")
    
    # Test 3: Check output directory
    reports_dir = script_dir / "reports"
    log.write(f"[TEST 3] Reports directory exists: {reports_dir.exists()}\n")
    if reports_dir.exists():
        log.write(f"         Is writable: {os.access(reports_dir, os.W_OK)}\n")
        log.write(f"         Current contents: {list(reports_dir.glob('*'))}\n")
    log.write("\n")
    
    # Test 4: Execute profiler
    log.write(f"[TEST 4] Executing profiler...\n")
    log.write(f"         Command: {sys.executable} {profiler_script}\n")
    log.write(f"         From directory: {script_dir / 'RYZEN-LLM'}\n")
    log.write("-" * 80 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(profiler_script)],
            cwd=str(script_dir / "RYZEN-LLM"),
            capture_output=True,
            text=True,
            timeout=120
        )
        
        log.write("STDOUT:\n")
        log.write(result.stdout)
        log.write("\n\nSTDERR:\n")
        log.write(result.stderr)
        log.write(f"\n\nReturn code: {result.returncode}\n")
        
    except subprocess.TimeoutExpired:
        log.write("ERROR: Subprocess timeout (120 seconds exceeded)\n")
    except Exception as e:
        log.write(f"ERROR: {type(e).__name__}: {e}\n")
    
    log.write("\n" + "-" * 80 + "\n")
    log.write("[TEST 5] Post-execution file check:\n")
    
    # Check results
    json_exists = json_report.exists()
    md_exists = md_report.exists()
    
    log.write(f"         JSON report exists: {json_exists}\n")
    if json_exists:
        log.write(f"         JSON file size: {json_report.stat().st_size} bytes\n")
    
    log.write(f"         Markdown report exists: {md_exists}\n")
    if md_exists:
        log.write(f"         Markdown file size: {md_report.stat().st_size} bytes\n")
    
    log.write("\n")
    log.write("[SUMMARY]\n")
    log.write(f"Profiler executed: {'YES' if result.returncode is not None else 'UNKNOWN'}\n")
    log.write(f"Return code: {result.returncode if result.returncode is not None else 'N/A'}\n")
    log.write(f"Output files created: {json_exists and md_exists}\n")
    log.write(f"Execution successful: {'YES - All files created' if json_exists and md_exists else 'NO - Files not created'}\n")

print(f"✅ Execution log written to: {output_log}")
print(f"\nTo view results, read: {output_log}")
