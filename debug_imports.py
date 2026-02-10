#!/usr/bin/env python3
import sys
from pathlib import Path

print("Step 1: Starting imports...")

SCRIPTS_DIR = Path('s:\\Ryot\\RYZEN-LLM\\scripts')
sys.path.insert(0, str(SCRIPTS_DIR))

print("Step 2: Path configured")

try:
    print("Step 3: Importing training_metrics_collector...")
    from training_metrics_collector import TrainingMetricsCollector
    print("  ✓ TrainingMetricsCollector imported")
except Exception as e:
    print(f"  ✗ Failed: {e}")

try:
    print("Step 4: Importing optimization_controller...")
    from optimization_controller import OptimizationController
    print("  ✓ OptimizationController imported")
except Exception as e:
    print(f"  ✗ Failed: {e}")

try:
    print("Step 5: Importing optimization_orchestrator...")
    from optimization_orchestrator import OptimizationOrchestrator
    print("  ✓ OptimizationOrchestrator imported")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print("Step 6: All imports complete!")
