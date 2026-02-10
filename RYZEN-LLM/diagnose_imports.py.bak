#!/usr/bin/env python3
"""Diagnose import chain for Task 5"""
import sys
import traceback

sys.path.insert(0, '.')

print("=" * 60)
print("RYZEN-LLM Import Diagnostics")
print("=" * 60)

# Step 1: Test basic package structure
print("\n[1] Testing src.core package...")
try:
    import src.core
    print("    ✓ src.core package found")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    traceback.print_exc()

# Step 2: Test quantization module (dependency of weight_loader)
print("\n[2] Testing src.core.quantization...")
try:
    from src.core import quantization
    print("    ✓ quantization module imported")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    traceback.print_exc()

# Step 3: Test QuantizationEngine class
print("\n[3] Testing QuantizationEngine class...")
try:
    from src.core.quantization import QuantizationEngine
    print("    ✓ QuantizationEngine class available")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    traceback.print_exc()

# Step 4: Test create_aggressive_config
print("\n[4] Testing create_aggressive_config...")
try:
    from src.core.quantization import create_aggressive_config
    print("    ✓ create_aggressive_config available")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    traceback.print_exc()

# Step 5: Test weight_loader module
print("\n[5] Testing src.core.weight_loader...")
try:
    from src.core import weight_loader
    print("    ✓ weight_loader module imported")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    traceback.print_exc()

# Step 6: Test load_weights function
print("\n[6] Testing load_weights function...")
try:
    from src.core.weight_loader import load_weights
    print("    ✓ load_weights function available")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    traceback.print_exc()

# Step 7: Test WeightLoaderConfig class
print("\n[7] Testing WeightLoaderConfig class...")
try:
    from src.core.weight_loader import WeightLoaderConfig
    print("    ✓ WeightLoaderConfig class available")
except Exception as e:
    print(f"    ✗ FAILED: {e}")
    traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
