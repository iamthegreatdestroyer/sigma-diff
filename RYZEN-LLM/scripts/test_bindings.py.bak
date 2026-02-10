#!/usr/bin/env python3
"""Test C++ bindings import."""
import sys
import os
import traceback

# Add the bindings path
bindings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'python', 'ryzen_llm')
print(f"Adding path: {bindings_path}")
sys.path.insert(0, bindings_path)

# Check if file exists
pyd_file = os.path.join(bindings_path, 'ryzen_llm_bindings.pyd')
print(f"PYD file exists: {os.path.exists(pyd_file)}")

try:
    print("Attempting import...")
    import ryzen_llm_bindings as rlb
    print("SUCCESS! Import worked.")
    print(f"Module: {rlb}")
    print(f"Available: {dir(rlb)}")
except ImportError as e:
    print(f"ImportError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Error ({type(e).__name__}): {e}")
    traceback.print_exc()
