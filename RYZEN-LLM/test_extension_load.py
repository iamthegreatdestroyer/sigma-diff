#!/usr/bin/env python3
"""
Quick test to validate C++ extension loads and functions
"""
import sys
import os
from pathlib import Path

# Add build directory to Python path
build_dir = Path(__file__).parent / "build" / "python"
sys.path.insert(0, str(build_dir))
sys.path.insert(0, str(build_dir / "ryzanstein_llm"))

print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}")
print(f"Looking for extension in: {build_dir}")

try:
    # Try to import the compiled extension
    import ryzen_llm_bindings as bindings
    print("✅ Successfully imported ryzen_llm_bindings C++ extension")
    
    # Check available attributes
    attrs = dir(bindings)
    print(f"\nAvailable functions in bindings module:")
    for attr in attrs:
        if not attr.startswith("_"):
            print(f"  - {attr}")
    
    # Test basic function
    if hasattr(bindings, 'test_function'):
        result = bindings.test_function()
        print(f"\n✅ test_function() returned: {result}")
        assert result == 42, f"Expected 42, got {result}"
    else:
        print("\nℹ️ test_function not exposed in pybind11 module (in extern C block)")
    
    print("\n✅ C++ extension successfully compiled and loaded!")
    print("Extension is ready for use.")
    
except ImportError as e:
    print(f"❌ Failed to import C++ extension: {e}")
    print(f"\nAvailable files in {build_dir}:")
    if build_dir.exists():
        for f in build_dir.rglob("*"):
            if f.is_file():
                print(f"  {f.relative_to(build_dir)}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error testing C++ extension: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

