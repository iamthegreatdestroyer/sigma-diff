#!/usr/bin/env python3
"""Debug script to test calling main() directly"""

import sys
import os

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RYZEN-LLM", "scripts"))

print("=" * 60)
print("DEBUG: Calling main() directly")
print("=" * 60)

# Mock sys.argv to simulate command-line arguments
sys.argv = [
    'training_loop.py',
    '--config', 'training_configuration.yaml',
    '--epochs', '1',
    '--no-optimization'
]
print(f"\nCommand line args: {sys.argv[1:]}")

try:
    print("\nImporting training_loop module...")
    import training_loop
    print("  ✓ Module imported")
    
    print("\nCalling main()...")
    print("-" * 60)
    training_loop.main()
    print("-" * 60)
    
    print("  ✓ main() completed")
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR:")
    print(f"  Type: {type(e).__name__}")
    print(f"  Message: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
