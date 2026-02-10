#!/usr/bin/env python3
"""Debug script to test full run_training() method"""

import sys
import os

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RYZEN-LLM", "scripts"))

print("=" * 60)
print("DEBUG: Full run_training() Test")
print("=" * 60)

try:
    print("\nStep 1: Creating TrainingLoop...")
    from training_loop import TrainingLoop
    loop = TrainingLoop(
        config_path="training_configuration.yaml",
        enable_optimization=False,
        device="cpu"
    )
    print("  ✓ TrainingLoop created")
    
    print("\nStep 2: Running run_training(num_epochs=1)...")
    print("  [This will now call the full training method]")
    print("-" * 60)
    
    loop.run_training(num_epochs=1)
    
    print("-" * 60)
    print("  ✓ Training completed successfully!")
    
    print("\n" + "=" * 60)
    print("SUCCESS - run_training() works!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR during training:")
    print(f"  Type: {type(e).__name__}")
    print(f"  Message: {str(e)}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
