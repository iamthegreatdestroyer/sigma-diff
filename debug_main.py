#!/usr/bin/env python3
"""Debug script to test TrainingLoop initialization and setup"""

import sys
import os

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RYZEN-LLM", "scripts"))

print("=" * 60)
print("DEBUG: TrainingLoop Initialization Test")
print("=" * 60)

try:
    print("\nStep 1: Importing modules...")
    from training_loop import TrainingLoop
    print("  ✓ TrainingLoop imported")
    
    print("\nStep 2: Loading configuration...")
    config_path = "training_configuration.yaml"
    if not os.path.exists(config_path):
        print(f"  ✗ Config file not found: {config_path}")
        sys.exit(1)
    print(f"  ✓ Config file exists: {config_path}")
    
    print("\nStep 3: Creating TrainingLoop instance...")
    loop = TrainingLoop(config_path=config_path)
    print("  ✓ TrainingLoop instantiated")
    
    print("\nStep 4: TrainingLoop attributes:")
    print(f"  - device: {loop.device}")
    print(f"  - model initialized: {loop.model is not None}")
    print(f"  - config loaded: {loop.config is not None}")
    
    print("\nStep 5: Calling setup_optimizations()...")
    loop.setup_optimizations()
    print("  ✓ setup_optimizations() completed")
    
    print("\nStep 6: Calling setup_model_and_data()...")
    loop.setup_model_and_data()
    print("  ✓ setup_model_and_data() completed")
    
    print("\nStep 7: Testing train_epoch() signature...")
    # Just check if method exists and can be called
    if hasattr(loop, 'train_epoch'):
        print("  ✓ train_epoch() method exists")
    else:
        print("  ✗ train_epoch() method NOT found")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED - TrainingLoop initialization working!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR at initialization:")
    print(f"  Type: {type(e).__name__}")
    print(f"  Message: {str(e)}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
