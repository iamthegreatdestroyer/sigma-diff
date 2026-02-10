#!/usr/bin/env python3
"""Debug script to test actual training execution"""

import sys
import os

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RYZEN-LLM", "scripts"))

print("=" * 60)
print("DEBUG: Minimal Training Execution Test")
print("=" * 60)

try:
    print("\nStep 1: Importing TrainingLoop...")
    from training_loop import TrainingLoop
    print("  ✓ Imported")
    
    print("\nStep 2: Creating TrainingLoop with test config...")
    loop = TrainingLoop(
        config_path="training_configuration.yaml",
        enable_optimization=False,  # Disable optimization for minimal test
        device="cpu"
    )
    print(f"  ✓ Created (device={loop.device})")
    
    print("\nStep 3: Running setup_model_and_data()...")
    loop.setup_model_and_data()
    print(f"  ✓ Model: {type(loop.model).__name__}")
    print(f"  ✓ Train loader batches: {len(loop.train_loader)}")
    
    print("\nStep 4: Testing first batch iteration...")
    for batch_idx, (input_ids, labels) in enumerate(loop.train_loader):
        print(f"  Batch {batch_idx}: input_ids.shape={input_ids.shape}, labels.shape={labels.shape}")
        if batch_idx == 0:  # Just test first batch
            break
    print("  ✓ DataLoader iteration works")
    
    print("\nStep 5: Running MINIMAL training (1 epoch, 10 batches max)...")
    print("  Starting train epoch 0...")
    
    # Manually run simplified epoch to test
    loop.model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, (input_ids, labels) in enumerate(loop.train_loader):
        print(f"    Processing batch {batch_idx}...", end=' ', flush=True)
        
        # Move to device
        input_ids = input_ids.to(loop.device)
        labels = labels.to(loop.device)
        
        # Forward pass
        try:
            logits = loop.model(input_ids)
            print(f"forward OK", end=' ')
        except Exception as e:
            print(f"FORWARD ERROR: {e}")
            raise
        
        # Loss
        import torch.nn as nn
        try:
            loss = nn.functional.cross_entropy(
                logits.view(-1, loop.config['model']['vocab_size']),
                labels.view(-1)
            )
            print(f"loss={loss.item():.4f}", end=' ')
        except Exception as e:
            print(f"LOSS ERROR: {e}")
            raise
        
        # Backward
        try:
            loss.backward()
            print(f"backward OK", end=' ')
        except Exception as e:
            print(f"BACKWARD ERROR: {e}")
            raise
        
        # Optimizer step
        try:
            loop.optimizer.step()
            loop.optimizer.zero_grad()
            print(f"optim OK")
        except Exception as e:
            print(f"OPTIM ERROR: {e}")
            raise
        
        epoch_loss += loss.item()
        num_batches += 1
        
        # Stop after 10 batches for testing
        if batch_idx >= 9:
            print(f"    (Stopping after 10 batches for test)")
            break
    
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    print(f"\n  ✓ Epoch completed: {num_batches} batches, avg_loss={avg_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED - Training execution working!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR during training:")
    print(f"  Type: {type(e).__name__}")
    print(f"  Message: {str(e)}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
