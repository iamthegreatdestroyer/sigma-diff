#!/usr/bin/env python
"""Quick test to verify ScaledTransformerModel returns tensor (not tuple)"""

import torch
import sys
sys.path.insert(0, 'S:\\Ryot\\RYZEN-LLM')

from models.scaled_transformer import ScaledTransformerModel

print("=" * 70)
print("TEST: ScaledTransformerModel Forward Pass Return Type")
print("=" * 70)

# Create model
model = ScaledTransformerModel(
    vocab_size=2048,
    embedding_dim=256,
    num_heads=4,
    num_layers=2,
    ff_dim=512,
    max_seq_len=128,
    num_classes=2
)

# Create synthetic input
batch_x = torch.randint(0, 2048, (16, 128))

print(f"\n✅ Model created")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Forward pass
print(f"\n🔄 Running forward pass...")
output = model(batch_x)

print(f"\n✅ Forward pass successful!")
print(f"   Output type: {type(output)}")
print(f"   Output shape: {output.shape if isinstance(output, torch.Tensor) else 'N/A'}")

# Verify it's a tensor
if isinstance(output, torch.Tensor):
    print(f"\n✅ SUCCESS: Model returns TENSOR (not tuple)")
    print(f"   Can use with CrossEntropyLoss: YES")
    
    # Test loss computation
    batch_y = torch.randint(0, 2, (16,))
    loss_fn = torch.nn.CrossEntropyLoss()
    try:
        loss = loss_fn(output, batch_y)
        print(f"   Loss computation: ✅ SUCCESS")
        print(f"   Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"   Loss computation: ❌ FAILED - {e}")
else:
    print(f"\n❌ FAILED: Model returns {type(output)} (expected Tensor)")
    print(f"   This will cause CrossEntropyLoss to fail")

print("\n" + "=" * 70)
