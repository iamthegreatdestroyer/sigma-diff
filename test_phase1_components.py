"""
PHASE 1 COMPONENTS - ISOLATION TEST
================================
Test each optimization component separately to identify hang point
"""

import sys
import time
import traceback
from pathlib import Path
import torch

# Add RYZEN-LLM paths
ryzen_root = Path(__file__).parent / "RYZEN-LLM"
sys.path.insert(0, str(ryzen_root / "scripts"))
sys.path.insert(0, str(ryzen_root / "models"))

def test_kernel_optimizer():
    """Test KernelOptimizer initialization."""
    print("\n" + "="*70)
    print("TEST 1: KernelOptimizer")
    print("="*70)
    
    try:
        print("Importing KernelOptimizer...")
        from kernel_optimizer import KernelOptimizer
        print("✅ Import successful")
        
        print("Initializing KernelOptimizer...")
        start = time.time()
        component = KernelOptimizer()
        elapsed = time.time() - start
        
        print(f"✅ Initialization successful ({elapsed:.2f}s)")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_semantic_compressor():
    """Test SemanticCompressor initialization."""
    print("\n" + "="*70)
    print("TEST 2: SemanticCompressor")
    print("="*70)
    
    try:
        print("Importing SemanticCompressor...")
        from semantic_compression import SemanticCompressor
        print("✅ Import successful")
        
        print("Initializing SemanticCompressor...")
        start = time.time()
        component = SemanticCompressor()
        elapsed = time.time() - start
        
        print(f"✅ Initialization successful ({elapsed:.2f}s)")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_inference_scaling_engine():
    """Test InferenceScalingEngine initialization."""
    print("\n" + "="*70)
    print("TEST 3: InferenceScalingEngine")
    print("="*70)
    
    try:
        print("Importing InferenceScalingEngine...")
        from inference_scaling import InferenceScalingEngine
        print("✅ Import successful")
        
        print("Initializing InferenceScalingEngine...")
        start = time.time()
        component = InferenceScalingEngine()
        elapsed = time.time() - start
        
        print(f"✅ Initialization successful ({elapsed:.2f}s)")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_scaled_transformer_model():
    """Test ScaledTransformerModel creation."""
    print("\n" + "="*70)
    print("TEST 4: ScaledTransformerModel Creation")
    print("="*70)
    
    try:
        print("Importing ScaledTransformerModel...")
        from scaled_transformer import ScaledTransformerModel
        print("✅ Import successful")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        print("Creating model...")
        start = time.time()
        model = ScaledTransformerModel(
            vocab_size=32000,
            embedding_dim=512,
            num_heads=8,
            num_layers=6,
            ff_dim=2048,
            max_seq_len=1024,
            dropout=0.1,
            num_classes=10
        ).to(device)
        elapsed = time.time() - start
        
        param_count = model.count_parameters()
        print(f"✅ Model creation successful ({elapsed:.2f}s)")
        print(f"   Parameters: {param_count:,}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_synthetic_training():
    """Test simple training loop."""
    print("\n" + "="*70)
    print("TEST 5: Simple Training Loop")
    print("="*70)
    
    try:
        import torch.nn as nn
        import torch.optim as optim
        from scaled_transformer import ScaledTransformerModel
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Creating small model for test...")
        model = ScaledTransformerModel(
            vocab_size=5000,
            embedding_dim=256,
            num_heads=4,
            num_layers=2,
            ff_dim=512,
            max_seq_len=32,
            dropout=0.1,
            num_classes=10
        ).to(device)
        
        print("Creating synthetic data...")
        x = torch.randint(0, 5000, (8, 32)).to(device)
        y = torch.randint(0, 10, (8,)).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print("Running 1 forward/backward pass...")
        start = time.time()
        
        model.train()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        elapsed = time.time() - start
        print(f"✅ Training loop successful ({elapsed:.2f}s)")
        print(f"   Loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PHASE 1 COMPONENTS - ISOLATION TEST SUITE")
    print("="*70)
    print("\nTesting each component separately to identify hang point...")
    
    results = {}
    
    # Test each component
    results['KernelOptimizer'] = test_kernel_optimizer()
    results['SemanticCompressor'] = test_semantic_compressor()
    results['InferenceScalingEngine'] = test_inference_scaling_engine()
    results['ScaledTransformerModel'] = test_scaled_transformer_model()
    results['TrainingLoop'] = test_synthetic_training()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {component}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Components work in isolation")
        print("\nNext step: Run full train_scaled_model_debug.py")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"❌ FAILED COMPONENTS: {', '.join(failed)}")
        print("\nThese components need investigation:")
        for comp in failed:
            print(f"  - Check {comp} implementation in RYZEN-LLM/")
    print("="*70)


if __name__ == "__main__":
    main()
