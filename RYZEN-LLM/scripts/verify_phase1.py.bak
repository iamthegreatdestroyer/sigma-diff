#!/usr/bin/env python3
"""
Phase 1 Completion Verification
===============================

Run this script to verify Phase 1 is complete.
"""

import sys
from pathlib import Path


def verify_files():
    """Verify all required files exist."""
    required_files = [
        # Tokenizer
        "src/core/tokenizer/__init__.py",
        "src/core/tokenizer/base.py",
        "src/core/tokenizer/bpe_tokenizer.py",
        
        # Model
        "src/core/model/__init__.py",
        "src/core/model/config.py",
        "src/core/model/loader.py",
        "src/core/model/quantization.py",
        "src/core/model/layers/__init__.py",
        "src/core/model/layers/ffn.py",
        "src/core/model/layers/rmsnorm.py",
        "src/core/model/layers/transformer.py",
        
        # Engine
        "src/core/engine/__init__.py",
        "src/core/engine/inference.py",
        "src/core/engine/kv_cache.py",
        "src/core/engine/attention.py",
        "src/core/engine/sampling.py",
        "src/core/engine/rope.py",
    ]
    
    missing = []
    for filepath in required_files:
        if not Path(filepath).exists():
            missing.append(filepath)
    
    if missing:
        print("❌ Missing files:")
        for f in missing:
            print(f"   - {f}")
        return False
    
    print(f"✓ All {len(required_files)} required files present")
    return True


def verify_imports():
    """Verify all imports work."""
    try:
        from src.core.tokenizer import BPETokenizer
        from src.core.model import BitNetConfig, ModelLoader, QuantizedTensor, BitNetMLP, RMSNorm
        from src.core.engine import RyotEngine, KVCache
        
        from src.api.interfaces import InferenceEngine, CacheManagerProtocol
        from src.api.types import GenerationConfig, GenerationResult
        
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def verify_protocol_compliance():
    """Verify protocol compliance."""
    try:
        from src.core.engine import RyotEngine
        from src.api.interfaces import InferenceEngine, CacheManagerProtocol
        from src.core.engine import KVCache
        
        engine = RyotEngine()
        assert isinstance(engine, InferenceEngine), "Engine should implement InferenceEngine"
        
        cache = KVCache(num_layers=32, num_heads=32, head_dim=128, max_length=4096)
        assert isinstance(cache, CacheManagerProtocol), "Cache should implement CacheManagerProtocol"
        
        print("✓ Protocol compliance verified")
        return True
    except Exception as e:
        print(f"❌ Protocol compliance error: {e}")
        return False


def verify_generation():
    """Verify generation works."""
    try:
        from src.core.engine import RyotEngine
        from src.api.types import GenerationConfig
        
        engine = RyotEngine()
        result = engine.generate(
            "Test prompt",
            GenerationConfig(max_tokens=5)
        )
        
        assert result.completion_tokens > 0
        assert len(result.generated_text) > 0
        
        print("✓ Generation verification passed")
        return True
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False


def main():
    """Run all verifications."""
    print("=" * 60)
    print("  PHASE 1: Ryot LLM - Full Completion Verification")
    print("=" * 60)
    print()
    
    results = [
        verify_files(),
        verify_imports(),
        verify_protocol_compliance(),
        verify_generation(),
    ]
    
    print()
    if all(results):
        print("=" * 60)
        print("  ✅ PHASE 1 VERIFICATION COMPLETE")
        print("=" * 60)
        print()
        print("Ready for Phase 2: ΣLANG Integration")
        return 0
    else:
        print("=" * 60)
        print("  ❌ PHASE 1 VERIFICATION FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
