#!/usr/bin/env python3
"""
Test suite for high-level quantization Python API.

Tests the QuantizationConfig, QuantizationEngine, and BatchQuantizer classes.
"""

import sys
import os
import numpy as np
from pathlib import Path
import importlib.util

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load quantization module from file
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

quantization_path = project_root / "src" / "core" / "quantization.py"
spec = importlib.util.spec_from_file_location("quantization", quantization_path)
quantization_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantization_module)

# Import classes
QuantizationConfig = quantization_module.QuantizationConfig
QuantizationEngine = quantization_module.QuantizationEngine
BatchQuantizer = quantization_module.BatchQuantizer
create_default_config = quantization_module.create_default_config
create_aggressive_config = quantization_module.create_aggressive_config
estimate_model_size = quantization_module.estimate_model_size

# ============================================================================
# Test Classes
# ============================================================================

class TestQuantizationConfig:
    """Test QuantizationConfig class."""
    
    def test_default_creation(self):
        """Test default config creation."""
        config = QuantizationConfig()
        assert config.weight_group_size == 128
        assert config.per_group_scaling == True
        assert config.activation_clip_value == 6.0
        print("[OK] Default config created")
    
    def test_custom_creation(self):
        """Test custom config creation."""
        config = QuantizationConfig(
            weight_group_size=64,
            per_group_scaling=False,
            activation_clip_value=4.0,
        )
        assert config.weight_group_size == 64
        assert config.per_group_scaling == False
        assert config.activation_clip_value == 4.0
        print("[OK] Custom config created")
    
    def test_cpp_conversion(self):
        """Test conversion to C++ config."""
        config = QuantizationConfig(weight_group_size=64)
        cpp_config = config.to_cpp_config()
        assert cpp_config is not None
        assert cpp_config.weight_group_size == 64
        print("[OK] Config conversion to C++ works")
    
    def test_repr(self):
        """Test string representation."""
        config = QuantizationConfig(weight_group_size=128)
        repr_str = repr(config)
        assert 'QuantizationConfig' in repr_str
        assert '128' in repr_str
        print(f"[OK] Config repr: {repr_str}")


class TestQuantizationEngine:
    """Test QuantizationEngine class."""
    
    def test_engine_creation(self):
        """Test engine creation."""
        engine = QuantizationEngine()
        assert engine is not None
        print("[OK] Engine created")
    
    def test_engine_with_config(self):
        """Test engine with custom config."""
        config = QuantizationConfig(weight_group_size=64)
        engine = QuantizationEngine(config)
        assert engine.config.weight_group_size == 64
        print("[OK] Engine accepts custom config")
    
    def test_quantize_weights_2d(self):
        """Test weight quantization for 2D matrix."""
        weights = np.random.randn(32, 64).astype(np.float32)
        engine = QuantizationEngine()
        
        ternary = engine.quantize_weights(weights)
        
        assert ternary.rows == 32
        assert ternary.cols == 64
        print("[OK] 2D weight quantization")
    
    def test_quantize_weights_1d(self):
        """Test weight quantization for 1D vector."""
        weights = np.random.randn(128).astype(np.float32)
        engine = QuantizationEngine()
        
        ternary = engine.quantize_weights(weights)
        
        # Should be reshaped to (128, 1)
        assert ternary.rows == 128
        assert ternary.cols == 1
        print("[OK] 1D weight quantization (reshaped)")
    
    def test_quantize_weights_invalid_dtype(self):
        """Test error handling for non-FP32 weights."""
        weights = np.random.randint(0, 10, (32, 64)).astype(np.int32)
        engine = QuantizationEngine()
        
        try:
            engine.quantize_weights(weights)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "FP32" in str(e)
            print("[OK] Invalid dtype raises ValueError")
    
    def test_quantize_weights_empty(self):
        """Test error handling for empty weights."""
        weights = np.array([], dtype=np.float32)
        engine = QuantizationEngine()
        
        try:
            engine.quantize_weights(weights)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "empty" in str(e).lower()
            print("[OK] Empty array raises ValueError")
    
    def test_dequantize_weights(self):
        """Test weight dequantization."""
        weights = np.random.randn(16, 32).astype(np.float32)
        engine = QuantizationEngine()
        
        ternary = engine.quantize_weights(weights)
        recovered = engine.dequantize_weights(ternary)
        
        assert recovered.shape == weights.shape
        assert np.all(np.isfinite(recovered))
        print("[OK] Weight dequantization")
    
    def test_quantize_activations(self):
        """Test activation quantization."""
        acts = np.random.randn(256).astype(np.float32)
        engine = QuantizationEngine()
        
        quant_acts = engine.quantize_activations(acts)
        
        assert quant_acts.size() == len(acts)
        assert quant_acts.scale > 0
        print(f"[OK] Activation quantization (scale={quant_acts.scale:.4f})")
    
    def test_dequantize_activations(self):
        """Test activation dequantization."""
        acts = np.random.randn(256).astype(np.float32)
        engine = QuantizationEngine()
        
        quant_acts = engine.quantize_activations(acts)
        recovered = engine.dequantize_activations(quant_acts)
        
        assert recovered.shape == acts.shape
        assert np.all(np.isfinite(recovered))
        print("[OK] Activation dequantization")
    
    def test_compute_error(self):
        """Test error computation."""
        original = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        quantized = np.array([1.05, 2.0, 2.95, 4.05], dtype=np.float32)
        
        engine = QuantizationEngine()
        error = engine.compute_error(original, quantized)
        
        assert error > 0
        assert np.isfinite(error)
        print(f"[OK] Error computation: {error:.6f}")
    
    def test_compute_error_zero(self):
        """Test zero error for identical arrays."""
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        
        engine = QuantizationEngine()
        error = engine.compute_error(values, values)
        
        assert error == 0.0
        print("[OK] Zero error for identical arrays")
    
    def test_compute_error_shape_mismatch(self):
        """Test error handling for shape mismatch."""
        original = np.array([1.0, 2.0], dtype=np.float32)
        quantized = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        engine = QuantizationEngine()
        
        try:
            engine.compute_error(original, quantized)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "Shape" in str(e)
            print("[OK] Shape mismatch raises ValueError")
    
    def test_quantize_and_measure(self):
        """Test combined quantize and measure operation."""
        weights = np.random.randn(32, 64).astype(np.float32)
        engine = QuantizationEngine()
        
        result = engine.quantize_and_measure(weights, recover=True)
        
        assert 'ternary' in result
        assert 'recovered' in result
        assert 'error' in result
        assert 'compression' in result
        assert result['recovered'].shape == weights.shape
        assert result['compression'] > 1.0  # Should have compression
        print(f"[OK] Quantize and measure: error={result['error']:.6f}, "
              f"compression={result['compression']:.2f}x")
    
    def test_weight_caching(self):
        """Test weight caching functionality."""
        weights = np.random.randn(32, 64).astype(np.float32)
        engine = QuantizationEngine()
        
        # First quantization (not cached)
        ternary1 = engine.quantize_weights(weights, name="test", cache=True)
        
        # Second quantization (should use cache)
        ternary2 = engine.quantize_weights(weights, name="test", cache=True)
        
        # Should be the same object (cached)
        assert ternary1 is ternary2
        
        # Check cache stats
        stats = engine.get_cache_stats()
        assert stats['cached_weights'] == 1
        
        print("[OK] Weight caching works")
    
    def test_cache_clearing(self):
        """Test cache clearing."""
        weights = np.random.randn(32, 64).astype(np.float32)
        engine = QuantizationEngine()
        
        engine.quantize_weights(weights, name="test", cache=True)
        assert engine.get_cache_stats()['cached_weights'] == 1
        
        engine.clear_cache(weights=True, activations=False)
        assert engine.get_cache_stats()['cached_weights'] == 0
        
        print("[OK] Cache clearing works")


class TestBatchQuantizer:
    """Test BatchQuantizer class."""
    
    def test_batch_quantizer_creation(self):
        """Test batch quantizer creation."""
        quantizer = BatchQuantizer()
        assert quantizer is not None
        print("[OK] Batch quantizer created")
    
    def test_quantize_dict(self):
        """Test quantizing dictionary of weights."""
        weights_dict = {
            'layer1': np.random.randn(32, 64).astype(np.float32),
            'layer2': np.random.randn(64, 128).astype(np.float32),
            'layer3': np.random.randn(128, 64).astype(np.float32),
        }
        
        quantizer = BatchQuantizer()
        results = quantizer.quantize_dict(weights_dict, measure_error=False)
        
        assert len(results) == 3
        assert 'layer1' in results
        assert 'layer2' in results
        assert 'layer3' in results
        print("[OK] Dictionary quantization")
    
    def test_quantize_dict_with_error(self):
        """Test quantizing dictionary with error measurement."""
        weights_dict = {
            'layer1': np.random.randn(32, 64).astype(np.float32),
            'layer2': np.random.randn(64, 128).astype(np.float32),
        }
        
        quantizer = BatchQuantizer()
        results = quantizer.quantize_dict(weights_dict, measure_error=True)
        
        assert len(results) == 4  # 2 weights + 2 errors
        assert 'layer1_error' in results
        assert 'layer2_error' in results
        print("[OK] Dictionary quantization with error measurement")
    
    def test_quantize_layer_weights(self):
        """Test quantizing transformer layer weights."""
        layer_dict = {
            'self_attn_q_proj': np.random.randn(768, 768).astype(np.float32),
            'self_attn_k_proj': np.random.randn(768, 768).astype(np.float32),
            'self_attn_v_proj': np.random.randn(768, 768).astype(np.float32),
            'mlp_fc1': np.random.randn(768, 3072).astype(np.float32),
            'mlp_fc2': np.random.randn(3072, 768).astype(np.float32),
        }
        
        quantizer = BatchQuantizer()
        results = quantizer.quantize_layer_weights(layer_dict)
        
        assert len(results) == 5
        print("[OK] Layer weight quantization")


class TestConfigFactory:
    """Test configuration factory functions."""
    
    def test_default_config(self):
        """Test default config factory."""
        config = create_default_config()
        assert config.weight_group_size == 128
        assert config.per_group_scaling == True
        print("[OK] Default config factory")
    
    def test_aggressive_config(self):
        """Test aggressive config factory."""
        config = create_aggressive_config()
        assert config.weight_group_size == 256  # Larger for more compression
        assert config.activation_clip_value == 4.0  # Tighter clipping
        print("[OK] Aggressive config factory")


class TestModelSizeEstimation:
    """Test model size estimation utilities."""
    
    def test_estimate_model_size(self):
        """Test model size estimation."""
        weights_shapes = {
            'attn': (768, 768),
            'mlp_up': (768, 3072),
            'mlp_down': (3072, 768),
        }
        
        sizes = estimate_model_size(weights_shapes)
        
        assert 'original_mb' in sizes
        assert 'ternary_mb' in sizes
        assert 'compression_ratio' in sizes
        assert sizes['compression_ratio'] > 1.0
        
        print(f"[OK] Size estimation: "
              f"{sizes['original_mb']:.2f}MB -> "
              f"{sizes['ternary_mb']:.2f}MB "
              f"({sizes['compression_ratio']:.1f}x compression)")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all test suites."""
    
    print("\n" + "="*80)
    print("Quantization Python API Test Suite")
    print("="*80)
    
    test_classes = [
        TestQuantizationConfig,
        TestQuantizationEngine,
        TestBatchQuantizer,
        TestConfigFactory,
        TestModelSizeEstimation,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 80)
        
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, test_method)
                method()
                passed_tests += 1
            except AssertionError as e:
                print(f"[FAIL] {test_method} FAILED: {e}")
                failed_tests += 1
            except Exception as e:
                print(f"[FAIL] {test_method} ERROR: {e}")
                failed_tests += 1
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"Total Tests:   {total_tests}")
    print(f"Passed:        {passed_tests}")
    print(f"Failed:        {failed_tests}")
    print(f"Success Rate:  {100*passed_tests/total_tests:.1f}%")
    print("="*80 + "\n")
    
    return failed_tests == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
