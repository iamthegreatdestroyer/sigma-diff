"""
Weight Loader Integration Tests
[PHASE2-004-TEST] - Test WeightLoader with QuantizationEngine

Tests cover:
- Weight format detection
- SafeTensors loading (with mock)
- PyTorch loading (with mock)
- Quantization integration
- Statistics tracking
- Error handling
"""

import sys
import os
from pathlib import Path
import json
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import importlib.util

# Setup path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))  # For C++ bindings
sys.path.insert(0, str(project_root / "src" / "core"))

# Load quantization module
quantization_spec = importlib.util.spec_from_file_location(
    "quantization",
    project_root / "src" / "core" / "quantization.py"
)
quantization = importlib.util.module_from_spec(quantization_spec)
sys.modules['quantization'] = quantization
quantization_spec.loader.exec_module(quantization)

# Load weight_loader module
weight_loader_spec = importlib.util.spec_from_file_location(
    "weight_loader",
    project_root / "src" / "core" / "weight_loader.py"
)
weight_loader = importlib.util.module_from_spec(weight_loader_spec)
sys.modules['weight_loader'] = weight_loader
weight_loader_spec.loader.exec_module(weight_loader)


class TestWeightLoaderConfig:
    """Test WeightLoaderConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = weight_loader.WeightLoaderConfig()
        
        assert config.quantize == True
        assert config.auto_aggressive == False
        assert config.device == "cpu"
        assert config.validate_shapes == True
        assert config.compute_error == True
        print("[OK] Default config created")
    
    def test_custom_config(self):
        """Test custom configuration."""
        quant_config = quantization.create_aggressive_config()
        config = weight_loader.WeightLoaderConfig(
            quantize=True,
            quantization_config=quant_config,
            device="cpu",
        )
        
        assert config.quantize == True
        assert config.quantization_config is not None
        print("[OK] Custom config created")
    
    def test_auto_aggressive_config(self):
        """Test auto-aggressive configuration."""
        config = weight_loader.WeightLoaderConfig(
            quantize=True,
            auto_aggressive=True,
        )
        
        assert config.quantization_config is not None
        assert config.quantization_config.weight_group_size == 256  # Aggressive setting
        print("[OK] Auto-aggressive config initialized")


class TestWeightLoaderDetection:
    """Test weight format detection."""
    
    def test_safetensors_detection(self):
        """Test SafeTensors format detection."""
        loader = weight_loader.WeightLoader()
        fmt = loader.detect_format("model.safetensors")
        
        assert fmt == weight_loader.WeightFormat.SAFETENSORS
        print("[OK] SafeTensors format detected")
    
    def test_pytorch_detection(self):
        """Test PyTorch format detection."""
        loader = weight_loader.WeightLoader()
        fmt = loader.detect_format("model.pth")
        
        assert fmt == weight_loader.WeightFormat.PYTORCH
        print("[OK] PyTorch format detected")
    
    def test_gguf_detection(self):
        """Test GGUF format detection."""
        loader = weight_loader.WeightLoader()
        fmt = loader.detect_format("model.gguf")
        
        assert fmt == weight_loader.WeightFormat.GGUF
        print("[OK] GGUF format detected")
    
    def test_unknown_format(self):
        """Test unknown format detection."""
        loader = weight_loader.WeightLoader()
        fmt = loader.detect_format("model.unknown")
        
        assert fmt == weight_loader.WeightFormat.CUSTOM
        print("[OK] Unknown format detected as CUSTOM")


class TestWeightLoaderQuantization:
    """Test weight quantization during loading."""
    
    def test_quantize_weights_dict(self):
        """Test quantization of weight dictionary."""
        loader = weight_loader.WeightLoader(
            weight_loader.WeightLoaderConfig(quantize=True)
        )
        
        # Create test weights
        weights = {
            'layer1': np.random.randn(64, 128).astype(np.float32),
            'layer2': np.random.randn(128, 256).astype(np.float32),
        }
        
        # Quantize
        quantized = loader._quantize_weights(weights)
        
        assert len(quantized) == 2
        assert 'layer1' in quantized
        assert 'layer2' in quantized
        
        # Check compression ratio calculated
        assert loader.stats.compression_ratio > 1.0
        print(f"[OK] Quantized {len(quantized)} layers, compression {loader.stats.compression_ratio:.2f}x")
    
    def test_quantization_disabled(self):
        """Test loading without quantization."""
        loader = weight_loader.WeightLoader(
            weight_loader.WeightLoaderConfig(quantize=False)
        )
        
        weights = {
            'layer1': np.random.randn(64, 128).astype(np.float32),
        }
        
        # Load without quantization
        quantized = loader._quantize_weights(weights)
        
        # Should return original weights
        assert np.array_equal(quantized['layer1'], weights['layer1'])
        print("[OK] Quantization disabled - original weights returned")
    
    def test_compression_stats(self):
        """Test compression statistics calculation."""
        loader = weight_loader.WeightLoader(
            weight_loader.WeightLoaderConfig(quantize=True, compute_error=True)
        )
        
        weights = {
            'embedding': np.random.randn(1000, 256).astype(np.float32),
            'mlp_out': np.random.randn(768, 3072).astype(np.float32),
        }
        
        quantized = loader._quantize_weights(weights)
        stats = loader.get_stats()
        
        assert stats.total_parameters > 0
        assert stats.original_size_mb > 0
        assert stats.quantized_size_mb > 0
        assert stats.compression_ratio > 1.0
        assert stats.mean_layer_error >= 0
        
        print(f"[OK] Compression stats calculated:")
        print(f"      Original: {stats.original_size_mb:.2f}MB")
        print(f"      Quantized: {stats.quantized_size_mb:.2f}MB")
        print(f"      Ratio: {stats.compression_ratio:.2f}x")
        print(f"      Error: {stats.mean_layer_error:.6f}")


class TestWeightLoaderAPI:
    """Test WeightLoader public API."""
    
    def test_loader_creation(self):
        """Test creating WeightLoader instance."""
        loader = weight_loader.WeightLoader()
        
        assert loader.config is not None
        assert loader.quantizer is not None
        assert loader.stats is not None
        print("[OK] WeightLoader instance created")
    
    def test_safetensors_not_available(self):
        """Test error when SafeTensors not available."""
        loader = weight_loader.WeightLoader()
        
        # Mock safetensors not available
        with patch.dict('sys.modules', {'safetensors': None}):
            weight_loader.SAFETENSORS_AVAILABLE = False
            
            try:
                loader.load_safetensors("dummy.safetensors")
                assert False, "Should raise ImportError"
            except ImportError as e:
                assert "safetensors" in str(e).lower()
                print("[OK] ImportError raised when safetensors not available")
            
            # Restore
            weight_loader.SAFETENSORS_AVAILABLE = SAFETENSORS_AVAILABLE
    
    def test_file_not_found(self):
        """Test error when file not found."""
        loader = weight_loader.WeightLoader()
        
        # Try loading non-existent PyTorch file
        try:
            import torch
            loader.load_pytorch("/nonexistent/path/model.pth")
            assert False, "Should raise FileNotFoundError"
        except ImportError:
            # torch not installed, skip test
            print("[SKIP] test_file_not_found - torch not installed")
        except FileNotFoundError:
            print("[OK] FileNotFoundError raised for missing file")
    
    def test_clear_cache(self):
        """Test clearing loader cache."""
        loader = weight_loader.WeightLoader()
        
        # Add some dummy weights
        loader.loaded_weights['test'] = np.array([1, 2, 3])
        assert len(loader.loaded_weights) > 0
        
        # Clear cache
        loader.clear_cache()
        assert len(loader.loaded_weights) == 0
        print("[OK] Cache cleared successfully")


class TestCompressionStats:
    """Test CompressionStats data structure."""
    
    def test_stats_creation(self):
        """Test creating compression stats."""
        stats = weight_loader.CompressionStats(
            total_parameters=1000000,
            original_size_mb=10.0,
            quantized_size_mb=2.5,
            compression_ratio=4.0,
        )
        
        assert stats.total_parameters == 1000000
        assert stats.original_size_mb == 10.0
        assert stats.quantized_size_mb == 2.5
        assert stats.compression_ratio == 4.0
        print("[OK] CompressionStats created")
    
    def test_stats_repr(self):
        """Test stats string representation."""
        stats = weight_loader.CompressionStats(
            total_parameters=1000000,
            original_size_mb=20.0,
            quantized_size_mb=5.0,
            compression_ratio=4.0,
            mean_layer_error=0.001,
        )
        
        repr_str = repr(stats)
        assert "CompressionStats" in repr_str
        assert "compression=4.00x" in repr_str
        print("[OK] Stats repr formatted correctly")
    
    def test_layer_stats_tracking(self):
        """Test per-layer statistics tracking."""
        stats = weight_loader.CompressionStats()
        
        stats.layer_stats['layer1'] = {
            'num_params': 100000,
            'original_mb': 0.4,
            'quantized_mb': 0.1,
            'compression_ratio': 4.0,
            'error': 0.001,
        }
        
        assert 'layer1' in stats.layer_stats
        assert stats.layer_stats['layer1']['compression_ratio'] == 4.0
        print("[OK] Layer statistics tracked")


class TestConvenienceFunctions:
    """Test convenience loading functions."""
    
    def test_load_weights_function(self):
        """Test load_weights convenience function."""
        # Skip test if torch not available
        try:
            import torch
        except ImportError:
            print("[SKIP] test_load_weights_function - torch not installed")
            return
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = f.name
        
        try:
            # Mock PyTorch loading
            import torch
            original_load = torch.load
            torch.load = Mock(return_value={
                'weight1': np.random.randn(128, 256).astype(np.float32),
            })
            
            try:
                weights, stats = weight_loader.load_weights(
                    temp_path,
                    quantize=True,
                    aggressive=False,
                )
                
                assert isinstance(weights, dict)
                assert isinstance(stats, weight_loader.CompressionStats)
                print("[OK] load_weights convenience function works")
            finally:
                torch.load = original_load
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_and_quantize_function(self):
        """Test load_and_quantize convenience function."""
        # Skip test if torch not available
        try:
            import torch
        except ImportError:
            print("[SKIP] test_load_and_quantize_function - torch not installed")
            return
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = f.name
        
        try:
            # Mock PyTorch loading
            import torch
            original_load = torch.load
            torch.load = Mock(return_value={
                'weight1': np.random.randn(64, 128).astype(np.float32),
            })
            
            try:
                config = quantization.create_default_config()
                weights, stats = weight_loader.load_and_quantize(
                    temp_path,
                    config=config,
                )
                
                assert isinstance(weights, dict)
                assert isinstance(stats, weight_loader.CompressionStats)
                print("[OK] load_and_quantize convenience function works")
            finally:
                torch.load = original_load
        finally:
            Path(temp_path).unlink(missing_ok=True)


# Preserve original SAFETENSORS_AVAILABLE for restoration
SAFETENSORS_AVAILABLE = weight_loader.SAFETENSORS_AVAILABLE


def run_all_tests():
    """Run all weight loader tests."""
    
    test_classes = [
        TestWeightLoaderConfig,
        TestWeightLoaderDetection,
        TestWeightLoaderQuantization,
        TestWeightLoaderAPI,
        TestCompressionStats,
        TestConvenienceFunctions,
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("=" * 60)
        
        test_instance = test_class()
        test_methods = [
            m for m in dir(test_instance)
            if m.startswith('test_') and callable(getattr(test_instance, m))
        ]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_instance, test_method)()
                passed_tests += 1
            except Exception as e:
                print(f"[FAIL] {test_method}: {e}")
    
    print("\n" + "=" * 60)
    print(f"\nTest Summary:")
    print(f"  Total Tests:  {total_tests}")
    print(f"  Passed:       {passed_tests}")
    print(f"  Failed:       {total_tests - passed_tests}")
    print(f"  Success Rate: {100.0 * passed_tests / total_tests:.1f}%")
    print()
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
