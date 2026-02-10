"""
Core Test Suite for Sprint 4.2: Model Optimization & Quantization

This module provides essential tests covering core optimization components
with the correct API matching the implementation.

Sprint: 4.2 - Model Optimization & Quantization
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Import optimization components
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimization import (
    # Quantization
    QuantizationStrategy,
    QuantizationConfig,
    QuantizationMode,
    DynamicQuantizer,
    StaticQuantizer,
    MixedPrecisionQuantizer,
    QuantizationResult,
    create_quantizer,
    
    # Compression
    CompressionStrategy,
    CompressionConfig,
    WeightSharingCompressor,
    LowRankCompressor,
    DistillationCompressor,
    CompressionResult,
    LowRankLinear,
    WeightSharedLinear,
    create_compressor,
    
    # Pruning
    PruningStrategy,
    PruningSchedule,
    PruningConfig,
    UnstructuredPruner,
    StructuredPruner,
    GradualPruner,
    PruningResult,
    create_pruner,
    
    # Calibration
    CalibrationStrategy,
    CalibrationMode,
    CalibrationPhase,
    CalibrationConfig,
    CalibrationDataset,
    PerTensorCalibrator,
    PerChannelCalibrator,
    AdaptiveCalibrator,
    CalibrationResult,
    create_calibrator,
    compute_min_max_range,
    compute_percentile_range,
    compute_scale_zero_point,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)
    
    return SimpleModel()


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randn(4, 64)


@pytest.fixture
def calibration_data():
    """Create calibration data."""
    return [torch.randn(8, 64) for _ in range(10)]


# =============================================================================
# Quantization Tests
# =============================================================================

class TestQuantizationConfig:
    """Tests for QuantizationConfig dataclass."""
    
    def test_default_config(self):
        """Test default quantization configuration."""
        config = QuantizationConfig()
        assert config.weight_mode == QuantizationMode.INT8
        assert config.activation_mode == QuantizationMode.INT8
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = QuantizationConfig(
            strategy=QuantizationStrategy.DYNAMIC,
            weight_mode=QuantizationMode.INT4,
        )
        assert config.strategy == QuantizationStrategy.DYNAMIC
        assert config.weight_mode == QuantizationMode.INT4


class TestDynamicQuantizer:
    """Tests for DynamicQuantizer - correct API."""
    
    def test_initialization(self):
        """Test dynamic quantizer initialization."""
        config = QuantizationConfig(strategy=QuantizationStrategy.DYNAMIC)
        quantizer = DynamicQuantizer(config)
        assert quantizer.config == config
    
    def test_quantize_model(self, simple_model, sample_input):
        """Test dynamic quantization."""
        config = QuantizationConfig(strategy=QuantizationStrategy.DYNAMIC)
        quantizer = DynamicQuantizer(config)
        result = quantizer.quantize(simple_model)  # Pass model to quantize()
        
        assert isinstance(result, QuantizationResult)
        assert result.metrics is not None
    
    def test_forward_pass(self, simple_model, sample_input):
        """Test quantized model can do forward pass."""
        config = QuantizationConfig(strategy=QuantizationStrategy.DYNAMIC)
        quantizer = DynamicQuantizer(config)
        result = quantizer.quantize(simple_model)
        
        # Forward pass should work
        output = result.model(sample_input)
        assert output.shape == (4, 10)


class TestStaticQuantizer:
    """Tests for StaticQuantizer - correct API."""
    
    def test_initialization(self):
        """Test static quantizer initialization."""
        config = QuantizationConfig(strategy=QuantizationStrategy.STATIC)
        quantizer = StaticQuantizer(config)
        assert quantizer.config == config
    
    def test_quantize_model(self, simple_model, calibration_data):
        """Test static quantization with calibration."""
        config = QuantizationConfig(strategy=QuantizationStrategy.STATIC)
        quantizer = StaticQuantizer(config)
        
        # Calibrate then quantize
        quantizer.calibrate(simple_model, calibration_data)
        result = quantizer.quantize(simple_model)
        
        assert isinstance(result, QuantizationResult)


class TestQuantizerFactory:
    """Tests for quantizer factory function."""
    
    def test_create_dynamic_quantizer(self):
        """Test creating dynamic quantizer via factory."""
        config = QuantizationConfig(strategy=QuantizationStrategy.DYNAMIC)
        quantizer = create_quantizer(config)
        assert isinstance(quantizer, DynamicQuantizer)
    
    def test_create_static_quantizer(self):
        """Test creating static quantizer via factory."""
        # Factory takes strategy enum, not config
        quantizer = create_quantizer(strategy=QuantizationStrategy.STATIC)
        assert isinstance(quantizer, StaticQuantizer)


# =============================================================================
# Compression Tests
# =============================================================================

class TestCompressionConfig:
    """Tests for CompressionConfig dataclass."""
    
    def test_default_config(self):
        """Test default compression configuration."""
        config = CompressionConfig()
        assert isinstance(config.strategy, CompressionStrategy)
    
    def test_low_rank_config(self):
        """Test low-rank compression configuration."""
        config = CompressionConfig(
            strategy=CompressionStrategy.LOW_RANK,
            rank_ratio=0.5
        )
        assert config.strategy == CompressionStrategy.LOW_RANK
        assert config.rank_ratio == 0.5


class TestLowRankLinear:
    """Tests for LowRankLinear layer."""
    
    def test_initialization(self):
        """Test low-rank linear layer initialization."""
        layer = LowRankLinear(64, 128, rank=32)
        assert layer.in_features == 64
        assert layer.out_features == 128
        assert layer.rank == 32
    
    def test_forward_pass(self):
        """Test forward pass through low-rank layer."""
        layer = LowRankLinear(64, 128, rank=32)
        x = torch.randn(4, 64)
        output = layer(x)
        assert output.shape == (4, 128)
    
    def test_parameter_reduction(self):
        """Test that parameters are reduced."""
        full_layer = nn.Linear(64, 128)
        low_rank = LowRankLinear(64, 128, rank=16)
        
        full_params = sum(p.numel() for p in full_layer.parameters())
        lr_params = sum(p.numel() for p in low_rank.parameters())
        
        assert lr_params < full_params


class TestWeightSharedLinear:
    """Tests for WeightSharedLinear layer."""
    
    def test_initialization(self):
        """Test weight shared linear layer initialization."""
        layer = WeightSharedLinear(64, 128, n_clusters=16)
        assert layer.in_features == 64
        assert layer.out_features == 128
    
    def test_forward_pass(self):
        """Test forward pass through weight shared layer."""
        layer = WeightSharedLinear(64, 128, n_clusters=16)
        x = torch.randn(4, 64)
        output = layer(x)
        assert output.shape == (4, 128)


class TestWeightSharingCompressor:
    """Tests for WeightSharingCompressor - correct API."""
    
    def test_initialization(self):
        """Test weight sharing compressor initialization."""
        config = CompressionConfig(strategy=CompressionStrategy.WEIGHT_SHARING)
        compressor = WeightSharingCompressor(config)
        assert compressor.config == config
    
    def test_compression(self, simple_model, sample_input):
        """Test weight sharing compression."""
        config = CompressionConfig(
            strategy=CompressionStrategy.WEIGHT_SHARING,
            num_clusters=16
        )
        compressor = WeightSharingCompressor(config)
        result = compressor.compress(simple_model)
        
        assert isinstance(result, CompressionResult)


class TestLowRankCompressor:
    """Tests for LowRankCompressor - correct API."""
    
    def test_initialization(self):
        """Test low-rank compressor initialization."""
        config = CompressionConfig(
            strategy=CompressionStrategy.LOW_RANK,
            rank_ratio=0.5
        )
        compressor = LowRankCompressor(config)
        assert compressor.config == config
    
    def test_svd_decomposition(self, simple_model, sample_input):
        """Test SVD-based compression."""
        config = CompressionConfig(
            strategy=CompressionStrategy.LOW_RANK,
            rank_ratio=0.5
        )
        compressor = LowRankCompressor(config)
        result = compressor.compress(simple_model)
        
        assert isinstance(result, CompressionResult)
    
    def test_forward_after_compression(self, simple_model, sample_input):
        """Test compressed model can do forward pass."""
        config = CompressionConfig(
            strategy=CompressionStrategy.LOW_RANK,
            rank_ratio=0.5
        )
        compressor = LowRankCompressor(config)
        result = compressor.compress(simple_model)
        
        output = result.model(sample_input)
        assert output.shape == (4, 10)


class TestCompressorFactory:
    """Tests for compressor factory function."""
    
    def test_create_weight_sharing(self):
        """Test creating weight sharing compressor."""
        # Factory signature: create_compressor(strategy, config=None, **kwargs)
        compressor = create_compressor(CompressionStrategy.WEIGHT_SHARING)
        assert isinstance(compressor, WeightSharingCompressor)
    
    def test_create_low_rank(self):
        """Test creating low-rank compressor."""
        # Factory signature: create_compressor(strategy, config=None, **kwargs)
        compressor = create_compressor(CompressionStrategy.LOW_RANK)
        assert isinstance(compressor, LowRankCompressor)


# =============================================================================
# Pruning Tests
# =============================================================================

class TestPruningConfig:
    """Tests for PruningConfig dataclass."""
    
    def test_default_config(self):
        """Test default pruning configuration."""
        config = PruningConfig()
        assert config.target_sparsity == 0.5
        assert config.strategy == PruningStrategy.MAGNITUDE
    
    def test_gradual_pruning_config(self):
        """Test gradual pruning configuration."""
        config = PruningConfig(
            schedule=PruningSchedule.LINEAR,
            num_pruning_steps=10
        )
        assert config.schedule == PruningSchedule.LINEAR


class TestUnstructuredPruner:
    """Tests for UnstructuredPruner - correct API."""
    
    def test_initialization(self):
        """Test unstructured pruner initialization."""
        config = PruningConfig(target_sparsity=0.5)
        pruner = UnstructuredPruner(config)
        assert pruner.config == config
    
    def test_magnitude_pruning(self, simple_model):
        """Test magnitude-based pruning."""
        config = PruningConfig(
            target_sparsity=0.3,
            strategy=PruningStrategy.MAGNITUDE
        )
        pruner = UnstructuredPruner(config)
        result = pruner.prune(simple_model)
        
        assert isinstance(result, PruningResult)
    
    def test_forward_after_pruning(self, simple_model, sample_input):
        """Test pruned model can do forward pass."""
        config = PruningConfig(target_sparsity=0.3)
        pruner = UnstructuredPruner(config)
        result = pruner.prune(simple_model)
        
        output = result.model(sample_input)
        assert output.shape == (4, 10)


class TestStructuredPruner:
    """Tests for StructuredPruner - correct API."""
    
    def test_initialization(self):
        """Test structured pruner initialization."""
        config = PruningConfig(target_sparsity=0.3)
        pruner = StructuredPruner(config)
        assert pruner.config == config
    
    def test_channel_pruning(self, simple_model):
        """Test channel pruning."""
        config = PruningConfig(target_sparsity=0.2)
        pruner = StructuredPruner(config)
        result = pruner.prune(simple_model)
        
        assert isinstance(result, PruningResult)


class TestGradualPruner:
    """Tests for GradualPruner - correct API."""
    
    def test_initialization(self):
        """Test gradual pruner initialization."""
        config = PruningConfig(
            schedule=PruningSchedule.LINEAR,
            num_pruning_steps=5
        )
        pruner = GradualPruner(config)
        assert pruner.config == config


class TestPrunerFactory:
    """Tests for pruner factory function."""
    
    def test_create_unstructured(self):
        """Test creating unstructured pruner."""
        config = PruningConfig()
        pruner = create_pruner(config)
        assert isinstance(pruner, UnstructuredPruner)


# =============================================================================
# Calibration Tests
# =============================================================================

class TestCalibrationConfig:
    """Tests for CalibrationConfig dataclass."""
    
    def test_default_config(self):
        """Test default calibration configuration."""
        config = CalibrationConfig()
        assert config.strategy == CalibrationStrategy.HISTOGRAM
    
    def test_percentile_config(self):
        """Test percentile calibration configuration."""
        config = CalibrationConfig(
            strategy=CalibrationStrategy.PERCENTILE,
            percentile=99.9
        )
        assert config.strategy == CalibrationStrategy.PERCENTILE
        assert config.percentile == 99.9


class TestCalibrationDataset:
    """Tests for CalibrationDataset."""
    
    def test_initialization(self, calibration_data):
        """Test calibration dataset initialization.
        
        Note: CalibrationDataset flattens batches into individual samples.
        With 10 batches of 8 samples each, total = 80 samples.
        """
        dataset = CalibrationDataset(calibration_data)
        # Dataset flattens batches: 10 batches * 8 samples = 80 total
        expected_samples = sum(batch.shape[0] for batch in calibration_data)
        assert len(dataset) == expected_samples
    
    def test_iteration(self, calibration_data):
        """Test iterating over dataset."""
        dataset = CalibrationDataset(calibration_data)
        samples = list(dataset)
        # Flattened: all individual samples
        expected_samples = sum(batch.shape[0] for batch in calibration_data)
        assert len(samples) == expected_samples


class TestPerTensorCalibrator:
    """Tests for PerTensorCalibrator - correct API."""
    
    def test_initialization(self):
        """Test per-tensor calibrator initialization."""
        config = CalibrationConfig(mode=CalibrationMode.PER_TENSOR)
        calibrator = PerTensorCalibrator(config)
        assert calibrator.config == config
    
    def test_calibration(self, simple_model, calibration_data):
        """Test per-tensor calibration."""
        config = CalibrationConfig(mode=CalibrationMode.PER_TENSOR)
        calibrator = PerTensorCalibrator(config)
        result = calibrator.calibrate(simple_model, calibration_data)
        
        assert isinstance(result, CalibrationResult)


class TestPerChannelCalibrator:
    """Tests for PerChannelCalibrator - correct API."""
    
    def test_initialization(self):
        """Test per-channel calibrator initialization."""
        config = CalibrationConfig(mode=CalibrationMode.PER_CHANNEL)
        calibrator = PerChannelCalibrator(config)
        assert calibrator.config == config


class TestAdaptiveCalibrator:
    """Tests for AdaptiveCalibrator - correct API."""
    
    def test_initialization(self):
        """Test adaptive calibrator initialization."""
        config = CalibrationConfig(strategy=CalibrationStrategy.ADAPTIVE)
        calibrator = AdaptiveCalibrator(config)
        assert calibrator.config == config


class TestCalibrationUtilities:
    """Tests for calibration utility functions."""
    
    def test_compute_min_max_range(self):
        """Test min-max range computation."""
        tensor = torch.randn(100, 64)
        min_val, max_val = compute_min_max_range(tensor)
        assert min_val <= max_val
    
    def test_compute_percentile_range(self):
        """Test percentile range computation."""
        tensor = torch.randn(100, 64)
        min_val, max_val = compute_percentile_range(tensor, percentile=99.0)
        assert min_val <= max_val
    
    def test_compute_scale_zero_point(self):
        """Test scale and zero-point computation."""
        min_val, max_val = -1.0, 1.0
        scale, zero_point = compute_scale_zero_point(min_val, max_val)
        assert scale > 0


class TestCalibratorFactory:
    """Tests for calibrator factory function."""
    
    def test_create_per_tensor(self):
        """Test creating per-tensor calibrator."""
        config = CalibrationConfig(mode=CalibrationMode.PER_TENSOR)
        calibrator = create_calibrator(config)
        assert isinstance(calibrator, PerTensorCalibrator)
    
    def test_create_per_channel(self):
        """Test creating per-channel calibrator."""
        config = CalibrationConfig(mode=CalibrationMode.PER_CHANNEL)
        calibrator = create_calibrator(config)
        assert isinstance(calibrator, PerChannelCalibrator)
    
    def test_create_adaptive(self):
        """Test creating adaptive calibrator."""
        config = CalibrationConfig(strategy=CalibrationStrategy.ADAPTIVE)
        calibrator = create_calibrator(config)
        assert isinstance(calibrator, AdaptiveCalibrator)


# =============================================================================
# Integration Tests
# =============================================================================

class TestOptimizationPipeline:
    """Integration tests for optimization pipeline."""
    
    def test_quantization_pipeline(self, simple_model, sample_input, calibration_data):
        """Test full quantization pipeline."""
        # Create quantizer
        config = QuantizationConfig(strategy=QuantizationStrategy.DYNAMIC)
        quantizer = DynamicQuantizer(config)
        
        # Quantize
        result = quantizer.quantize(simple_model)
        
        # Verify
        assert result.model is not None
        output = result.model(sample_input)
        assert output.shape == (4, 10)
    
    def test_compression_pipeline(self, simple_model, sample_input):
        """Test full compression pipeline."""
        # Create compressor
        config = CompressionConfig(
            strategy=CompressionStrategy.LOW_RANK,
            rank_ratio=0.5
        )
        compressor = LowRankCompressor(config)
        
        # Compress
        result = compressor.compress(simple_model)
        
        # Verify
        assert result.model is not None
        output = result.model(sample_input)
        assert output.shape == (4, 10)
    
    def test_pruning_pipeline(self, simple_model, sample_input):
        """Test full pruning pipeline."""
        # Create pruner
        config = PruningConfig(target_sparsity=0.3)
        pruner = UnstructuredPruner(config)
        
        # Prune
        result = pruner.prune(simple_model)
        
        # Verify
        assert result.model is not None
        output = result.model(sample_input)
        assert output.shape == (4, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
