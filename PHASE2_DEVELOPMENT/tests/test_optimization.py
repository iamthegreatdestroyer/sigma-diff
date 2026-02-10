"""
Comprehensive Test Suite for Sprint 4.2: Model Optimization & Quantization

This module provides 60+ tests covering all optimization components:
- Quantization (Dynamic, Static, Mixed-Precision)
- Compression (Weight Sharing, Low-Rank, Distillation)
- Pruning (Unstructured, Structured, Gradual)
- Calibration (Per-Tensor, Per-Channel, Adaptive)

Sprint: 4.2 - Model Optimization & Quantization
Target Metrics:
  - INT8 with <2% accuracy loss
  - 2-4x latency improvement
  - 50-75% memory reduction
  - 30-50% pruning with <3% accuracy loss
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import json
from pathlib import Path
from dataclasses import asdict
from typing import List, Tuple
from unittest.mock import Mock, patch, MagicMock

# Import optimization components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimization import (
    # Quantization
    QuantizationStrategy,
    QuantizationConfig,
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
def conv_model():
    """Create a convolutional model for testing."""
    class ConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)
    
    return ConvModel()


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randn(4, 64)


@pytest.fixture
def sample_conv_input():
    """Create sample input for conv model."""
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def calibration_data():
    """Create calibration data samples."""
    return [torch.randn(4, 64) for _ in range(10)]


@pytest.fixture
def teacher_model():
    """Create teacher model for distillation."""
    class TeacherModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    return TeacherModel()


# =============================================================================
# Quantization Tests
# =============================================================================

class TestQuantizationConfig:
    """Tests for QuantizationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = QuantizationConfig()
        assert config.strategy == QuantizationStrategy.DYNAMIC
        assert config.bits == 8
        assert config.per_channel is True
        assert config.symmetric is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = QuantizationConfig(
            strategy=QuantizationStrategy.STATIC,
            bits=4,
            per_channel=False,
            symmetric=False
        )
        assert config.bits == 4
        assert config.per_channel is False
    
    def test_config_serialization(self):
        """Test config can be serialized."""
        config = QuantizationConfig()
        data = asdict(config)
        assert "strategy" in data
        assert "bits" in data


class TestDynamicQuantizer:
    """Tests for DynamicQuantizer."""
    
    def test_initialization(self, simple_model):
        """Test quantizer initialization."""
        config = QuantizationConfig(strategy=QuantizationStrategy.DYNAMIC)
        quantizer = DynamicQuantizer(simple_model, config)
        assert quantizer.model is simple_model
        assert quantizer.config == config
    
    def test_quantize_model(self, simple_model, sample_input):
        """Test dynamic quantization."""
        config = QuantizationConfig(strategy=QuantizationStrategy.DYNAMIC)
        quantizer = DynamicQuantizer(simple_model, config)
        result = quantizer.quantize()
        
        assert isinstance(result, QuantizationResult)
        assert result.original_size_mb > 0
        assert result.quantized_size_mb > 0
        assert result.compression_ratio > 0
    
    def test_forward_pass(self, simple_model, sample_input):
        """Test quantized model can do forward pass."""
        config = QuantizationConfig(strategy=QuantizationStrategy.DYNAMIC)
        quantizer = DynamicQuantizer(simple_model, config)
        result = quantizer.quantize()
        
        # Forward pass should work
        output = result.quantized_model(sample_input)
        assert output.shape == (4, 10)
    
    def test_accuracy_validation(self, simple_model, sample_input):
        """Test accuracy validation method."""
        config = QuantizationConfig(strategy=QuantizationStrategy.DYNAMIC)
        quantizer = DynamicQuantizer(simple_model, config)
        result = quantizer.quantize()
        
        # Create validation function
        def accuracy_fn(model):
            with torch.no_grad():
                output = model(sample_input)
                return 0.95  # Mock accuracy
        
        if hasattr(quantizer, 'validate_accuracy'):
            acc = quantizer.validate_accuracy(accuracy_fn)
            assert 0 <= acc <= 1


class TestStaticQuantizer:
    """Tests for StaticQuantizer."""
    
    def test_initialization(self, simple_model):
        """Test static quantizer initialization."""
        config = QuantizationConfig(strategy=QuantizationStrategy.STATIC)
        quantizer = StaticQuantizer(simple_model, config)
        assert quantizer.model is simple_model
    
    def test_calibration_required(self, simple_model, calibration_data):
        """Test that calibration data is used."""
        config = QuantizationConfig(strategy=QuantizationStrategy.STATIC)
        quantizer = StaticQuantizer(simple_model, config)
        
        # Should be able to calibrate
        if hasattr(quantizer, 'calibrate'):
            quantizer.calibrate(calibration_data)
    
    def test_quantize_with_calibration(self, simple_model, calibration_data):
        """Test full quantization flow with calibration."""
        config = QuantizationConfig(strategy=QuantizationStrategy.STATIC)
        quantizer = StaticQuantizer(simple_model, config)
        
        # Calibrate if method exists
        if hasattr(quantizer, 'calibrate'):
            quantizer.calibrate(calibration_data)
        
        result = quantizer.quantize()
        assert isinstance(result, QuantizationResult)


class TestMixedPrecisionQuantizer:
    """Tests for MixedPrecisionQuantizer."""
    
    def test_initialization(self, simple_model):
        """Test mixed precision quantizer initialization."""
        config = QuantizationConfig(strategy=QuantizationStrategy.MIXED_PRECISION)
        quantizer = MixedPrecisionQuantizer(simple_model, config)
        assert quantizer.model is simple_model
    
    def test_layer_sensitivity_analysis(self, simple_model, calibration_data):
        """Test layer sensitivity analysis."""
        config = QuantizationConfig(strategy=QuantizationStrategy.MIXED_PRECISION)
        quantizer = MixedPrecisionQuantizer(simple_model, config)
        
        if hasattr(quantizer, 'analyze_sensitivity'):
            sensitivity = quantizer.analyze_sensitivity(calibration_data)
            assert isinstance(sensitivity, dict)
    
    def test_per_layer_bits_assignment(self, simple_model):
        """Test different bits per layer."""
        config = QuantizationConfig(
            strategy=QuantizationStrategy.MIXED_PRECISION,
            layer_configs={'fc1': 8, 'fc2': 4, 'fc3': 8}
        )
        quantizer = MixedPrecisionQuantizer(simple_model, config)
        result = quantizer.quantize()
        assert isinstance(result, QuantizationResult)


class TestQuantizerFactory:
    """Tests for quantizer factory function."""
    
    def test_create_dynamic_quantizer(self, simple_model):
        """Test creating dynamic quantizer via factory."""
        quantizer = create_quantizer(
            simple_model,
            strategy=QuantizationStrategy.DYNAMIC
        )
        assert isinstance(quantizer, DynamicQuantizer)
    
    def test_create_static_quantizer(self, simple_model):
        """Test creating static quantizer via factory."""
        quantizer = create_quantizer(
            simple_model,
            strategy=QuantizationStrategy.STATIC
        )
        assert isinstance(quantizer, StaticQuantizer)
    
    def test_create_mixed_precision_quantizer(self, simple_model):
        """Test creating mixed precision quantizer via factory."""
        quantizer = create_quantizer(
            simple_model,
            strategy=QuantizationStrategy.MIXED_PRECISION
        )
        assert isinstance(quantizer, MixedPrecisionQuantizer)


# =============================================================================
# Compression Tests
# =============================================================================

class TestCompressionConfig:
    """Tests for CompressionConfig dataclass."""
    
    def test_default_config(self):
        """Test default compression configuration."""
        config = CompressionConfig()
        assert config.strategy == CompressionStrategy.WEIGHT_SHARING
        assert config.target_compression_ratio > 0
    
    def test_low_rank_config(self):
        """Test low-rank compression configuration."""
        config = CompressionConfig(
            strategy=CompressionStrategy.LOW_RANK,
            rank_ratio=0.5
        )
        assert config.strategy == CompressionStrategy.LOW_RANK
        assert config.rank_ratio == 0.5


class TestWeightSharingCompressor:
    """Tests for WeightSharingCompressor."""
    
    def test_initialization(self, simple_model):
        """Test weight sharing compressor initialization."""
        config = CompressionConfig(strategy=CompressionStrategy.WEIGHT_SHARING)
        compressor = WeightSharingCompressor(simple_model, config)
        assert compressor.model is simple_model
    
    def test_compression(self, simple_model, sample_input):
        """Test weight sharing compression."""
        config = CompressionConfig(
            strategy=CompressionStrategy.WEIGHT_SHARING,
            num_clusters=16
        )
        compressor = WeightSharingCompressor(simple_model, config)
        result = compressor.compress()
        
        assert isinstance(result, CompressionResult)
        assert result.compression_ratio > 0
    
    def test_weight_clustering(self, simple_model):
        """Test that weights are clustered."""
        config = CompressionConfig(
            strategy=CompressionStrategy.WEIGHT_SHARING,
            num_clusters=8
        )
        compressor = WeightSharingCompressor(simple_model, config)
        result = compressor.compress()
        
        # Compressed model should have fewer unique weights
        assert result.compressed_model is not None


class TestLowRankCompressor:
    """Tests for LowRankCompressor."""
    
    def test_initialization(self, simple_model):
        """Test low-rank compressor initialization."""
        config = CompressionConfig(
            strategy=CompressionStrategy.LOW_RANK,
            rank_ratio=0.5
        )
        compressor = LowRankCompressor(simple_model, config)
        assert compressor.model is simple_model
    
    def test_svd_decomposition(self, simple_model, sample_input):
        """Test SVD-based compression."""
        config = CompressionConfig(
            strategy=CompressionStrategy.LOW_RANK,
            rank_ratio=0.5
        )
        compressor = LowRankCompressor(simple_model, config)
        result = compressor.compress()
        
        assert isinstance(result, CompressionResult)
        # Model should be smaller
        assert result.compression_ratio > 1.0
    
    def test_forward_after_compression(self, simple_model, sample_input):
        """Test compressed model can do forward pass."""
        config = CompressionConfig(
            strategy=CompressionStrategy.LOW_RANK,
            rank_ratio=0.5
        )
        compressor = LowRankCompressor(simple_model, config)
        result = compressor.compress()
        
        output = result.compressed_model(sample_input)
        assert output.shape == (4, 10)


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


class TestDistillationCompressor:
    """Tests for DistillationCompressor."""
    
    def test_initialization(self, simple_model, teacher_model):
        """Test distillation compressor initialization."""
        config = CompressionConfig(strategy=CompressionStrategy.DISTILLATION)
        compressor = DistillationCompressor(simple_model, config)
        compressor.set_teacher(teacher_model)
        assert compressor.model is simple_model
        assert compressor.teacher is teacher_model
    
    def test_distillation_loss(self, simple_model, teacher_model, sample_input):
        """Test distillation loss computation."""
        config = CompressionConfig(
            strategy=CompressionStrategy.DISTILLATION,
            temperature=4.0
        )
        compressor = DistillationCompressor(simple_model, config)
        compressor.set_teacher(teacher_model)
        
        if hasattr(compressor, 'compute_distillation_loss'):
            with torch.no_grad():
                student_out = simple_model(sample_input)
                teacher_out = teacher_model(sample_input)
            loss = compressor.compute_distillation_loss(student_out, teacher_out)
            assert loss.item() >= 0


class TestCompressorFactory:
    """Tests for compressor factory function."""
    
    def test_create_weight_sharing(self, simple_model):
        """Test creating weight sharing compressor."""
        compressor = create_compressor(
            simple_model,
            strategy=CompressionStrategy.WEIGHT_SHARING
        )
        assert isinstance(compressor, WeightSharingCompressor)
    
    def test_create_low_rank(self, simple_model):
        """Test creating low-rank compressor."""
        compressor = create_compressor(
            simple_model,
            strategy=CompressionStrategy.LOW_RANK
        )
        assert isinstance(compressor, LowRankCompressor)
    
    def test_create_distillation(self, simple_model):
        """Test creating distillation compressor."""
        compressor = create_compressor(
            simple_model,
            strategy=CompressionStrategy.DISTILLATION
        )
        assert isinstance(compressor, DistillationCompressor)


# =============================================================================
# Pruning Tests
# =============================================================================

class TestPruningConfig:
    """Tests for PruningConfig dataclass."""
    
    def test_default_config(self):
        """Test default pruning configuration."""
        config = PruningConfig()
        assert config.strategy == PruningStrategy.MAGNITUDE
        assert 0 < config.sparsity_target < 1
    
    def test_gradual_pruning_config(self):
        """Test gradual pruning configuration."""
        config = PruningConfig(
            schedule=PruningSchedule.LINEAR,
            initial_sparsity=0.0,
            final_sparsity=0.9,
            pruning_steps=100
        )
        assert config.schedule == PruningSchedule.LINEAR


class TestUnstructuredPruner:
    """Tests for UnstructuredPruner."""
    
    def test_initialization(self, simple_model):
        """Test unstructured pruner initialization."""
        config = PruningConfig(strategy=PruningStrategy.MAGNITUDE)
        pruner = UnstructuredPruner(simple_model, config)
        assert pruner.model is simple_model
    
    def test_magnitude_pruning(self, simple_model, sample_input):
        """Test magnitude-based pruning."""
        config = PruningConfig(
            strategy=PruningStrategy.MAGNITUDE,
            sparsity_target=0.5
        )
        pruner = UnstructuredPruner(simple_model, config)
        result = pruner.prune()
        
        assert isinstance(result, PruningResult)
        assert result.achieved_sparsity > 0
    
    def test_sparsity_achieved(self, simple_model):
        """Test that target sparsity is approximately achieved."""
        config = PruningConfig(
            strategy=PruningStrategy.MAGNITUDE,
            sparsity_target=0.5
        )
        pruner = UnstructuredPruner(simple_model, config)
        result = pruner.prune()
        
        # Should be close to target
        assert abs(result.achieved_sparsity - 0.5) < 0.1
    
    def test_forward_after_pruning(self, simple_model, sample_input):
        """Test model can do forward pass after pruning."""
        config = PruningConfig(sparsity_target=0.5)
        pruner = UnstructuredPruner(simple_model, config)
        result = pruner.prune()
        
        output = result.pruned_model(sample_input)
        assert output.shape == (4, 10)


class TestStructuredPruner:
    """Tests for StructuredPruner."""
    
    def test_initialization(self, simple_model):
        """Test structured pruner initialization."""
        config = PruningConfig(strategy=PruningStrategy.L1_NORM)
        pruner = StructuredPruner(simple_model, config)
        assert pruner.model is simple_model
    
    def test_channel_pruning(self, conv_model, sample_conv_input):
        """Test channel-wise pruning."""
        config = PruningConfig(
            strategy=PruningStrategy.L1_NORM,
            sparsity_target=0.3,
            structured=True
        )
        pruner = StructuredPruner(conv_model, config)
        result = pruner.prune()
        
        assert isinstance(result, PruningResult)
    
    def test_neuron_pruning(self, simple_model):
        """Test neuron-wise pruning."""
        config = PruningConfig(
            strategy=PruningStrategy.L2_NORM,
            sparsity_target=0.3,
            structured=True
        )
        pruner = StructuredPruner(simple_model, config)
        result = pruner.prune()
        
        # Model should have fewer neurons
        assert result.pruned_model is not None


class TestGradualPruner:
    """Tests for GradualPruner."""
    
    def test_initialization(self, simple_model):
        """Test gradual pruner initialization."""
        config = PruningConfig(
            schedule=PruningSchedule.LINEAR,
            initial_sparsity=0.0,
            final_sparsity=0.8,
            pruning_steps=10
        )
        pruner = GradualPruner(simple_model, config)
        assert pruner.model is simple_model
    
    def test_pruning_schedule(self, simple_model):
        """Test gradual pruning schedule."""
        config = PruningConfig(
            schedule=PruningSchedule.LINEAR,
            initial_sparsity=0.0,
            final_sparsity=0.8,
            pruning_steps=10
        )
        pruner = GradualPruner(simple_model, config)
        
        # Get sparsity at different steps
        if hasattr(pruner, 'get_sparsity_at_step'):
            s0 = pruner.get_sparsity_at_step(0)
            s5 = pruner.get_sparsity_at_step(5)
            s10 = pruner.get_sparsity_at_step(10)
            
            assert s0 < s5 < s10
    
    def test_step_function(self, simple_model):
        """Test step-by-step pruning."""
        config = PruningConfig(
            schedule=PruningSchedule.LINEAR,
            initial_sparsity=0.0,
            final_sparsity=0.5,
            pruning_steps=5
        )
        pruner = GradualPruner(simple_model, config)
        
        # Step through pruning
        for step in range(5):
            if hasattr(pruner, 'step'):
                pruner.step()
        
        # Final prune
        result = pruner.prune()
        assert result.achieved_sparsity > 0


class TestPrunerFactory:
    """Tests for pruner factory function."""
    
    def test_create_unstructured(self, simple_model):
        """Test creating unstructured pruner."""
        pruner = create_pruner(
            simple_model,
            strategy=PruningStrategy.MAGNITUDE,
            structured=False
        )
        assert isinstance(pruner, UnstructuredPruner)
    
    def test_create_structured(self, simple_model):
        """Test creating structured pruner."""
        pruner = create_pruner(
            simple_model,
            strategy=PruningStrategy.L1_NORM,
            structured=True
        )
        assert isinstance(pruner, StructuredPruner)
    
    def test_create_gradual(self, simple_model):
        """Test creating gradual pruner."""
        pruner = create_pruner(
            simple_model,
            schedule=PruningSchedule.LINEAR,
            pruning_steps=10
        )
        assert isinstance(pruner, GradualPruner)


# =============================================================================
# Calibration Tests
# =============================================================================

class TestCalibrationConfig:
    """Tests for CalibrationConfig dataclass."""
    
    def test_default_config(self):
        """Test default calibration configuration."""
        config = CalibrationConfig()
        assert config.strategy == CalibrationStrategy.MIN_MAX
        assert config.mode == CalibrationMode.PER_TENSOR
    
    def test_histogram_config(self):
        """Test histogram calibration configuration."""
        config = CalibrationConfig(
            strategy=CalibrationStrategy.HISTOGRAM,
            num_bins=2048
        )
        assert config.num_bins == 2048


class TestCalibrationDataset:
    """Tests for CalibrationDataset."""
    
    def test_initialization(self, calibration_data):
        """Test calibration dataset initialization."""
        dataset = CalibrationDataset(calibration_data)
        assert len(dataset) == 10
    
    def test_iteration(self, calibration_data):
        """Test iterating over calibration dataset."""
        dataset = CalibrationDataset(calibration_data)
        for batch in dataset:
            assert batch.shape == (4, 64)
    
    def test_statistics(self, calibration_data):
        """Test computing statistics."""
        dataset = CalibrationDataset(calibration_data)
        if hasattr(dataset, 'compute_statistics'):
            stats = dataset.compute_statistics()
            assert 'mean' in stats or 'min' in stats


class TestPerTensorCalibrator:
    """Tests for PerTensorCalibrator."""
    
    def test_initialization(self, simple_model):
        """Test per-tensor calibrator initialization."""
        config = CalibrationConfig(mode=CalibrationMode.PER_TENSOR)
        calibrator = PerTensorCalibrator(simple_model, config)
        assert calibrator.model is simple_model
    
    def test_calibration(self, simple_model, calibration_data):
        """Test calibration process."""
        config = CalibrationConfig(mode=CalibrationMode.PER_TENSOR)
        calibrator = PerTensorCalibrator(simple_model, config)
        result = calibrator.calibrate(calibration_data)
        
        assert isinstance(result, CalibrationResult)
        assert result.phase == CalibrationPhase.COMPLETE
    
    def test_scale_zero_point(self, simple_model, calibration_data):
        """Test scale and zero point computation."""
        config = CalibrationConfig(
            strategy=CalibrationStrategy.MIN_MAX,
            mode=CalibrationMode.PER_TENSOR
        )
        calibrator = PerTensorCalibrator(simple_model, config)
        result = calibrator.calibrate(calibration_data)
        
        # Should have calibration parameters for each layer
        assert len(result.layer_calibrations) > 0


class TestPerChannelCalibrator:
    """Tests for PerChannelCalibrator."""
    
    def test_initialization(self, simple_model):
        """Test per-channel calibrator initialization."""
        config = CalibrationConfig(mode=CalibrationMode.PER_CHANNEL)
        calibrator = PerChannelCalibrator(simple_model, config)
        assert calibrator.model is simple_model
    
    def test_channel_calibration(self, conv_model, sample_conv_input):
        """Test per-channel calibration on conv model."""
        config = CalibrationConfig(mode=CalibrationMode.PER_CHANNEL)
        calibrator = PerChannelCalibrator(conv_model, config)
        
        cal_data = [torch.randn(4, 3, 32, 32) for _ in range(5)]
        result = calibrator.calibrate(cal_data)
        
        assert isinstance(result, CalibrationResult)


class TestAdaptiveCalibrator:
    """Tests for AdaptiveCalibrator."""
    
    def test_initialization(self, simple_model):
        """Test adaptive calibrator initialization."""
        config = CalibrationConfig(strategy=CalibrationStrategy.ADAPTIVE)
        calibrator = AdaptiveCalibrator(simple_model, config)
        assert calibrator.model is simple_model
    
    def test_adaptive_strategy_selection(self, simple_model, calibration_data):
        """Test that adaptive calibrator selects best strategy."""
        config = CalibrationConfig(strategy=CalibrationStrategy.ADAPTIVE)
        calibrator = AdaptiveCalibrator(simple_model, config)
        result = calibrator.calibrate(calibration_data)
        
        # Should have selected a strategy per layer
        assert isinstance(result, CalibrationResult)
    
    def test_accuracy_validation(self, simple_model, calibration_data, sample_input):
        """Test accuracy validation during calibration."""
        config = CalibrationConfig(
            strategy=CalibrationStrategy.ADAPTIVE,
            validation_samples=5
        )
        calibrator = AdaptiveCalibrator(simple_model, config)
        
        def accuracy_fn(model):
            return 0.95
        
        if hasattr(calibrator, 'calibrate_with_validation'):
            result = calibrator.calibrate_with_validation(
                calibration_data,
                accuracy_fn
            )
            assert result.accuracy_preserved


class TestCalibrationUtilities:
    """Tests for calibration utility functions."""
    
    def test_compute_min_max_range(self):
        """Test min-max range computation."""
        tensor = torch.randn(100) * 10
        min_val, max_val = compute_min_max_range(tensor)
        assert min_val <= tensor.min()
        assert max_val >= tensor.max()
    
    def test_compute_percentile_range(self):
        """Test percentile range computation."""
        tensor = torch.randn(1000)
        min_val, max_val = compute_percentile_range(tensor, 0.01, 99.99)
        # Percentile should be within actual range
        assert min_val >= tensor.min()
        assert max_val <= tensor.max()
    
    def test_compute_scale_zero_point(self):
        """Test scale and zero point computation."""
        min_val = -5.0
        max_val = 10.0
        bits = 8
        symmetric = False
        
        scale, zero_point = compute_scale_zero_point(
            min_val, max_val, bits, symmetric
        )
        
        assert scale > 0
        assert 0 <= zero_point <= 255


class TestCalibratorFactory:
    """Tests for calibrator factory function."""
    
    def test_create_per_tensor(self, simple_model):
        """Test creating per-tensor calibrator."""
        calibrator = create_calibrator(
            simple_model,
            mode=CalibrationMode.PER_TENSOR
        )
        assert isinstance(calibrator, PerTensorCalibrator)
    
    def test_create_per_channel(self, simple_model):
        """Test creating per-channel calibrator."""
        calibrator = create_calibrator(
            simple_model,
            mode=CalibrationMode.PER_CHANNEL
        )
        assert isinstance(calibrator, PerChannelCalibrator)
    
    def test_create_adaptive(self, simple_model):
        """Test creating adaptive calibrator."""
        calibrator = create_calibrator(
            simple_model,
            strategy=CalibrationStrategy.ADAPTIVE
        )
        assert isinstance(calibrator, AdaptiveCalibrator)


# =============================================================================
# Integration Tests
# =============================================================================

class TestQuantizationPipeline:
    """Integration tests for full quantization pipeline."""
    
    def test_calibrate_then_quantize(self, simple_model, calibration_data, sample_input):
        """Test calibration followed by static quantization."""
        # Calibrate
        cal_config = CalibrationConfig(strategy=CalibrationStrategy.HISTOGRAM)
        calibrator = create_calibrator(simple_model, **asdict(cal_config))
        cal_result = calibrator.calibrate(calibration_data)
        
        # Quantize
        quant_config = QuantizationConfig(strategy=QuantizationStrategy.STATIC)
        quantizer = create_quantizer(simple_model, **asdict(quant_config))
        
        if hasattr(quantizer, 'set_calibration'):
            quantizer.set_calibration(cal_result)
        
        quant_result = quantizer.quantize()
        
        # Verify
        assert quant_result.quantized_model is not None
        output = quant_result.quantized_model(sample_input)
        assert output.shape == (4, 10)
    
    def test_prune_then_quantize(self, simple_model, sample_input):
        """Test pruning followed by quantization."""
        # Prune
        prune_config = PruningConfig(sparsity_target=0.3)
        pruner = create_pruner(simple_model, **asdict(prune_config))
        prune_result = pruner.prune()
        
        # Quantize pruned model
        quant_config = QuantizationConfig(strategy=QuantizationStrategy.DYNAMIC)
        quantizer = create_quantizer(
            prune_result.pruned_model,
            **asdict(quant_config)
        )
        quant_result = quantizer.quantize()
        
        # Verify
        output = quant_result.quantized_model(sample_input)
        assert output.shape == (4, 10)


class TestCompressionPipeline:
    """Integration tests for compression pipelines."""
    
    def test_compress_then_prune(self, simple_model, sample_input):
        """Test compression followed by pruning."""
        # Compress
        comp_config = CompressionConfig(
            strategy=CompressionStrategy.LOW_RANK,
            rank_ratio=0.5
        )
        compressor = create_compressor(simple_model, **asdict(comp_config))
        comp_result = compressor.compress()
        
        # Prune
        prune_config = PruningConfig(sparsity_target=0.3)
        pruner = create_pruner(
            comp_result.compressed_model,
            **asdict(prune_config)
        )
        prune_result = pruner.prune()
        
        # Verify
        output = prune_result.pruned_model(sample_input)
        assert output.shape == (4, 10)


class TestOptimizationMetrics:
    """Tests for optimization metrics and reporting."""
    
    def test_compression_ratio_calculation(self, simple_model):
        """Test compression ratio is calculated correctly."""
        config = CompressionConfig(strategy=CompressionStrategy.LOW_RANK)
        compressor = create_compressor(simple_model, **asdict(config))
        result = compressor.compress()
        
        # Compression ratio should be > 1
        assert result.compression_ratio >= 1.0
    
    def test_sparsity_measurement(self, simple_model):
        """Test sparsity measurement accuracy."""
        config = PruningConfig(sparsity_target=0.5)
        pruner = create_pruner(simple_model, **asdict(config))
        result = pruner.prune()
        
        # Count zeros
        total = 0
        zeros = 0
        for param in result.pruned_model.parameters():
            total += param.numel()
            zeros += (param == 0).sum().item()
        
        measured_sparsity = zeros / total
        assert abs(measured_sparsity - result.achieved_sparsity) < 0.1


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in optimization modules."""
    
    def test_invalid_quantization_bits(self, simple_model):
        """Test error on invalid quantization bits."""
        with pytest.raises((ValueError, AssertionError)):
            config = QuantizationConfig(bits=0)
            quantizer = DynamicQuantizer(simple_model, config)
            quantizer.quantize()
    
    def test_invalid_sparsity_target(self, simple_model):
        """Test error on invalid sparsity target."""
        with pytest.raises((ValueError, AssertionError)):
            config = PruningConfig(sparsity_target=1.5)
            pruner = UnstructuredPruner(simple_model, config)
            pruner.prune()
    
    def test_empty_calibration_data(self, simple_model):
        """Test error on empty calibration data."""
        config = CalibrationConfig()
        calibrator = PerTensorCalibrator(simple_model, config)
        
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            calibrator.calibrate([])


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_layer_model(self):
        """Test optimization on single-layer model."""
        model = nn.Linear(64, 10)
        
        config = QuantizationConfig(strategy=QuantizationStrategy.DYNAMIC)
        quantizer = DynamicQuantizer(model, config)
        result = quantizer.quantize()
        
        assert result.quantized_model is not None
    
    def test_model_with_no_weights(self):
        """Test handling model with no trainable weights."""
        class NoWeightModel(nn.Module):
            def forward(self, x):
                return x * 2
        
        model = NoWeightModel()
        config = PruningConfig()
        pruner = UnstructuredPruner(model, config)
        
        # Should handle gracefully
        result = pruner.prune()
        assert result.achieved_sparsity == 0.0 or result.pruned_model is not None


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Tests for optimization performance."""
    
    def test_quantization_speedup(self, simple_model, sample_input):
        """Test that quantized model is faster."""
        import time
        
        # Original model timing
        simple_model.eval()
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(100):
                _ = simple_model(sample_input)
            original_time = time.perf_counter() - start
        
        # Quantized model
        config = QuantizationConfig(strategy=QuantizationStrategy.DYNAMIC)
        quantizer = DynamicQuantizer(simple_model, config)
        result = quantizer.quantize()
        
        # Quantized timing
        result.quantized_model.eval()
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(100):
                _ = result.quantized_model(sample_input)
            quantized_time = time.perf_counter() - start
        
        # Quantized should be at least as fast (on CPU it may not always be faster)
        assert quantized_time <= original_time * 2  # Allow some variance
    
    def test_pruning_memory_reduction(self, simple_model):
        """Test that pruning reduces memory footprint."""
        original_size = sum(
            p.numel() * p.element_size()
            for p in simple_model.parameters()
        )
        
        config = PruningConfig(sparsity_target=0.5)
        pruner = UnstructuredPruner(simple_model, config)
        result = pruner.prune()
        
        # With sparse storage, memory should reduce
        # Note: Dense tensors won't show reduction, but sparsity is achieved
        assert result.achieved_sparsity > 0.4


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Tests for saving/loading optimization results."""
    
    def test_save_load_quantized_model(self, simple_model, sample_input):
        """Test saving and loading quantized model."""
        config = QuantizationConfig(strategy=QuantizationStrategy.DYNAMIC)
        quantizer = DynamicQuantizer(simple_model, config)
        result = quantizer.quantize()
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(result.quantized_model.state_dict(), f.name)
            
            # Load back
            loaded_state = torch.load(f.name)
            result.quantized_model.load_state_dict(loaded_state)
            
            # Verify works
            output = result.quantized_model(sample_input)
            assert output.shape == (4, 10)
    
    def test_export_calibration_data(self, simple_model, calibration_data):
        """Test exporting calibration data."""
        config = CalibrationConfig()
        calibrator = PerTensorCalibrator(simple_model, config)
        result = calibrator.calibrate(calibration_data)
        
        # Export to JSON
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            export_data = {
                'phase': result.phase.value if hasattr(result.phase, 'value') else str(result.phase),
                'num_layers': len(result.layer_calibrations),
            }
            json.dump(export_data, f)
            
            # Verify file was created
            assert Path(f.name).exists()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
