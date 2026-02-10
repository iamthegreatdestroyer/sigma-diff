"""
Ryzanstein Model Optimization & Quantization Module - Sprint 4.2

Comprehensive model optimization capabilities for inference acceleration:

Quantization:
    - INT8 quantization with <2% accuracy loss
    - INT4 quantization for memory-constrained deployment
    - Dynamic quantization for variable precision
    - Static quantization with calibration
    - Mixed-precision optimization

Compression:
    - Weight sharing and clustering
    - Low-rank factorization (SVD, Tucker)
    - Knowledge distillation hooks
    - Huffman encoding for storage

Pruning:
    - Structured pruning (channel, layer)
    - Unstructured pruning (weight, activation)
    - Importance scoring (gradient, magnitude, Taylor)
    - Gradual pruning schedules

Calibration:
    - Calibration dataset management
    - Per-channel and per-tensor calibration
    - Accuracy validation and monitoring
    - Hardware-specific auto-tuning

Performance Targets:
    - 2-4x latency improvement
    - 50-75% memory reduction
    - <2% accuracy loss for INT8
    - Hardware-optimized execution
"""

from typing import Dict, List, Optional, Any

# =============================================================================
# Quantization Components
# =============================================================================
from src.optimization.quantizer import (
    QuantizationStrategy,
    QuantizationMode,
    QuantizationConfig,
    QuantizationMetrics,
    QuantizationResult,
    LayerQuantizationInfo,
    Quantizer,
    DynamicQuantizer,
    StaticQuantizer,
    MixedPrecisionQuantizer,
    create_quantizer,
)

# =============================================================================
# Compression Components
# =============================================================================
from src.optimization.compressor import (
    CompressionStrategy,
    CompressionConfig,
    CompressionMetrics,
    CompressionResult,
    WeightCluster,
    ModelCompressor,
    WeightSharingCompressor,
    LowRankCompressor,
    DistillationCompressor,
    LowRankLinear,
    WeightSharedLinear,
    create_compressor,
)

# =============================================================================
# Pruning Components
# =============================================================================
from src.optimization.pruner import (
    PruningStrategy,
    PruningSchedule,
    PruningConfig,
    PruningMetrics,
    PruningResult,
    PruningMask,
    ModelPruner,
    StructuredPruner,
    UnstructuredPruner,
    GradualPruner,
    create_pruner,
)

# =============================================================================
# Calibration Components
# =============================================================================
from src.optimization.calibrator import (
    CalibrationStrategy,
    CalibrationMode,
    CalibrationPhase,
    CalibrationConfig,
    CalibrationMetrics,
    CalibrationResult,
    CalibrationDataset,
    ModelCalibrator,
    PerChannelCalibrator,
    PerTensorCalibrator,
    AdaptiveCalibrator,
    create_calibrator,
    compute_min_max_range,
    compute_percentile_range,
    compute_scale_zero_point,
)

__all__ = [
    # Quantization
    "QuantizationStrategy",
    "QuantizationMode",
    "QuantizationConfig",
    "QuantizationMetrics",
    "QuantizationResult",
    "LayerQuantizationInfo",
    "Quantizer",
    "DynamicQuantizer",
    "StaticQuantizer",
    "MixedPrecisionQuantizer",
    "create_quantizer",
    # Compression
    "CompressionStrategy",
    "CompressionConfig",
    "CompressionMetrics",
    "CompressionResult",
    "WeightCluster",
    "ModelCompressor",
    "WeightSharingCompressor",
    "LowRankCompressor",
    "DistillationCompressor",
    "LowRankLinear",
    "WeightSharedLinear",
    "create_compressor",
    # Pruning
    "PruningStrategy",
    "PruningSchedule",
    "PruningConfig",
    "PruningMetrics",
    "PruningResult",
    "PruningMask",
    "ModelPruner",
    "StructuredPruner",
    "UnstructuredPruner",
    "GradualPruner",
    "create_pruner",
    # Calibration
    "CalibrationStrategy",
    "CalibrationMode",
    "CalibrationPhase",
    "CalibrationConfig",
    "CalibrationMetrics",
    "CalibrationResult",
    "CalibrationDataset",
    "ModelCalibrator",
    "PerChannelCalibrator",
    "PerTensorCalibrator",
    "AdaptiveCalibrator",
    "create_calibrator",
    "compute_min_max_range",
    "compute_percentile_range",
    "compute_scale_zero_point",
]

__version__ = "4.2.0"
__sprint__ = "4.2"
__theme__ = "Model Optimization & Quantization"
