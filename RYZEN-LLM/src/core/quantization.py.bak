"""
High-level Python quantization API for BitNet.

This module provides a Pythonic interface to the C++ quantization engine,
handling common workflows and providing sensible defaults.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings

# Try to import C++ bindings, fall back to pure Python if not available
_USE_CPP_BINDINGS = False
_IMPORT_ERROR = None

# First, check if bindings should even be attempted
# (can be disabled via environment variable or if previous crashes detected)
import os
_SKIP_CPP_BINDINGS = os.environ.get('RYZEN_LLM_PURE_PYTHON', '0') == '1'

if not _SKIP_CPP_BINDINGS:
    try:
        import sys
        import platform
        import subprocess
        
        # Check CPU features before attempting import
        # AVX-512 instructions crash on CPUs that don't support them
        def _check_cpu_avx512_support():
            """Check if CPU supports AVX-512 instructions."""
            try:
                # On Windows, check using CPUID via platform module
                if platform.system() == 'Windows':
                    # Simple heuristic: check processor name for known AVX-512 CPUs
                    proc = platform.processor()
                    # Intel 10th gen+ Core and Xeon, AMD Zen 4+
                    avx512_indicators = ['11th Gen', '12th Gen', '13th Gen', '14th Gen', 
                                        'Xeon', 'Zen 4', '7000 Series', '9000 Series']
                    return any(ind in proc for ind in avx512_indicators)
                else:
                    # On Linux, check /proc/cpuinfo
                    try:
                        with open('/proc/cpuinfo', 'r') as f:
                            return 'avx512' in f.read().lower()
                    except:
                        return False
            except:
                return False
        
        _has_avx512 = _check_cpu_avx512_support()
        if not _has_avx512:
            warnings.warn(
                "CPU may not support AVX-512. Skipping C++ bindings to prevent crash. "
                "Set RYZEN_LLM_PURE_PYTHON=0 to force C++ bindings."
            )
            _SKIP_CPP_BINDINGS = True
        
        if not _SKIP_CPP_BINDINGS:
            # Add build path for bindings
            build_python_path = Path(__file__).parent.parent.parent / "build" / "python"
            if build_python_path.exists():
                sys.path.insert(0, str(build_python_path))
            
            from ryzen_llm.ryzen_llm_bindings import (
                QuantConfig as CppQuantConfig,
                TernaryWeight,
                QuantizedActivation,
                quantize_weights_ternary as cpp_quantize_weights,
                quantize_activations_int8 as cpp_quantize_activations,
                dequantize_weights as cpp_dequantize_weights,
                dequantize_activations as cpp_dequantize_activations,
                compute_quantization_error as cpp_compute_error,
            )
            _USE_CPP_BINDINGS = True
    except (ImportError, OSError, Exception) as e:
        _IMPORT_ERROR = str(e)
        warnings.warn(
            f"C++ bindings not available ({type(e).__name__}: {e}). "
            "Using pure Python fallback (slower but functional)."
        )

if not _USE_CPP_BINDINGS:
    @dataclass
    class CppQuantConfig:
        """Pure Python fallback for C++ QuantConfig."""
        weight_group_size: int = 128
        per_group_scaling: bool = True
        activation_clip_value: float = 6.0
        symmetric_activations: bool = True
    
    @dataclass
    class TernaryWeight:
        """Pure Python fallback for C++ TernaryWeight."""
        data: np.ndarray  # ternary values (-1, 0, +1)
        scales: np.ndarray  # per-group scales
        rows: int
        cols: int
        
        @property
        def packed_data(self) -> bytes:
            return self.data.tobytes()
    
    @dataclass
    class QuantizedActivation:
        """Pure Python fallback for C++ QuantizedActivation."""
        data: np.ndarray  # int8 values
        scale: float
        zero_point: int
        size: int
        
        @property
        def packed_data(self) -> bytes:
            return self.data.tobytes()
    
    def cpp_quantize_weights(weights: np.ndarray, rows: int, cols: int, config: CppQuantConfig) -> TernaryWeight:
        """Pure Python ternary weight quantization."""
        weights_2d = weights.reshape(rows, cols)
        
        group_size = config.weight_group_size
        num_groups = max(1, cols // group_size)
        
        # Compute per-group scales
        scales = np.zeros(num_groups, dtype=np.float32)
        ternary = np.zeros_like(weights_2d, dtype=np.int8)
        
        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, cols)
            group = weights_2d[:, start:end]
            
            # Scale is max absolute value
            scale = np.max(np.abs(group)) + 1e-10
            scales[g] = scale
            
            # Quantize to ternary: round(x/scale) clamped to {-1, 0, 1}
            normalized = group / scale
            ternary[:, start:end] = np.clip(np.round(normalized), -1, 1).astype(np.int8)
        
        return TernaryWeight(data=ternary, scales=scales, rows=rows, cols=cols)
    
    def cpp_quantize_activations(activations: np.ndarray, config: CppQuantConfig) -> QuantizedActivation:
        """Pure Python int8 activation quantization."""
        clip_val = config.activation_clip_value
        clipped = np.clip(activations, -clip_val, clip_val)
        
        scale = (2 * clip_val) / 255.0
        zero_point = 128
        
        quantized = np.round(clipped / scale + zero_point).astype(np.int8)
        
        return QuantizedActivation(
            data=quantized,
            scale=float(scale),
            zero_point=zero_point,
            size=activations.size
        )
    
    def cpp_dequantize_weights(ternary: TernaryWeight) -> np.ndarray:
        """Pure Python ternary weight dequantization."""
        weights = np.zeros((ternary.rows, ternary.cols), dtype=np.float32)
        group_size = ternary.cols // len(ternary.scales) if len(ternary.scales) > 0 else ternary.cols
        
        for g, scale in enumerate(ternary.scales):
            start = g * group_size
            end = min(start + group_size, ternary.cols)
            weights[:, start:end] = ternary.data[:, start:end].astype(np.float32) * scale
        
        return weights
    
    def cpp_dequantize_activations(quant: QuantizedActivation) -> np.ndarray:
        """Pure Python int8 activation dequantization."""
        return (quant.data.astype(np.float32) - quant.zero_point) * quant.scale
    
    def cpp_compute_error(original: np.ndarray, dequantized: np.ndarray) -> float:
        """Compute quantization error (MSE)."""
        return float(np.mean((original - dequantized) ** 2))


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class QuantizationConfig:
    """
    Configuration for BitNet quantization.
    
    Attributes:
        weight_group_size: Size of groups for per-group scaling (default: 128)
        per_group_scaling: Enable per-group scaling (default: True)
        activation_clip_value: Clipping threshold for activations (default: 6.0)
        symmetric_activations: Use symmetric INT8 quantization (default: True)
        dtype_weights: Data type for input weights (default: np.float32)
        dtype_activations: Data type for input activations (default: np.float32)
    """
    weight_group_size: int = 128
    per_group_scaling: bool = True
    activation_clip_value: float = 6.0
    symmetric_activations: bool = True
    dtype_weights: np.dtype = np.float32
    dtype_activations: np.dtype = np.float32
    
    def to_cpp_config(self) -> CppQuantConfig:
        """Convert to C++ QuantConfig."""
        config = CppQuantConfig()
        config.weight_group_size = self.weight_group_size
        config.per_group_scaling = self.per_group_scaling
        config.activation_clip_value = self.activation_clip_value
        config.symmetric_activations = self.symmetric_activations
        return config
    
    @classmethod
    def from_cpp_config(cls, cpp_config: CppQuantConfig) -> "QuantizationConfig":
        """Create from C++ QuantConfig."""
        return cls(
            weight_group_size=cpp_config.weight_group_size,
            per_group_scaling=cpp_config.per_group_scaling,
            activation_clip_value=cpp_config.activation_clip_value,
            symmetric_activations=cpp_config.symmetric_activations,
        )
    
    def __repr__(self) -> str:
        return (f"QuantizationConfig("
                f"group_size={self.weight_group_size}, "
                f"per_group={self.per_group_scaling}, "
                f"clip={self.activation_clip_value})")


# ============================================================================
# Quantization Engine
# ============================================================================

class QuantizationEngine:
    """
    High-level interface to BitNet quantization.
    
    Handles weight quantization, activation quantization, and provides
    utilities for measuring quantization accuracy.
    """
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize quantization engine.
        
        Args:
            config: QuantizationConfig instance (uses defaults if None)
        """
        self.config = config or QuantizationConfig()
        self._cpp_config = self.config.to_cpp_config()
        self._weight_cache: Dict[str, TernaryWeight] = {}
        self._activation_cache: Dict[str, QuantizedActivation] = {}
    
    def quantize_weights(
        self,
        weights: np.ndarray,
        name: Optional[str] = None,
        cache: bool = False,
    ) -> TernaryWeight:
        """
        Quantize weight matrix to ternary.
        
        Args:
            weights: FP32 weight array (can be 1D or 2D)
            name: Optional name for caching
            cache: Whether to cache the quantized result
            
        Returns:
            TernaryWeight object containing quantized values and scales
            
        Raises:
            ValueError: If weights are not FP32
            ValueError: If weights are empty
            
        Example:
            >>> engine = QuantizationEngine()
            >>> weights = np.random.randn(768, 3072).astype(np.float32)
            >>> ternary = engine.quantize_weights(weights, name="attn_weights")
            >>> print(f"Quantized: {ternary.rows}×{ternary.cols}")
        """
        if weights.dtype != np.float32:
            raise ValueError(
                f"Expected FP32 weights, got {weights.dtype}. "
                f"Convert with: weights.astype(np.float32)"
            )
        
        if weights.size == 0:
            raise ValueError("Cannot quantize empty weight array")
        
        # Check cache
        if cache and name and name in self._weight_cache:
            return self._weight_cache[name]
        
        # Handle 1D weights by reshaping
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
        elif weights.ndim != 2:
            raise ValueError(
                f"Weights must be 1D or 2D, got shape {weights.ndim}D"
            )
        
        rows, cols = weights.shape
        
        # Quantize
        ternary = cpp_quantize_weights(weights, rows, cols, self._cpp_config)
        
        # Cache if requested
        if cache and name:
            self._weight_cache[name] = ternary
        
        return ternary
    
    def quantize_activations(
        self,
        activations: np.ndarray,
        name: Optional[str] = None,
        cache: bool = False,
    ) -> QuantizedActivation:
        """
        Quantize activation tensor to INT8.
        
        Args:
            activations: FP32 activation array
            name: Optional name for caching
            cache: Whether to cache the quantized result
            
        Returns:
            QuantizedActivation object containing quantized values and scale
            
        Raises:
            ValueError: If activations are not FP32
            ValueError: If activations are empty
            
        Example:
            >>> engine = QuantizationEngine()
            >>> acts = np.random.randn(1024).astype(np.float32)
            >>> quant_acts = engine.quantize_activations(acts)
            >>> print(f"Scale factor: {quant_acts.scale}")
        """
        if activations.dtype != np.float32:
            raise ValueError(
                f"Expected FP32 activations, got {activations.dtype}. "
                f"Convert with: activations.astype(np.float32)"
            )
        
        if activations.size == 0:
            raise ValueError("Cannot quantize empty activation array")
        
        # Check cache
        if cache and name and name in self._activation_cache:
            return self._activation_cache[name]
        
        # Flatten to 1D for quantization
        activations_1d = activations.ravel()
        
        # Quantize
        quant_act = cpp_quantize_activations(activations_1d, self._cpp_config)
        
        # Cache if requested
        if cache and name:
            self._activation_cache[name] = quant_act
        
        return quant_act
    
    def dequantize_weights(self, ternary: TernaryWeight) -> np.ndarray:
        """
        Recover FP32 weights from ternary representation.
        
        Args:
            ternary: TernaryWeight object
            
        Returns:
            FP32 numpy array of shape (rows, cols)
            
        Example:
            >>> ternary = engine.quantize_weights(weights)
            >>> recovered = engine.dequantize_weights(ternary)
            >>> error = np.mean((weights - recovered)**2)
        """
        return cpp_dequantize_weights(ternary)
    
    def dequantize_activations(self, quant_act: QuantizedActivation) -> np.ndarray:
        """
        Recover FP32 activations from INT8 representation.
        
        Args:
            quant_act: QuantizedActivation object
            
        Returns:
            FP32 numpy array
            
        Example:
            >>> quant_acts = engine.quantize_activations(acts)
            >>> recovered = engine.dequantize_activations(quant_acts)
        """
        return cpp_dequantize_activations(quant_act)
    
    def compute_error(
        self,
        original: np.ndarray,
        quantized: np.ndarray,
    ) -> float:
        """
        Compute mean squared error between original and quantized values.
        
        Args:
            original: Original FP32 array
            quantized: Quantized FP32 array (same shape as original)
            
        Returns:
            MSE error (float)
            
        Raises:
            ValueError: If array shapes don't match
            
        Example:
            >>> error = engine.compute_error(weights, recovered)
            >>> print(f"Quantization MSE: {error:.6f}")
        """
        if original.shape != quantized.shape:
            raise ValueError(
                f"Shape mismatch: original {original.shape} vs "
                f"quantized {quantized.shape}"
            )
        
        return float(cpp_compute_error(original, quantized))
    
    def quantize_and_measure(
        self,
        weights: np.ndarray,
        recover: bool = True,
    ) -> Dict[str, Union[TernaryWeight, np.ndarray, float]]:
        """
        Quantize weights and measure quantization error.
        
        Args:
            weights: FP32 weight array
            recover: Whether to compute error (requires dequantization)
            
        Returns:
            Dictionary with keys:
            - 'ternary': TernaryWeight object
            - 'recovered': Recovered FP32 array (if recover=True)
            - 'error': MSE error (if recover=True)
            - 'compression': Compression ratio
            
        Example:
            >>> result = engine.quantize_and_measure(weights)
            >>> print(f"Error: {result['error']:.6f}")
            >>> print(f"Compression: {result['compression']:.1f}×")
        """
        ternary = self.quantize_weights(weights)
        
        # Calculate compression ratio
        # Original: FP32 = 4 bytes/element
        # Ternary: 2 bits/weight + scales (FP32 per group)
        original_bytes = weights.nbytes
        # Rough estimate: (rows*cols bits / 8) + (num_scales * 4 bytes)
        ternary_size = ternary.size()
        num_scales = ternary.num_scales()
        ternary_bytes = max(ternary_size // 4 + num_scales * 4, 1)
        
        result = {
            'ternary': ternary,
            'compression': original_bytes / ternary_bytes,
        }
        
        if recover:
            recovered = self.dequantize_weights(ternary)
            error = self.compute_error(weights, recovered)
            result['recovered'] = recovered
            result['error'] = error
        
        return result
    
    def clear_cache(self, weights: bool = True, activations: bool = True):
        """
        Clear quantization caches.
        
        Args:
            weights: Clear weight cache
            activations: Clear activation cache
        """
        if weights:
            self._weight_cache.clear()
        if activations:
            self._activation_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cached_weights': len(self._weight_cache),
            'cached_activations': len(self._activation_cache),
        }


# ============================================================================
# Batch Quantization
# ============================================================================

class BatchQuantizer:
    """
    Quantize multiple weight matrices with consistent configuration.
    """
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize batch quantizer.
        
        Args:
            config: QuantizationConfig instance
        """
        self.engine = QuantizationEngine(config)
    
    def quantize_dict(
        self,
        weights_dict: Dict[str, np.ndarray],
        measure_error: bool = False,
    ) -> Dict[str, Union[TernaryWeight, float]]:
        """
        Quantize a dictionary of weight matrices.
        
        Args:
            weights_dict: Dict mapping names to FP32 weight arrays
            measure_error: Whether to compute quantization errors
            
        Returns:
            Dictionary mapping names to quantized weights (and errors if requested)
            
        Example:
            >>> quantizer = BatchQuantizer()
            >>> weights = {
            ...     'layer1_weight': np.random.randn(768, 3072).astype(np.float32),
            ...     'layer2_weight': np.random.randn(3072, 768).astype(np.float32),
            ... }
            >>> ternary_weights = quantizer.quantize_dict(weights)
        """
        results = {}
        
        for name, weights in weights_dict.items():
            try:
                ternary = self.engine.quantize_weights(weights, name=name, cache=True)
                results[name] = ternary
                
                if measure_error:
                    recovered = self.engine.dequantize_weights(ternary)
                    error = self.engine.compute_error(weights, recovered)
                    results[f"{name}_error"] = error
                
            except Exception as e:
                warnings.warn(f"Failed to quantize {name}: {e}")
                continue
        
        return results
    
    def quantize_layer_weights(
        self,
        layer_dict: Dict[str, np.ndarray],
    ) -> Dict[str, TernaryWeight]:
        """
        Quantize all weight matrices in a transformer layer.
        
        Typical layer structure:
        - self_attn.q_proj: (hidden_size, hidden_size)
        - self_attn.k_proj: (hidden_size, hidden_size)
        - self_attn.v_proj: (hidden_size, hidden_size)
        - self_attn.o_proj: (hidden_size, hidden_size)
        - mlp.fc1: (hidden_size, intermediate_size)
        - mlp.fc2: (intermediate_size, hidden_size)
        
        Args:
            layer_dict: Dict of layer weights
            
        Returns:
            Dictionary of quantized weights with same keys
        """
        return self.quantize_dict(layer_dict, measure_error=False)


# ============================================================================
# Utilities
# ============================================================================

def create_default_config() -> QuantizationConfig:
    """Create default quantization configuration optimized for BitNet."""
    return QuantizationConfig(
        weight_group_size=128,
        per_group_scaling=True,
        activation_clip_value=6.0,
        symmetric_activations=True,
    )


def create_aggressive_config() -> QuantizationConfig:
    """Create aggressive quantization config for maximum compression."""
    return QuantizationConfig(
        weight_group_size=256,  # Larger groups = more compression
        per_group_scaling=True,
        activation_clip_value=4.0,  # Tighter clipping
        symmetric_activations=True,
    )


def estimate_model_size(
    weights_dict: Dict[str, Tuple[int, int]],
    original_dtype_bits: int = 32,
) -> Dict[str, float]:
    """
    Estimate model size before and after quantization.
    
    Args:
        weights_dict: Dict mapping names to (rows, cols) tuples
        original_dtype_bits: Bits per element in original format
        
    Returns:
        Dictionary with size estimates in MB
        
    Example:
        >>> weights_shapes = {
        ...     'attn_weights': (768, 768),
        ...     'mlp_weights': (768, 3072),
        ... }
        >>> sizes = estimate_model_size(weights_shapes)
        >>> print(f"Original: {sizes['original_mb']:.1f}MB")
        >>> print(f"Ternary: {sizes['ternary_mb']:.1f}MB")
    """
    total_elements = sum(rows * cols for rows, cols in weights_dict.values())
    original_bytes = total_elements * (original_dtype_bits // 8)
    
    # Ternary: 1.3 bits per element (1 bit value + 1.3 bits scaling)
    # Estimate: ~21% of original size
    ternary_bytes = original_bytes * 0.21
    
    return {
        'original_mb': original_bytes / (1024 ** 2),
        'ternary_mb': ternary_bytes / (1024 ** 2),
        'compression_ratio': original_bytes / ternary_bytes,
    }
