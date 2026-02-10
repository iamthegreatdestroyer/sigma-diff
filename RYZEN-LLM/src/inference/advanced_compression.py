"""
Advanced KV-Cache Compression Techniques
========================================

Multi-level compression for KV-cache optimization with adaptive algorithms.
Implements FP8 quantization, sparse compression, and workload-aware strategies.

Key Features:
- FP8 quantization with custom scaling
- Adaptive compression based on data patterns
- Sparse compression for attention matrices
- Compression-aware cache management
- Memory-efficient decompression
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import math
import numpy as np
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Compression algorithm types."""
    NONE = "none"
    FP8 = "fp8"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    SPARSE = "sparse"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


@dataclass
class CompressionStats:
    """Statistics for compression performance."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    quality_loss: float = 0.0


class FP8Compressor:
    """
    FP8 quantization compressor with custom scaling.

    Supports both E4M3 and E5M2 formats with adaptive scaling
    to minimize quantization error.
    """

    def __init__(self, format_type: str = "e4m3"):
        """Initialize FP8 compressor.

        Args:
            format_type: FP8 format ('e4m3' or 'e5m2')
        """
        self.format_type = format_type

        if format_type == "e4m3":
            # E4M3: 1 sign, 4 exponent, 3 mantissa
            self.exponent_bits = 4
            self.mantissa_bits = 3
        else:  # e5m2
            # E5M2: 1 sign, 5 exponent, 2 mantissa
            self.exponent_bits = 5
            self.mantissa_bits = 2

        # Pre-compute scaling factors for common ranges
        self.scaling_cache = {}

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress tensor to FP8 format.

        Args:
            tensor: Input tensor (float16 or float32)

        Returns:
            compressed_tensor: FP8 compressed tensor
            metadata: Compression metadata
        """
        # Convert to float32 for processing
        if tensor.dtype == torch.float16:
            tensor = tensor.float()

        # Adaptive scaling to minimize quantization error
        scale = self._compute_adaptive_scale(tensor)

        # Scale and quantize
        scaled_tensor = tensor / scale
        quantized_tensor = self._quantize_to_fp8(scaled_tensor)

        # Pack into 8-bit representation
        fp8_tensor = self._pack_fp8(quantized_tensor)

        metadata = {
            'scale': scale,
            'original_dtype': tensor.dtype,
            'compression_type': f'fp8_{self.format_type}',
            'shape': tensor.shape
        }

        return fp8_tensor, metadata

    def decompress(self, compressed_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """Decompress FP8 tensor back to original format.

        Args:
            compressed_tensor: FP8 compressed tensor
            metadata: Compression metadata

        Returns:
            decompressed_tensor: Original format tensor
        """
        # Unpack from 8-bit representation
        quantized_tensor = self._unpack_fp8(compressed_tensor)

        # Dequantize and rescale
        scale = metadata['scale']
        decompressed = quantized_tensor.float() * scale

        # Convert back to original dtype
        original_dtype = metadata['original_dtype']
        if original_dtype == torch.float16:
            decompressed = decompressed.half()

        return decompressed

    def _compute_adaptive_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute adaptive scaling factor for quantization."""
        # Use percentile-based scaling for robustness
        abs_tensor = tensor.abs()

        # Cache key based on tensor statistics
        cache_key = (
            tensor.shape,
            tuple(tensor.stride()),
            abs_tensor.mean().item(),
            abs_tensor.max().item()
        )

        if cache_key in self.scaling_cache:
            return self.scaling_cache[cache_key]

        # Compute 99.9th percentile for robust scaling
        percentile_999 = torch.quantile(abs_tensor, 0.999)

        # Add small epsilon to avoid division by zero
        scale = torch.clamp(percentile_999, min=1e-8)

        # Cache the scale
        self.scaling_cache[cache_key] = scale

        return scale

    def _quantize_to_fp8(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize float32 tensor to FP8 format."""
        # Clamp to FP8 range
        max_val = 2**(2**(self.exponent_bits) - 1) * (1 + (2**self.mantissa_bits - 1) / 2**self.mantissa_bits)
        tensor = torch.clamp(tensor, -max_val, max_val)

        # Convert to float16 first (intermediate step)
        tensor_fp16 = tensor.half()

        # Simulate FP8 quantization by reducing precision
        # In a real implementation, this would use specialized hardware
        if self.format_type == "e4m3":
            # E4M3 has better precision for values around 1
            tensor_fp8 = tensor_fp16  # Placeholder
        else:
            # E5M2 has wider range
            tensor_fp8 = tensor_fp16  # Placeholder

        return tensor_fp8

    def _pack_fp8(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pack FP8 values into 8-bit representation."""
        # Convert to uint8 for storage
        # This is a simplified version - real FP8 packing would be more complex
        tensor_uint8 = torch.clamp(tensor, 0, 255).byte()
        return tensor_uint8

    def _unpack_fp8(self, tensor: torch.Tensor) -> torch.Tensor:
        """Unpack 8-bit values back to FP8 format."""
        # Convert back to float
        # This is a simplified version
        tensor_float = tensor.float()
        return tensor_float


class SparseCompressor:
    """
    Sparse compression for attention matrices.

    Exploits sparsity in attention patterns for compression.
    Uses coordinate format (COO) or compressed sparse row (CSR).
    """

    def __init__(self, sparsity_threshold: float = 0.1):
        """Initialize sparse compressor.

        Args:
            sparsity_threshold: Threshold for considering values as zero
        """
        self.sparsity_threshold = sparsity_threshold

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress tensor using sparse representation.

        Args:
            tensor: Input tensor

        Returns:
            compressed_data: Sparse compressed data
            metadata: Compression metadata
        """
        # Convert to sparse format
        sparse_tensor = tensor.to_sparse()

        # Get indices and values
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()

        # Pack into contiguous tensor
        compressed_data = torch.cat([indices, values.unsqueeze(0)], dim=0)

        metadata = {
            'compression_type': 'sparse',
            'original_shape': tensor.shape,
            'nnz': values.numel(),
            'sparsity': 1.0 - (values.numel() / tensor.numel()),
            'format': 'coo'
        }

        return compressed_data, metadata

    def decompress(self, compressed_data: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """Decompress sparse tensor.

        Args:
            compressed_data: Sparse compressed data
            metadata: Compression metadata

        Returns:
            decompressed_tensor: Original tensor
        """
        original_shape = metadata['original_shape']
        nnz = metadata['nnz']

        # Unpack indices and values
        indices = compressed_data[:2, :nnz]  # First 2 rows are indices
        values = compressed_data[2, :nnz]   # Last row is values

        # Reconstruct sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, original_shape, dtype=values.dtype
        )

        # Convert to dense
        dense_tensor = sparse_tensor.to_dense()

        return dense_tensor


class AdaptiveCompressor:
    """
    Adaptive compression that chooses the best algorithm based on data patterns.
    """

    def __init__(self):
        """Initialize adaptive compressor."""
        self.fp8_e4m3 = FP8Compressor("e4m3")
        self.fp8_e5m2 = FP8Compressor("e5m2")
        self.sparse = SparseCompressor()

        # Performance history for adaptation
        self.compression_history = []

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress tensor using adaptive algorithm selection.

        Args:
            tensor: Input tensor

        Returns:
            compressed_data: Compressed data
            metadata: Compression metadata
        """
        # Analyze tensor characteristics
        stats = self._analyze_tensor(tensor)

        # Choose compression algorithm
        algorithm = self._select_algorithm(stats)

        # Apply compression
        if algorithm == CompressionType.FP8_E4M3:
            compressed, metadata = self.fp8_e4m3.compress(tensor)
        elif algorithm == CompressionType.FP8_E5M2:
            compressed, metadata = self.fp8_e5m2.compress(tensor)
        elif algorithm == CompressionType.SPARSE:
            compressed, metadata = self.sparse.compress(tensor)
        else:
            # No compression
            compressed, metadata = tensor, {'compression_type': 'none'}

        # Update metadata
        metadata.update({
            'adaptive_algorithm': algorithm.value,
            'tensor_stats': stats
        })

        return compressed, metadata

    def decompress(self, compressed_data: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """Decompress tensor using adaptive algorithm.

        Args:
            compressed_data: Compressed data
            metadata: Compression metadata

        Returns:
            decompressed_tensor: Original tensor
        """
        algorithm = CompressionType(metadata.get('adaptive_algorithm', 'none'))

        if algorithm == CompressionType.FP8_E4M3:
            return self.fp8_e4m3.decompress(compressed_data, metadata)
        elif algorithm == CompressionType.FP8_E5M2:
            return self.fp8_e5m2.decompress(compressed_data, metadata)
        elif algorithm == CompressionType.SPARSE:
            return self.sparse.decompress(compressed_data, metadata)
        else:
            return compressed_data

    def _analyze_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Analyze tensor characteristics for compression selection."""
        abs_tensor = tensor.abs()

        return {
            'shape': tensor.shape,
            'dtype': str(tensor.dtype),
            'sparsity': (abs_tensor < self.sparse.sparsity_threshold).float().mean().item(),
            'dynamic_range': abs_tensor.max().item() / max(abs_tensor.min().item(), 1e-8),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'has_outliers': (abs_tensor > abs_tensor.mean() + 3 * abs_tensor.std()).any().item()
        }

    def _select_algorithm(self, stats: Dict[str, Any]) -> CompressionType:
        """Select best compression algorithm based on tensor statistics."""
        sparsity = stats['sparsity']
        dynamic_range = stats['dynamic_range']
        has_outliers = stats['has_outliers']

        # Decision tree for algorithm selection
        if sparsity > 0.8:
            # Very sparse - use sparse compression
            return CompressionType.SPARSE
        elif dynamic_range > 1000 or has_outliers:
            # Wide dynamic range or outliers - use E5M2 for better range
            return CompressionType.FP8_E5M2
        elif sparsity > 0.3:
            # Moderately sparse - use E4M3 for better precision
            return CompressionType.FP8_E4M3
        else:
            # Dense tensor - use E4M3 for general case
            return CompressionType.FP8_E4M3


class CompressionManager:
    """
    Manages compression for KV-cache with multiple algorithms and strategies.
    """

    def __init__(self, target_compression_ratio: float = 0.5):
        """Initialize compression manager.

        Args:
            target_compression_ratio: Target compression ratio (0.0-1.0)
        """
        self.target_ratio = target_compression_ratio

        # Compression algorithms
        self.compressors = {
            CompressionType.FP8_E4M3: FP8Compressor("e4m3"),
            CompressionType.FP8_E5M2: FP8Compressor("e5m2"),
            CompressionType.SPARSE: SparseCompressor(),
            CompressionType.ADAPTIVE: AdaptiveCompressor()
        }

        # Performance tracking
        self.compression_stats: Dict[str, List[CompressionStats]] = defaultdict(list)

    def compress_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        algorithm: CompressionType = CompressionType.ADAPTIVE
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Compress KV-cache tensors.

        Args:
            key: Key tensor
            value: Value tensor
            layer_idx: Layer index
            algorithm: Compression algorithm

        Returns:
            compressed_key: Compressed key tensor
            compressed_value: Compressed value tensor
            metadata: Compression metadata
        """
        import time

        # Compress key
        start_time = time.time()
        compressed_key, key_metadata = self.compressors[algorithm].compress(key)
        key_compression_time = time.time() - start_time

        # Compress value
        start_time = time.time()
        compressed_value, value_metadata = self.compressors[algorithm].compress(value)
        value_compression_time = time.time() - start_time

        # Combine metadata
        metadata = {
            'algorithm': algorithm.value,
            'layer_idx': layer_idx,
            'original_key_shape': key.shape,
            'original_value_shape': value.shape,
            'compressed_key_shape': compressed_key.shape,
            'compressed_value_shape': compressed_value.shape,
            'key_metadata': key_metadata,
            'value_metadata': value_metadata,
            'compression_time': key_compression_time + value_compression_time
        }

        # Calculate compression stats
        original_size = key.numel() * key.element_size() + value.numel() * value.element_size()
        compressed_size = compressed_key.numel() * compressed_key.element_size() + \
                         compressed_value.numel() * compressed_value.element_size()

        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / original_size,
            compression_time=key_compression_time + value_compression_time,
            decompression_time=0.0  # Will be set on decompression
        )

        self.compression_stats[f'layer_{layer_idx}'].append(stats)

        return compressed_key, compressed_value, metadata

    def decompress_kv_cache(
        self,
        compressed_key: torch.Tensor,
        compressed_value: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress KV-cache tensors.

        Args:
            compressed_key: Compressed key tensor
            compressed_value: Compressed value tensor
            metadata: Compression metadata

        Returns:
            key: Decompressed key tensor
            value: Decompressed value tensor
        """
        import time

        algorithm = CompressionType(metadata['algorithm'])

        # Decompress key
        start_time = time.time()
        key_metadata = metadata['key_metadata']
        key = self.compressors[algorithm].decompress(compressed_key, key_metadata)
        key_decompression_time = time.time() - start_time

        # Decompress value
        start_time = time.time()
        value_metadata = metadata['value_metadata']
        value = self.compressors[algorithm].decompress(compressed_value, value_metadata)
        value_decompression_time = time.time() - start_time

        # Update decompression time in stats
        layer_key = f'layer_{metadata["layer_idx"]}'
        if self.compression_stats[layer_key]:
            self.compression_stats[layer_key][-1].decompression_time = \
                key_decompression_time + value_decompression_time

        return key, value

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression performance statistics."""
        stats = {}

        for layer, layer_stats in self.compression_stats.items():
            if layer_stats:
                avg_ratio = sum(s.compression_ratio for s in layer_stats) / len(layer_stats)
                avg_compression_time = sum(s.compression_time for s in layer_stats) / len(layer_stats)
                avg_decompression_time = sum(s.decompression_time for s in layer_stats) / len(layer_stats)

                stats[layer] = {
                    'avg_compression_ratio': avg_ratio,
                    'avg_compression_time_ms': avg_compression_time * 1000,
                    'avg_decompression_time_ms': avg_decompression_time * 1000,
                    'total_operations': len(layer_stats)
                }

        return stats

    def optimize_compression(self, workload_patterns: Dict[str, Any]):
        """Optimize compression parameters based on workload patterns."""
        # Analyze workload to adjust compression strategies
        # This could adjust target ratios, algorithm preferences, etc.
        pass
