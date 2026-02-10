"""
BitNet Ternary Quantization
===========================

Implements 1.58-bit ternary quantization (-1, 0, +1).
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class QuantizedTensor:
    """
    Ternary quantized tensor representation.
    
    Stores weights as packed bits with scale factors.
    """
    # Packed ternary values: 2 bits per weight
    # 00 = -1, 01 = 0, 10 = +1
    packed_weights: np.ndarray  # uint8
    
    # Scale factor per group
    scales: np.ndarray  # float32
    
    # Original shape
    shape: Tuple[int, ...]
    
    # Quantization group size
    group_size: int = 128
    
    def dequantize(self) -> np.ndarray:
        """Dequantize to float32."""
        # Unpack ternary values
        weights = self._unpack_ternary()
        
        # Apply scales
        weights = weights.astype(np.float32)
        
        # Reshape for group scaling
        reshaped = weights.reshape(-1, self.group_size)
        scales = self.scales.reshape(-1, 1)
        
        # Scale and reshape back
        scaled = reshaped * scales
        return scaled.reshape(self.shape)
    
    def _unpack_ternary(self) -> np.ndarray:
        """Unpack packed ternary values."""
        # Each byte contains 4 ternary values (2 bits each)
        result = []
        
        for byte in self.packed_weights.flat:
            for shift in [0, 2, 4, 6]:
                val = (byte >> shift) & 0b11
                if val == 0:
                    result.append(-1)
                elif val == 1:
                    result.append(0)
                else:
                    result.append(1)
        
        # Trim to actual size
        total_elements = int(np.prod(self.shape))
        return np.array(result[:total_elements], dtype=np.int8)
    
    @property
    def nbytes(self) -> int:
        """Total bytes used."""
        return self.packed_weights.nbytes + self.scales.nbytes


def quantize_ternary(
    weights: np.ndarray,
    group_size: int = 128,
) -> QuantizedTensor:
    """
    Quantize weights to ternary (-1, 0, +1).
    
    Uses absmax quantization per group.
    """
    original_shape = weights.shape
    weights = weights.flatten().astype(np.float32)
    
    # Pad to group size multiple
    padded_size = ((len(weights) + group_size - 1) // group_size) * group_size
    padded = np.zeros(padded_size, dtype=np.float32)
    padded[:len(weights)] = weights
    
    # Reshape to groups
    grouped = padded.reshape(-1, group_size)
    
    # Compute scale per group (absmax)
    scales = np.abs(grouped).max(axis=1, keepdims=True)
    scales = np.where(scales == 0, 1.0, scales)  # Avoid division by zero
    
    # Normalize
    normalized = grouped / scales
    
    # Quantize to ternary
    # Round to nearest of {-1, 0, +1}
    quantized = np.round(normalized).clip(-1, 1).astype(np.int8)
    
    # Pack ternary values (4 per byte)
    packed = _pack_ternary(quantized.flatten())
    
    return QuantizedTensor(
        packed_weights=packed,
        scales=scales.flatten().astype(np.float32),
        shape=original_shape,
        group_size=group_size,
    )


def _pack_ternary(values: np.ndarray) -> np.ndarray:
    """Pack ternary values into bytes (4 values per byte)."""
    # Map -1->0, 0->1, 1->2
    mapped = (values + 1).astype(np.uint8)
    
    # Pad to multiple of 4
    padded_size = ((len(mapped) + 3) // 4) * 4
    padded = np.zeros(padded_size, dtype=np.uint8)
    padded[:len(mapped)] = mapped
    
    # Pack 4 values per byte
    reshaped = padded.reshape(-1, 4)
    packed = (
        reshaped[:, 0]
        | (reshaped[:, 1] << 2)
        | (reshaped[:, 2] << 4)
        | (reshaped[:, 3] << 6)
    )
    
    return packed.astype(np.uint8)


def ternary_matmul(
    x: np.ndarray,
    weight: QuantizedTensor,
) -> np.ndarray:
    """
    Efficient matrix multiplication with ternary weights.
    
    Uses integer arithmetic where possible.
    """
    # Dequantize weight (in production, use optimized kernel)
    w = weight.dequantize()
    
    # Standard matmul
    return x @ w.T
