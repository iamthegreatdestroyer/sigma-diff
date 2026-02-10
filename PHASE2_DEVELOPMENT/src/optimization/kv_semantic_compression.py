"""
KV Cache Semantic Compression Module
Sprint 4.4 - Task 1: Semantic Compression

Reduces KV cache memory footprint through intelligent compression techniques:
- Low-rank approximation (SVD)
- Quantization (int8/int4)
- Token clustering
- Semantic pruning

Performance Target: >80% compression ratio with <5% accuracy loss
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import heapq
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums & Data Classes
# =============================================================================

class CompressionMethod(Enum):
    """Compression methods for KV cache."""
    NONE = auto()
    LOW_RANK = auto()        # SVD-based low-rank approximation
    QUANTIZATION = auto()    # int8/int4 quantization
    CLUSTERING = auto()      # Token clustering and grouping
    HYBRID = auto()          # Combination of methods


class QuantizationType(Enum):
    """Quantization precision levels."""
    FP32 = 4  # No compression
    FP16 = 2  # Half precision
    INT8 = 1  # 8-bit integer
    INT4 = 0.5  # 4-bit integer


@dataclass
class CompressionConfig:
    """Configuration for semantic compression."""
    method: CompressionMethod = CompressionMethod.HYBRID
    rank_fraction: float = 0.5  # 50% rank reduction via SVD
    quantization_type: QuantizationType = QuantizationType.INT8
    cluster_threshold: float = 0.95  # Similarity threshold for clustering
    pruning_threshold: float = 0.1  # Prune tokens with <10% importance
    enable_adaptive: bool = True  # Adapt compression based on usage


@dataclass
class CompressionStats:
    """Statistics about compression effectiveness."""
    original_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0
    accuracy_loss_percent: float = 0.0
    
    @property
    def memory_saved_percent(self) -> float:
        if self.original_size == 0:
            return 0.0
        return (1.0 - self.compression_ratio) * 100


# =============================================================================
# Compression Algorithms
# =============================================================================

class CompressionAlgorithm(ABC):
    """Base class for compression algorithms."""
    
    @abstractmethod
    def compress(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Compress data, return compressed data and metadata."""
        pass
    
    @abstractmethod
    def decompress(self, compressed: np.ndarray, metadata: Dict) -> np.ndarray:
        """Decompress data using metadata."""
        pass


class LowRankCompression(CompressionAlgorithm):
    """
    Low-rank approximation using SVD.
    
    K_compressed = U @ S[:r] @ V[:r].T  (where r = rank_fraction * original_rank)
    """
    
    def __init__(self, rank_fraction: float = 0.5):
        self.rank_fraction = rank_fraction
    
    def compress(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Compress using SVD low-rank approximation."""
        # data shape: (seq_len, hidden_dim)
        
        # Perform SVD
        U, S, Vt = np.linalg.svd(data, full_matrices=False)
        
        # Compute target rank
        target_rank = max(1, int(len(S) * self.rank_fraction))
        
        # Truncate to target rank
        U_r = U[:, :target_rank]
        S_r = S[:target_rank]
        Vt_r = Vt[:target_rank, :]
        
        # Store compressed representation
        metadata = {
            "method": "low_rank",
            "original_shape": data.shape,
            "target_rank": target_rank,
            "full_rank": len(S),
        }
        
        return U_r, S_r, Vt_r, metadata
    
    def decompress(self, U_r: np.ndarray, S_r: np.ndarray, 
                   Vt_r: np.ndarray, metadata: Dict) -> np.ndarray:
        """Reconstruct from low-rank approximation."""
        # Reconstruct: U_r @ S_r @ Vt_r (with S as diagonal matrix)
        S_diag = np.diag(S_r)
        reconstructed = U_r @ S_diag @ Vt_r
        return reconstructed


class QuantizationCompression(CompressionAlgorithm):
    """
    Quantization compression to reduce precision.
    
    Supports: FP16, INT8, INT4
    """
    
    def __init__(self, quant_type: QuantizationType = QuantizationType.INT8):
        self.quant_type = quant_type
    
    def compress(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Quantize data to lower precision."""
        
        if self.quant_type == QuantizationType.FP32:
            # No compression
            return data.astype(np.float32), {"type": "fp32"}
        
        elif self.quant_type == QuantizationType.FP16:
            # Convert to float16
            quantized = data.astype(np.float16)
            return quantized, {
                "type": "fp16",
                "original_shape": data.shape,
                "original_dtype": str(data.dtype),
            }
        
        elif self.quant_type == QuantizationType.INT8:
            # INT8 quantization
            # Map [min, max] to [-128, 127]
            min_val = data.min()
            max_val = data.max()
            
            # Avoid division by zero
            scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
            
            quantized = ((data - min_val) / scale - 128).astype(np.int8)
            
            return quantized, {
                "type": "int8",
                "original_shape": data.shape,
                "min_val": float(min_val),
                "max_val": float(max_val),
                "scale": float(scale),
            }
        
        elif self.quant_type == QuantizationType.INT4:
            # INT4 quantization (2 values per byte)
            # Map [min, max] to [0, 15]
            min_val = data.min()
            max_val = data.max()
            
            scale = (max_val - min_val) / 15.0 if max_val > min_val else 1.0
            normalized = (data - min_val) / scale
            quantized = np.round(normalized).astype(np.uint8)
            
            # Pack 2 values per byte
            flat = quantized.flatten()
            if len(flat) % 2 != 0:
                flat = np.append(flat, 0)
            
            packed = np.zeros(len(flat) // 2, dtype=np.uint8)
            for i in range(0, len(flat), 2):
                packed[i // 2] = (flat[i] << 4) | (flat[i+1] & 0x0F)
            
            return packed, {
                "type": "int4",
                "original_shape": data.shape,
                "min_val": float(min_val),
                "max_val": float(max_val),
                "scale": float(scale),
            }
    
    def decompress(self, compressed: np.ndarray, metadata: Dict) -> np.ndarray:
        """Dequantize to original precision."""
        
        if metadata["type"] == "fp32":
            return compressed
        
        elif metadata["type"] == "fp16":
            return compressed.astype(np.float32)
        
        elif metadata["type"] == "int8":
            # Dequantize INT8
            scale = metadata["scale"]
            min_val = metadata["min_val"]
            dequantized = (compressed.astype(np.float32) + 128) * scale + min_val
            return dequantized
        
        elif metadata["type"] == "int4":
            # Unpack INT4
            packed = compressed
            original_shape = metadata["original_shape"]
            total_elements = np.prod(original_shape)
            
            unpacked = np.zeros(total_elements, dtype=np.uint8)
            for i in range(len(packed)):
                unpacked[2*i] = (packed[i] >> 4) & 0x0F
                unpacked[2*i+1] = packed[i] & 0x0F
            
            # Dequantize
            scale = metadata["scale"]
            min_val = metadata["min_val"]
            dequantized = unpacked[:total_elements].astype(np.float32) * scale + min_val
            
            return dequantized.reshape(original_shape)


class ClusteringCompression(CompressionAlgorithm):
    """
    Token clustering compression.
    
    Groups similar tokens and stores representative clusters.
    """
    
    def __init__(self, num_clusters: int = 256):
        self.num_clusters = num_clusters
    
    def compress(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Cluster tokens and store assignments."""
        # data shape: (seq_len, hidden_dim)
        
        # Simple k-means clustering
        # Initialize cluster centers randomly from data
        indices = np.random.choice(len(data), self.num_clusters, replace=False)
        centers = data[indices].copy()
        
        # Run k-means (simplified: 1 iteration)
        distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)
        
        # Update centers
        for i in range(self.num_clusters):
            mask = assignments == i
            if mask.sum() > 0:
                centers[i] = data[mask].mean(axis=0)
        
        metadata = {
            "method": "clustering",
            "original_shape": data.shape,
            "num_clusters": self.num_clusters,
        }
        
        return centers, assignments, metadata
    
    def decompress(self, centers: np.ndarray, assignments: np.ndarray, 
                   metadata: Dict) -> np.ndarray:
        """Reconstruct from cluster assignments."""
        reconstructed = centers[assignments]
        return reconstructed.reshape(metadata["original_shape"])


# =============================================================================
# Adaptive Compression Engine
# =============================================================================

class AdaptiveCompressionEngine:
    """
    Dynamically selects and applies compression methods.
    
    Measures accuracy impact and adapts compression parameters.
    """
    
    def __init__(self, config: CompressionConfig = CompressionConfig()):
        self.config = config
        self.stats_history: List[CompressionStats] = []
        self.compression_log: Dict[str, int] = defaultdict(int)
    
    def compress_batch(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        original_keys: Optional[np.ndarray] = None
    ) -> Tuple[Dict, CompressionStats]:
        """
        Compress KV cache batch intelligently.
        
        Args:
            keys: Key cache (seq_len, hidden_dim)
            values: Value cache (seq_len, hidden_dim)
            original_keys: Original uncompressed keys for accuracy measurement
        
        Returns:
            (compressed_data_dict, stats)
        """
        import time
        start = time.time()
        
        stats = CompressionStats(
            original_size=keys.nbytes + values.nbytes
        )
        
        compressed_data = {}
        
        # Apply selected compression method
        if self.config.method == CompressionMethod.LOW_RANK:
            # Low-rank compression
            lr = LowRankCompression(self.config.rank_fraction)
            U_k, S_k, Vt_k, meta_k = lr.compress(keys)
            U_v, S_v, Vt_v, meta_v = lr.compress(values)
            
            compressed_data = {
                "keys": (U_k, S_k, Vt_k, meta_k),
                "values": (U_v, S_v, Vt_v, meta_v),
                "method": "low_rank"
            }
        
        elif self.config.method == CompressionMethod.QUANTIZATION:
            # Quantization
            quant = QuantizationCompression(self.config.quantization_type)
            q_keys, meta_k = quant.compress(keys)
            q_values, meta_v = quant.compress(values)
            
            compressed_data = {
                "keys": (q_keys, meta_k),
                "values": (q_values, meta_v),
                "method": "quantization"
            }
        
        elif self.config.method == CompressionMethod.HYBRID:
            # Hybrid: Quantization + Low-rank
            quant = QuantizationCompression(self.config.quantization_type)
            q_keys, meta_k = quant.compress(keys)
            q_values, meta_v = quant.compress(values)
            
            # Then apply low-rank to quantized values
            lr = LowRankCompression(self.config.rank_fraction)
            keys_q = q_keys.astype(np.float32)
            U_k, S_k, Vt_k, meta_lr_k = lr.compress(keys_q)
            
            compressed_data = {
                "keys": (U_k, S_k, Vt_k, meta_lr_k, meta_k),
                "values": (q_values, meta_v),
                "method": "hybrid"
            }
        
        # Calculate compression metrics
        compressed_size = sum(
            v[0].nbytes if isinstance(v, tuple) else v.nbytes
            for v in compressed_data.values() if v != "hybrid" and v != "low_rank" and v != "quantization"
        )
        
        stats.compressed_size = compressed_size
        stats.compression_ratio = compressed_size / stats.original_size if stats.original_size > 0 else 1.0
        stats.compression_time_ms = (time.time() - start) * 1000
        
        # Measure accuracy if original provided
        if original_keys is not None:
            reconstructed = self.decompress(compressed_data)
            accuracy_loss = np.mean(np.abs(reconstructed["keys"] - original_keys))
            stats.accuracy_loss_percent = accuracy_loss * 100
        
        self.stats_history.append(stats)
        self.compression_log[self.config.method.name] += 1
        
        return compressed_data, stats
    
    def decompress(self, compressed_data: Dict) -> Dict:
        """Decompress KV cache."""
        method = compressed_data.get("method", "hybrid")
        
        if method == "low_rank":
            lr = LowRankCompression()
            U_k, S_k, Vt_k, meta_k = compressed_data["keys"]
            keys = lr.decompress(U_k, S_k, Vt_k, meta_k)
            
            U_v, S_v, Vt_v, meta_v = compressed_data["values"]
            values = lr.decompress(U_v, S_v, Vt_v, meta_v)
        
        elif method == "quantization":
            quant = QuantizationCompression()
            q_keys, meta_k = compressed_data["keys"]
            keys = quant.decompress(q_keys, meta_k)
            
            q_values, meta_v = compressed_data["values"]
            values = quant.decompress(q_values, meta_v)
        
        elif method == "hybrid":
            # Decompress hybrid
            quant = QuantizationCompression()
            q_keys, meta_k = compressed_data["values"]
            keys = quant.decompress(q_keys, meta_k)
            
            values_data = compressed_data["values"]
            q_values, meta_v = values_data
            values = quant.decompress(q_values, meta_v)
        
        else:
            keys = compressed_data["keys"]
            values = compressed_data["values"]
        
        return {"keys": keys, "values": values}
    
    def get_stats(self) -> CompressionStats:
        """Get aggregated statistics."""
        if not self.stats_history:
            return CompressionStats()
        
        total_original = sum(s.original_size for s in self.stats_history)
        total_compressed = sum(s.compressed_size for s in self.stats_history)
        avg_time = np.mean([s.compression_time_ms for s in self.stats_history])
        avg_accuracy_loss = np.mean([s.accuracy_loss_percent for s in self.stats_history])
        
        stats = CompressionStats(
            original_size=total_original,
            compressed_size=total_compressed,
            compression_ratio=total_compressed / total_original if total_original > 0 else 1.0,
            compression_time_ms=avg_time,
            accuracy_loss_percent=avg_accuracy_loss,
        )
        
        return stats


# =============================================================================
# Convenience Functions
# =============================================================================

def create_compression_engine(
    method: str = "hybrid",
    rank_fraction: float = 0.5,
    quant_type: str = "int8"
) -> AdaptiveCompressionEngine:
    """
    Create a compression engine with specified configuration.
    
    Args:
        method: "none", "low_rank", "quantization", "hybrid"
        rank_fraction: Rank reduction fraction (0-1)
        quant_type: "fp32", "fp16", "int8", "int4"
    
    Returns:
        AdaptiveCompressionEngine
    """
    method_map = {
        "none": CompressionMethod.NONE,
        "low_rank": CompressionMethod.LOW_RANK,
        "quantization": CompressionMethod.QUANTIZATION,
        "hybrid": CompressionMethod.HYBRID,
    }
    
    quant_map = {
        "fp32": QuantizationType.FP32,
        "fp16": QuantizationType.FP16,
        "int8": QuantizationType.INT8,
        "int4": QuantizationType.INT4,
    }
    
    config = CompressionConfig(
        method=method_map.get(method, CompressionMethod.HYBRID),
        rank_fraction=rank_fraction,
        quantization_type=quant_map.get(quant_type, QuantizationType.INT8),
    )
    
    return AdaptiveCompressionEngine(config)
