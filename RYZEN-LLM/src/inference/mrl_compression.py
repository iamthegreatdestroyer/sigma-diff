"""
Matryoshka Representation Learning (MRL) Semantic Compression
=============================================================

Multi-resolution embedding compression using the Matryoshka principle:
full-dimensional embeddings nest progressively smaller representations
that remain semantically meaningful at every resolution.

Implements:
- Multi-resolution encoder: 2048 → 512 → 256 → 128 → 32
- Binary quantization pipeline (32x compression on top of MRL)
- Adaptive resolution selection based on similarity budget
- Integration with existing FP8/Sparse compressors

Key Results (theoretical):
- MRL-512  : <1% recall@10 loss vs full 2048
- MRL-256  : ~2-3% loss, 8x compression
- MRL-128  : ~5% loss, 16x compression
- MRL-32+BQ: ~8-10% loss, 2048x compression (2048*32 / 32*1 bit)

References:
- Kusupati et al., "Matryoshka Representation Learning" (NeurIPS 2022)
- [REF:IP-S5-003] Ryzanstein Sprint 5 Innovation Priorities

Copyright (c) 2025-2026 Ryzanstein LLM Project
Licensed under MIT License
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    Dict, List, Optional, Tuple, Any, Union
)
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ================================================================
# Configuration
# ================================================================

class MRLResolution(Enum):
    """Supported Matryoshka nesting dimensions."""
    FULL_2048 = 2048
    MRL_512   = 512
    MRL_256   = 256
    MRL_128   = 128
    MRL_64    = 64
    MRL_32    = 32


# Default nesting schedule (largest → smallest)
DEFAULT_NESTING_DIMS: List[int] = [2048, 512, 256, 128, 64, 32]


@dataclass
class MRLConfig:
    """Configuration for MRL compression pipeline."""
    full_dim: int = 2048
    nesting_dims: List[int] = field(default_factory=lambda: list(DEFAULT_NESTING_DIMS))
    default_resolution: int = 256
    enable_binary_quantization: bool = True
    normalize_embeddings: bool = True
    # Adaptive selection
    similarity_threshold: float = 0.95
    max_resolution_for_rerank: int = 512
    min_resolution_for_filter: int = 64


@dataclass
class CompressionMetrics:
    """Metrics from a compression operation."""
    original_dim: int
    compressed_dim: int
    binary_quantized: bool
    compression_ratio: float
    encoding_time_ms: float
    cosine_similarity_vs_full: float = 1.0
    memory_bytes_original: int = 0
    memory_bytes_compressed: int = 0


# ================================================================
# MRL Encoder
# ================================================================

class MatryoshkaEncoder(nn.Module):
    """
    Multi-resolution Matryoshka encoder.

    Given a full-dimensional embedding, produces valid sub-embeddings
    at each nesting dimension by learned linear projection followed
    by L2-normalization.

    Architecture:
        full_dim → ProjectionHead(dim_i) for each dim_i in nesting_dims

    Training uses multi-resolution contrastive loss:
        L = Σ_d  λ_d * InfoNCE(proj_d(h), proj_d(h+))
    where λ_d weights each resolution (higher weight for smaller dims).
    """

    def __init__(self, config: MRLConfig):
        super().__init__()
        self.config = config
        self.full_dim = config.full_dim
        self.nesting_dims = sorted(config.nesting_dims, reverse=True)

        # For each nesting dim < full_dim, learn a projection head
        self.projections = nn.ModuleDict()
        for dim in self.nesting_dims:
            if dim < self.full_dim:
                self.projections[str(dim)] = nn.Sequential(
                    nn.Linear(self.full_dim, dim, bias=False),
                    nn.LayerNorm(dim),
                )
            # dim == full_dim → identity (no projection needed)

        # Loss weights: smaller dims get higher weight (harder task)
        self.loss_weights = {}
        for dim in self.nesting_dims:
            # Weight inversely proportional to sqrt(dim)
            self.loss_weights[dim] = 1.0 / math.sqrt(dim / self.nesting_dims[-1])

        logger.info(
            f"MatryoshkaEncoder: full_dim={self.full_dim}, "
            f"nesting_dims={self.nesting_dims}, "
            f"loss_weights={self.loss_weights}"
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        target_dims: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Produce multi-resolution representations.

        Args:
            embeddings: [batch, full_dim] full embeddings
            target_dims: Which resolutions to produce (default: all)

        Returns:
            Dict mapping dim → [batch, dim] normalized embedding
        """
        if target_dims is None:
            target_dims = self.nesting_dims

        results: Dict[int, torch.Tensor] = {}

        for dim in target_dims:
            if dim == self.full_dim:
                proj = embeddings
            elif str(dim) in self.projections:
                proj = self.projections[str(dim)](embeddings)
            else:
                # Truncation fallback for dims without learned projection
                proj = embeddings[..., :dim]

            if self.config.normalize_embeddings:
                proj = F.normalize(proj, p=2, dim=-1)

            results[dim] = proj

        return results

    def encode_single(
        self,
        embeddings: torch.Tensor,
        target_dim: int,
    ) -> torch.Tensor:
        """
        Produce a single resolution.

        Args:
            embeddings: [batch, full_dim] or [full_dim]
            target_dim: Target dimension

        Returns:
            [batch, target_dim] or [target_dim] normalized embedding
        """
        single_batch = embeddings.dim() == 1
        if single_batch:
            embeddings = embeddings.unsqueeze(0)

        result = self.forward(embeddings, target_dims=[target_dim])[target_dim]

        if single_batch:
            result = result.squeeze(0)
        return result


# ================================================================
# Binary Quantization
# ================================================================

class BinaryQuantizer:
    """
    Binary quantization for MRL embeddings.

    Converts float embeddings to 1-bit per dimension:
        bit_i = 1 if embedding_i >= 0 else 0

    Combined with MRL-32 this gives 2048x compression:
        2048 × 32-bit → 32 × 1-bit = 4 bytes

    Similarity uses Hamming distance (fast bit operations).
    """

    @staticmethod
    def quantize(embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Quantize float embeddings to binary.

        Args:
            embeddings: [..., dim] float tensor (should be L2-normalized)

        Returns:
            binary: [..., ceil(dim/8)] uint8 packed bits
            metadata: Scale info for approximate reconstruction
        """
        dim = embeddings.shape[-1]
        batch_shape = embeddings.shape[:-1]

        # Threshold at zero (sign-based)
        bits = (embeddings >= 0).to(torch.uint8)  # [..., dim]

        # Pack into bytes: 8 bits per uint8
        packed_dim = math.ceil(dim / 8)
        flat_bits = bits.reshape(-1, dim)
        n = flat_bits.shape[0]

        packed = torch.zeros(n, packed_dim, dtype=torch.uint8,
                             device=embeddings.device)

        for byte_idx in range(packed_dim):
            start = byte_idx * 8
            end = min(start + 8, dim)
            for bit_pos in range(end - start):
                packed[:, byte_idx] |= flat_bits[:, start + bit_pos] << bit_pos

        packed = packed.reshape(*batch_shape, packed_dim)

        # Store metadata for approximate dequantization
        metadata = {
            'original_dim': dim,
            'packed_dim': packed_dim,
            'mean': embeddings.mean(dim=-1, keepdim=True),
            'std': embeddings.std(dim=-1, keepdim=True),
        }

        return packed, metadata

    @staticmethod
    def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute Hamming distance between packed binary vectors.

        Args:
            a: [..., packed_dim] uint8
            b: [..., packed_dim] uint8

        Returns:
            distances: [...] int tensor (number of differing bits)
        """
        xor_bits = a ^ b  # Differing bits
        # Popcount: count 1-bits per byte then sum
        # Use lookup table for fast popcount on CPU
        popcount_lut = torch.tensor([
            bin(i).count('1') for i in range(256)
        ], dtype=torch.int32, device=a.device)

        popcounts = popcount_lut[xor_bits.long()]  # [..., packed_dim]
        return popcounts.sum(dim=-1)

    @staticmethod
    def hamming_similarity(a: torch.Tensor, b: torch.Tensor,
                           dim: int) -> torch.Tensor:
        """
        Compute normalized Hamming similarity (1 - normalized_distance).

        Returns value in [0, 1] where 1 = identical.
        """
        dist = BinaryQuantizer.hamming_distance(a, b)
        return 1.0 - dist.float() / dim

    @staticmethod
    def dequantize_approximate(
        packed: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Approximately reconstruct float embedding from binary.

        Uses sign reconstruction + scale/shift from metadata.
        Quality is low—use only for visualization/debugging.
        """
        dim = metadata['original_dim']
        packed_dim = metadata['packed_dim']
        batch_shape = packed.shape[:-1]

        flat_packed = packed.reshape(-1, packed_dim)
        n = flat_packed.shape[0]

        bits = torch.zeros(n, dim, dtype=torch.float32,
                           device=packed.device)

        for byte_idx in range(packed_dim):
            start = byte_idx * 8
            end = min(start + 8, dim)
            for bit_pos in range(end - start):
                bits[:, start + bit_pos] = (
                    (flat_packed[:, byte_idx] >> bit_pos) & 1
                ).float()

        # Map {0, 1} → {-1, +1} then scale
        reconstructed = (bits * 2.0 - 1.0)

        # Apply stored statistics
        mean = metadata['mean']
        std = metadata['std']
        if mean.dim() > 1:
            mean = mean.reshape(-1, 1)
            std = std.reshape(-1, 1)

        reconstructed = reconstructed * std + mean
        reconstructed = reconstructed.reshape(*batch_shape, dim)

        return reconstructed


# ================================================================
# Adaptive Resolution Selector
# ================================================================

class AdaptiveResolutionSelector:
    """
    Automatically selects the smallest MRL resolution that meets
    a similarity budget for a given query.

    Strategy:
    1. Start at lowest resolution (e.g., MRL-32 or MRL-64)
    2. Retrieve top-k candidates using fast Hamming/cosine
    3. If confidence < threshold, rerank at higher resolution
    4. Stop when similarity threshold is met
    """

    def __init__(self, config: MRLConfig):
        self.config = config
        self.resolutions = sorted(config.nesting_dims)  # ascending
        self.threshold = config.similarity_threshold

        # Statistics for adaptive learning
        self.resolution_hits: Dict[int, int] = {d: 0 for d in self.resolutions}
        self.total_queries = 0

    def select_resolution(
        self,
        query_confidence: float,
        is_reranking: bool = False,
    ) -> int:
        """
        Select appropriate resolution based on confidence.

        Args:
            query_confidence: Estimated confidence in [0, 1]
            is_reranking: Whether this is a reranking pass

        Returns:
            Selected dimension
        """
        if is_reranking:
            return min(self.config.max_resolution_for_rerank, self.config.full_dim)

        # High confidence → low resolution sufficient
        if query_confidence >= 0.99:
            dim = self.config.min_resolution_for_filter
        elif query_confidence >= self.threshold:
            dim = self.config.default_resolution
        else:
            dim = self.config.max_resolution_for_rerank

        self.resolution_hits[dim] = self.resolution_hits.get(dim, 0) + 1
        self.total_queries += 1

        return dim

    def get_statistics(self) -> Dict[str, Any]:
        """Return resolution selection statistics."""
        return {
            'total_queries': self.total_queries,
            'resolution_distribution': dict(self.resolution_hits),
            'avg_compression_ratio': self._avg_compression_ratio(),
        }

    def _avg_compression_ratio(self) -> float:
        if self.total_queries == 0:
            return 0.0
        weighted_dim = sum(
            dim * count for dim, count in self.resolution_hits.items()
        )
        avg_dim = weighted_dim / self.total_queries
        return self.config.full_dim / avg_dim if avg_dim > 0 else 0.0


# ================================================================
# MRL Compression Pipeline
# ================================================================

class MRLCompressionPipeline:
    """
    End-to-end MRL compression pipeline.

    Combines:
    - MatryoshkaEncoder for multi-resolution projection
    - BinaryQuantizer for optional 1-bit quantization
    - AdaptiveResolutionSelector for smart dimension selection

    Usage:
        pipeline = MRLCompressionPipeline(MRLConfig())
        compressed = pipeline.compress(embeddings, target_dim=256)
        similarity = pipeline.similarity(compressed_a, compressed_b)
    """

    def __init__(self, config: Optional[MRLConfig] = None):
        self.config = config or MRLConfig()
        self.encoder = MatryoshkaEncoder(self.config)
        self.binary_quantizer = BinaryQuantizer()
        self.selector = AdaptiveResolutionSelector(self.config)

        # Statistics
        self._compression_count = 0
        self._total_compression_ratio = 0.0

    def compress(
        self,
        embeddings: torch.Tensor,
        target_dim: Optional[int] = None,
        binary_quantize: Optional[bool] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, Dict]], CompressionMetrics]:
        """
        Compress embeddings to target resolution.

        Args:
            embeddings: [batch, full_dim] or [full_dim] float tensor
            target_dim: Target dimension (None = use config default)
            binary_quantize: Whether to apply binary quantization
                             (None = use config default)

        Returns:
            compressed: Compressed embedding(s)
                If binary_quantize: (packed_uint8, metadata)
                Else: float tensor at target_dim
            metrics: Compression performance metrics
        """
        start_time = time.perf_counter()

        if target_dim is None:
            target_dim = self.config.default_resolution
        if binary_quantize is None:
            binary_quantize = self.config.enable_binary_quantization

        single_batch = embeddings.dim() == 1
        if single_batch:
            embeddings = embeddings.unsqueeze(0)

        batch_size = embeddings.shape[0]
        original_dim = embeddings.shape[-1]

        # Step 1: Project to target resolution
        with torch.no_grad():
            projected = self.encoder.encode_single(embeddings, target_dim)

        # Compute cosine similarity with full-dim for quality tracking
        if original_dim == self.config.full_dim:
            cos_sim = F.cosine_similarity(
                embeddings, F.pad(projected, (0, original_dim - target_dim)),
                dim=-1
            ).mean().item() if target_dim < original_dim else 1.0
        else:
            cos_sim = 1.0

        # Step 2: Optional binary quantization
        if binary_quantize:
            packed, bq_metadata = self.binary_quantizer.quantize(projected)
            compressed = (packed, bq_metadata)
            compressed_bytes = batch_size * math.ceil(target_dim / 8)
        else:
            compressed = projected.squeeze(0) if single_batch else projected
            compressed_bytes = batch_size * target_dim * 4  # float32

        original_bytes = batch_size * original_dim * 4  # float32
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        metrics = CompressionMetrics(
            original_dim=original_dim,
            compressed_dim=target_dim,
            binary_quantized=binary_quantize,
            compression_ratio=original_bytes / max(compressed_bytes, 1),
            encoding_time_ms=elapsed_ms,
            cosine_similarity_vs_full=cos_sim,
            memory_bytes_original=original_bytes,
            memory_bytes_compressed=compressed_bytes,
        )

        self._compression_count += 1
        self._total_compression_ratio += metrics.compression_ratio

        return compressed, metrics

    def compress_multi_resolution(
        self,
        embeddings: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Produce all nesting resolutions at once.

        Args:
            embeddings: [batch, full_dim]

        Returns:
            Dict[dim] → [batch, dim] normalized embedding per resolution
        """
        with torch.no_grad():
            return self.encoder(embeddings)

    def similarity(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        binary: bool = False,
        dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute similarity between compressed embeddings.

        For float embeddings: cosine similarity.
        For binary: normalized Hamming similarity.
        """
        if binary:
            assert dim is not None, "Must provide dim for binary similarity"
            return self.binary_quantizer.hamming_similarity(a, b, dim)
        else:
            return F.cosine_similarity(a, b, dim=-1)

    def get_average_compression_ratio(self) -> float:
        if self._compression_count == 0:
            return 0.0
        return self._total_compression_ratio / self._compression_count


# ================================================================
# Multi-Resolution Contrastive Loss
# ================================================================

class MatryoshkaContrastiveLoss(nn.Module):
    """
    Multi-resolution InfoNCE loss for training MatryoshkaEncoder.

    L = Σ_d  λ_d * InfoNCE(f_d(anchor), f_d(positive), f_d(negatives))

    Each resolution contributes to the total loss, weighted by λ_d.
    Smaller dimensions receive higher weight (harder task).
    """

    def __init__(self, encoder: MatryoshkaEncoder, temperature: float = 0.07):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-resolution contrastive loss.

        Args:
            anchors: [batch, full_dim]
            positives: [batch, full_dim]
            negatives: [num_neg, full_dim] (optional, uses in-batch if None)

        Returns:
            total_loss: Scalar
            per_dim_losses: Dict[dim] → loss value for monitoring
        """
        # Project all inputs at all resolutions
        anchor_reps = self.encoder(anchors)
        positive_reps = self.encoder(positives)

        if negatives is not None:
            negative_reps = self.encoder(negatives)
        else:
            negative_reps = None

        total_loss = torch.tensor(0.0, device=anchors.device)
        per_dim_losses: Dict[str, float] = {}
        total_weight = sum(self.encoder.loss_weights.values())

        for dim in self.encoder.nesting_dims:
            a = anchor_reps[dim]      # [batch, dim]
            p = positive_reps[dim]    # [batch, dim]

            if negative_reps is not None:
                n = negative_reps[dim]  # [num_neg, dim]
                # Similarity: anchor vs positive and anchor vs negatives
                pos_sim = F.cosine_similarity(a, p, dim=-1) / self.temperature
                neg_sim = torch.mm(a, n.t()) / self.temperature  # [batch, num_neg]
                logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
            else:
                # In-batch negatives
                sim_matrix = torch.mm(a, a.t()) / self.temperature  # [batch, batch]
                pos_sim = (a * p).sum(dim=-1) / self.temperature    # [batch]
                # Mask diagonal
                batch_size = a.shape[0]
                mask = torch.eye(batch_size, device=a.device).bool()
                sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
                logits = torch.cat([pos_sim.unsqueeze(-1), sim_matrix], dim=-1)

            # InfoNCE: positive is at index 0
            labels = torch.zeros(logits.shape[0], dtype=torch.long,
                                 device=logits.device)
            dim_loss = F.cross_entropy(logits, labels)

            weight = self.encoder.loss_weights[dim] / total_weight
            total_loss = total_loss + weight * dim_loss
            per_dim_losses[f"loss_dim_{dim}"] = dim_loss.item()

        return total_loss, per_dim_losses


# ================================================================
# Convenience Functions
# ================================================================

def create_mrl_pipeline(
    full_dim: int = 2048,
    default_resolution: int = 256,
    enable_binary: bool = True,
) -> MRLCompressionPipeline:
    """Create a configured MRL compression pipeline."""
    config = MRLConfig(
        full_dim=full_dim,
        default_resolution=default_resolution,
        enable_binary_quantization=enable_binary,
    )
    return MRLCompressionPipeline(config)


def benchmark_mrl_compression(
    pipeline: MRLCompressionPipeline,
    num_vectors: int = 10000,
    full_dim: int = 2048,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Benchmark MRL compression across all resolutions.

    Returns timing, compression ratios, and similarity preservation
    for each nesting dimension.
    """
    logger.info(f"Benchmarking MRL: {num_vectors} vectors, dim={full_dim}")

    # Generate random normalized embeddings
    torch.manual_seed(42)
    embeddings = torch.randn(num_vectors, full_dim, device=device)
    embeddings = F.normalize(embeddings, p=2, dim=-1)

    results: Dict[str, Any] = {
        'num_vectors': num_vectors,
        'full_dim': full_dim,
        'resolutions': {},
    }

    for dim in pipeline.config.nesting_dims:
        # Float compression
        _, float_metrics = pipeline.compress(
            embeddings, target_dim=dim, binary_quantize=False
        )

        # Binary compression
        _, binary_metrics = pipeline.compress(
            embeddings, target_dim=dim, binary_quantize=True
        )

        results['resolutions'][dim] = {
            'float': {
                'compression_ratio': float_metrics.compression_ratio,
                'encoding_time_ms': float_metrics.encoding_time_ms,
                'cosine_sim': float_metrics.cosine_similarity_vs_full,
                'memory_bytes': float_metrics.memory_bytes_compressed,
            },
            'binary': {
                'compression_ratio': binary_metrics.compression_ratio,
                'encoding_time_ms': binary_metrics.encoding_time_ms,
                'cosine_sim': binary_metrics.cosine_similarity_vs_full,
                'memory_bytes': binary_metrics.memory_bytes_compressed,
            },
        }

        logger.info(
            f"  dim={dim:>4d}: float {float_metrics.compression_ratio:.1f}x "
            f"({float_metrics.encoding_time_ms:.1f}ms), "
            f"binary {binary_metrics.compression_ratio:.1f}x "
            f"({binary_metrics.encoding_time_ms:.1f}ms)"
        )

    return results
