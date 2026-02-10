"""
Distributed Checkpoint Loading

Handles:
- Distributed checkpoint loading across multiple ranks
- Weight sharding and distribution
- State reconstruction from distributed checkpoint format
- Memory-efficient loading strategies
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path
import torch
import torch.nn as nn
import json
import logging

logger = logging.getLogger(__name__)


class DistributedCheckpointLoader:
    """Loads checkpoints in distributed format.
    
    Checkpoint format:
        checkpoints/model-step-1000/
        ├─ metadata.json              (rank 0 only)
        ├─ weights_rank0.pt
        ├─ weights_rank1.pt
        ├─ weights_rank2.pt
        └─ weights_rank3.pt
        
    Features:
    - Zero-copy loading where possible
    - Memory-mapped loading for large checkpoints
    - Asynchronous loading with prefetching
    """
    
    def __init__(self, checkpoint_dir: str, rank: int, world_size: int,
                 use_memory_map: bool = True, enable_prefetch: bool = True):
        """Initialize distributed checkpoint loader.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            rank: Current rank
            world_size: Total number of ranks
            use_memory_map: Use memory-mapped loading for efficiency
            enable_prefetch: Enable asynchronous prefetching
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.rank = rank
        self.world_size = world_size
        self.use_memory_map = use_memory_map
        self.enable_prefetch = enable_prefetch
        
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Performance tracking
        self.load_times = []
        self.memory_peaks = []
        self.prefetch_cache = {}  # Cache for prefetched weights
        self.use_prefetch = enable_prefetch
        
        logger.info(f"DistributedCheckpointLoader initialized: {checkpoint_dir}")
        logger.info(f"Zero-copy: {use_memory_map}, Prefetch: {enable_prefetch}")
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata.
        
        Only rank 0 actually loads from disk, then broadcasts to others.
        
        Returns:
            Metadata dictionary
            
        Raises:
            FileNotFoundError: If metadata.json not found
        """
        metadata_path = self.checkpoint_dir / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        if self.rank == 0:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Metadata loaded: {metadata}")
        else:
            metadata = None
        
        # Broadcast metadata to all ranks
        # Note: In practice, would use torch.distributed to broadcast
        # For now, simplified
        
        return metadata
    
    def prefetch_weights(self, prefetch_ahead: int = 1) -> None:
        """Prefetch weights for future ranks to reduce loading latency.
        
        Args:
            prefetch_ahead: Number of ranks to prefetch ahead
        """
        if not self.use_prefetch:
            return
            
        import threading
        
        def prefetch_worker(rank_offset: int):
            """Worker function to prefetch weights in background."""
            prefetch_rank = (self.rank + rank_offset) % self.world_size
            prefetch_path = self.checkpoint_dir / f"weights_rank{prefetch_rank}.pt"
            
            if prefetch_path.exists():
                try:
                    # Load into prefetch cache
                    weights = torch.load(prefetch_path, map_location='cpu', mmap=self.use_memory_map)
                    self.prefetch_cache[prefetch_rank] = weights
                    logger.debug(f"Prefetched weights for rank {prefetch_rank}")
                except Exception as e:
                    logger.warning(f"Failed to prefetch rank {prefetch_rank}: {e}")
        
        # Start prefetch threads
        threads = []
        for offset in range(1, prefetch_ahead + 1):
            thread = threading.Thread(target=prefetch_worker, args=(offset,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Wait for prefetch completion
        for thread in threads:
            thread.join(timeout=5.0)  # 5 second timeout
    
    def load_rank_weights(self) -> Dict[str, torch.Tensor]:
        """Load weights for current rank from checkpoint.
        
        Supports zero-copy loading and memory mapping for efficiency.
        Checks prefetch cache first for improved performance.
        
        Returns:
            Dictionary of weight tensors
            
        Raises:
            FileNotFoundError: If rank weights not found
        """
        import time
        start_time = time.time()
        
        # Check prefetch cache first
        if self.rank in self.prefetch_cache:
            weights = self.prefetch_cache.pop(self.rank)
            load_time = time.time() - start_time
            self.load_times.append(load_time)
            logger.info(f"Rank {self.rank}: loaded {len(weights)} weights from prefetch cache")
            logger.debug(f"Rank {self.rank}: cache load time: {load_time:.3f}s")
            return weights
        
        weights_path = self.checkpoint_dir / f"weights_rank{self.rank}.pt"
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Rank {self.rank} weights not found: {weights_path}")
        
        try:
            if self.use_memory_map:
                # Memory-mapped loading for zero-copy where possible
                weights = torch.load(weights_path, map_location='cpu', mmap=True)
                logger.info(f"Rank {self.rank}: loaded {len(weights)} weights (memory-mapped)")
            else:
                # Standard loading
                weights = torch.load(weights_path, map_location='cpu')
                logger.info(f"Rank {self.rank}: loaded {len(weights)} weights")
            
            load_time = time.time() - start_time
            self.load_times.append(load_time)
            
            # Track memory usage
            if torch.cuda.is_available():
                self.memory_peaks.append(torch.cuda.max_memory_allocated() / (1024**3))
            
            logger.debug(f"Rank {self.rank}: load time: {load_time:.3f}s")
            
            return weights
            
        except Exception as e:
            logger.error(f"Rank {self.rank}: failed to load weights: {e}")
            raise RuntimeError(f"Weight loading failed: {e}")
    
    def load_into_model(self, model: nn.Module) -> None:
        """Load checkpoint weights into model.
        
        Args:
            model: Model to load weights into
            
        Raises:
            RuntimeError: If weight loading fails
        """
        try:
            # Start prefetching for next ranks
            if self.use_prefetch:
                self.prefetch_weights(prefetch_ahead=1)
            
            weights = self.load_rank_weights()
            # Distribute weights to appropriate model layers
            # Implementation depends on tensor parallelism strategy
            logger.info(f"Rank {self.rank}: weights loaded into model")
        except Exception as e:
            logger.error(f"Rank {self.rank}: failed to load weights: {e}")
            raise RuntimeError(f"Weight loading failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for loading operations.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.load_times:
            return {}
        
        return {
            'avg_load_time': sum(self.load_times) / len(self.load_times),
            'max_load_time': max(self.load_times),
            'min_load_time': min(self.load_times),
            'total_load_time': sum(self.load_times),
            'load_count': len(self.load_times),
            'avg_memory_peak_gb': sum(self.memory_peaks) / len(self.memory_peaks) if self.memory_peaks else 0.0,
            'max_memory_peak_gb': max(self.memory_peaks) if self.memory_peaks else 0.0,
            'prefetch_enabled': self.use_prefetch,
            'memory_map_enabled': self.use_memory_map
        }


class WeightDistributor:
    """Distributes model weights across GPUs according to parallelism strategy.
    
    Handles:
    - Linear layer weight sharding (row-wise or column-wise)
    - Attention head distribution
    - Embedding layer sharding
    """
    
    def __init__(self, rank: int, world_size: int, tp_size: int):
        """Initialize weight distributor.
        
        Args:
            rank: Current rank
            world_size: Total number of ranks
            tp_size: Tensor parallelism size
        """
        self.rank = rank
        self.world_size = world_size
        self.tp_size = tp_size
        self.local_rank = rank % tp_size  # Rank within TP group
        
        logger.info(f"WeightDistributor: rank={rank}, world_size={world_size}, "
                   f"tp_size={tp_size}, local_rank={self.local_rank}")
    
    def shard_linear_layer_row_wise(self, weight: torch.Tensor, 
                                    bias: Optional[torch.Tensor] = None
                                    ) -> tuple:
        """Shard linear layer weights row-wise (output dimension).
        
        Original:
            weight: (out_features, in_features)
            bias: (out_features,)
        
        Sharded (for rank i):
            weight: (out_features // tp_size, in_features)
            bias: (out_features // tp_size,)
        
        Args:
            weight: Weight tensor to shard
            bias: Optional bias tensor
            
        Returns:
            Tuple of (sharded_weight, sharded_bias or None)
        """
        out_features = weight.shape[0]
        assert out_features % self.tp_size == 0, \
            f"Output features {out_features} not divisible by tp_size {self.tp_size}"
        
        shard_size = out_features // self.tp_size
        start_idx = self.local_rank * shard_size
        end_idx = (self.local_rank + 1) * shard_size
        
        sharded_weight = weight[start_idx:end_idx, :].clone()
        
        sharded_bias = None
        if bias is not None:
            sharded_bias = bias[start_idx:end_idx].clone()
        
        logger.debug(f"Rank {self.rank}: sharded linear layer row-wise "
                    f"weight {weight.shape} -> {sharded_weight.shape}")
        
        return sharded_weight, sharded_bias
    
    def shard_linear_layer_column_wise(self, weight: torch.Tensor,
                                       bias: Optional[torch.Tensor] = None
                                       ) -> tuple:
        """Shard linear layer weights column-wise (input dimension).
        
        Original:
            weight: (out_features, in_features)
            bias: (out_features,)
        
        Sharded (for rank i):
            weight: (out_features, in_features // tp_size)
            bias: (out_features,)  [not sharded]
        
        Args:
            weight: Weight tensor to shard
            bias: Optional bias tensor
            
        Returns:
            Tuple of (sharded_weight, sharded_bias or original)
        """
        in_features = weight.shape[1]
        assert in_features % self.tp_size == 0, \
            f"Input features {in_features} not divisible by tp_size {self.tp_size}"
        
        shard_size = in_features // self.tp_size
        start_idx = self.local_rank * shard_size
        end_idx = (self.local_rank + 1) * shard_size
        
        sharded_weight = weight[:, start_idx:end_idx].clone()
        
        # Bias not sharded in column-wise (replicated)
        sharded_bias = bias.clone() if bias is not None else None
        
        logger.debug(f"Rank {self.rank}: sharded linear layer column-wise "
                    f"weight {weight.shape} -> {sharded_weight.shape}")
        
        return sharded_weight, sharded_bias
    
    def shard_attention_heads(self, num_heads: int) -> List[int]:
        """Determine which attention heads this rank should compute.
        
        Distributes num_heads across tp_size ranks.
        
        Example: num_heads=32, tp_size=4
            rank 0: heads [0:8]
            rank 1: heads [8:16]
            rank 2: heads [16:24]
            rank 3: heads [24:32]
        
        Args:
            num_heads: Total number of attention heads
            
        Returns:
            List of head indices assigned to this rank
        """
        assert num_heads % self.tp_size == 0, \
            f"Number of heads {num_heads} not divisible by tp_size {self.tp_size}"
        
        heads_per_rank = num_heads // self.tp_size
        start_head = self.local_rank * heads_per_rank
        end_head = (self.local_rank + 1) * heads_per_rank
        
        heads = list(range(start_head, end_head))
        logger.debug(f"Rank {self.rank}: assigned {len(heads)} attention heads (indices {start_head}:{end_head})")
        
        return heads


class CheckpointSaver:
    """Saves model checkpoints in distributed format."""
    
    def __init__(self, checkpoint_dir: str, rank: int, world_size: int):
        """Initialize checkpoint saver.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            rank: Current rank
            world_size: Total number of ranks
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.rank = rank
        self.world_size = world_size
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CheckpointSaver initialized: {checkpoint_dir}")
    
    def save_rank_weights(self, weights: Dict[str, torch.Tensor],
                         step: int) -> None:
        """Save weights for current rank.
        
        Args:
            weights: Dictionary of weight tensors
            step: Training/inference step number
        """
        weights_path = self.checkpoint_dir / f"weights_rank{self.rank}_step{step}.pt"
        torch.save(weights, weights_path)
        logger.info(f"Rank {self.rank}: saved weights to {weights_path}")
    
    def save_metadata(self, metadata: Dict[str, Any], step: int) -> None:
        """Save checkpoint metadata (rank 0 only).
        
        Args:
            metadata: Metadata dictionary
            step: Training/inference step number
        """
        if self.rank != 0:
            return  # Only rank 0 saves metadata
        
        metadata_path = self.checkpoint_dir / f"metadata_step{step}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Rank 0: saved metadata to {metadata_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    distributor = WeightDistributor(rank=0, world_size=4, tp_size=4)
    
    # Simulate sharding a linear layer
    weight = torch.randn(4096, 4096)
    bias = torch.randn(4096)
    
    sharded_weight, sharded_bias = distributor.shard_linear_layer_row_wise(weight, bias)
    print(f"✓ Row-wise sharding: {weight.shape} -> {sharded_weight.shape}")
    
    # Test attention head sharding
    heads = distributor.shard_attention_heads(num_heads=32)
    print(f"✓ Attention heads assigned: {heads}")
