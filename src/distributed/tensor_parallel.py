"""
Task 1.1.5: Tensor Parallelism Layer Implementation

Row-wise tensor parallelism for distributed LLM inference.
Partitions model weights across multiple GPUs using row-wise strategy.

Architecture:
  - RowParallelLinear: Output dimension sharding
  - ColumnParallelLinear: Input dimension sharding  
  - DistributedModelWrapper: Automatic parallelization
  - Communication utilities: NCCL all-reduce integration

Performance Target:
  - 3.8-4.2× speedup on 4 GPUs (>95% efficiency)
  - <1ms all-reduce latency for 10MB tensors
  - <10% communication overhead vs. computation
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import ReduceOp

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallelism."""
    
    # World parallelism settings
    world_size: int = 1
    rank: int = 0
    
    # Communication settings
    backend: str = "nccl"
    use_async_reduce: bool = False
    
    # Memory settings
    gradient_checkpointing: bool = False
    
    # Logging
    debug: bool = False


# ============================================================================
# Communication Utilities
# ============================================================================

class DistributedAllReduce(torch.autograd.Function):
    """All-reduce backward compatibility function."""
    
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Perform all-reduce on tensor.
        
        Args:
            tensor: Input tensor (any shape)
            
        Returns:
            Tensor with values summed across all ranks
        """
        if dist.is_initialized():
            dist.all_reduce(tensor, op=ReduceOp.SUM)
        return tensor
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass: Broadcast gradient."""
        if dist.is_initialized():
            dist.all_reduce(grad_output, op=ReduceOp.SUM)
        return grad_output


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform all-reduce sum operation on tensor.
    
    Args:
        tensor: Input tensor
        
    Returns:
        All-reduced tensor (in-place modification)
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=ReduceOp.SUM)
    return tensor


def broadcast_tensor(tensor: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source rank to all ranks.
    
    Args:
        tensor: Input tensor
        src_rank: Source rank for broadcast
        
    Returns:
        Broadcasted tensor
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast(tensor, src=src_rank)
    return tensor


# ============================================================================
# Tensor Parallel Layers
# ============================================================================

class RowParallelLinear(nn.Module):
    """
    Linear layer with row-wise (output dimension) tensor parallelism.
    
    Weight partitioning:
      W ∈ ℝ^(out_features × in_features) → W_i ∈ ℝ^(out_features/world_size × in_features)
    
    Forward pass:
      Input (replicated): x ∈ ℝ^(batch × in_features)
      Local output: y_i = x @ W_i.T
      Final output: y = [y_0 | y_1 | ... | y_N-1] (concatenated)
    
    Communication: None required (implicit in concatenation)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize row-parallel linear layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
            world_size: Number of parallel processes (auto-detect if None)
            rank: Current rank (auto-detect if None)
            dtype: Data type for weights and bias
        """
        super().__init__()
        
        # Auto-detect world size and rank
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        
        # Verify dimensions
        if out_features % world_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by "
                f"world_size ({world_size})"
            )
        
        self.out_features_local = out_features // world_size
        
        # Initialize weights and bias (local partitions)
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features_local,
                in_features,
                dtype=dtype,
                device=torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
            )
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    self.out_features_local,
                    dtype=dtype,
                    device=torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
                )
            )
        else:
            self.register_parameter("bias", None)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize weights and bias using standard normal distribution."""
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='linear')
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Compute linear transformation on partitioned output dimension.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features] (replicated)
            
        Returns:
            Output tensor [batch_size, seq_len, out_features_local]
            
        Note:
            For full output, concatenate across all ranks:
            y = torch.cat([y_0, y_1, ..., y_N-1], dim=-1)
        """
        return F.linear(x, self.weight, self.bias)
    
    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return (
            f"in_features={self.in_features}, "
            f"out_features_local={self.out_features_local}, "
            f"out_features_total={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"world_size={self.world_size}, "
            f"rank={self.rank}"
        )


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column-wise (input dimension) tensor parallelism.
    
    Weight partitioning:
      W ∈ ℝ^(out_features × in_features) → W_i ∈ ℝ^(out_features × in_features/world_size)
    
    Forward pass:
      Input (partitioned): x_i ∈ ℝ^(batch × in_features/world_size)
      Local output: y_i = x_i @ W_i.T
      Final output: y = all_reduce(y_i, op=sum)
    
    Communication: One all-reduce at end of forward pass
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize column-parallel linear layer.
        
        Args:
            in_features: Input feature dimension (local partition)
            out_features: Output feature dimension (replicated)
            bias: Whether to use bias
            world_size: Number of parallel processes (auto-detect if None)
            rank: Current rank (auto-detect if None)
            dtype: Data type for weights and bias
        """
        super().__init__()
        
        # Auto-detect world size and rank
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.in_features = in_features
        self.in_features_local = in_features // world_size
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        
        # Verify dimensions
        if in_features % world_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"world_size ({world_size})"
            )
        
        # Initialize weights and bias (input partitioned, output replicated)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                self.in_features_local,
                dtype=dtype,
                device=device
            )
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize weights and bias using standard normal distribution."""
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='linear')
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Compute linear transformation with all-reduce synchronization.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features_local] (partitioned)
            
        Returns:
            Output tensor [batch_size, seq_len, out_features] (replicated)
        """
        # Local computation
        output = F.linear(x, self.weight, self.bias)
        
        # Synchronize across all ranks
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(output, op=ReduceOp.SUM)
        
        return output
    
    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return (
            f"in_features_local={self.in_features_local}, "
            f"in_features_total={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"world_size={self.world_size}, "
            f"rank={self.rank}"
        )


# ============================================================================
# Distributed Model Wrapper
# ============================================================================

class DistributedModelWrapper(nn.Module):
    """
    Wrapper that automatically converts standard model to distributed tensor parallel.
    
    Converts:
      nn.Linear → RowParallelLinear (for most layers)
      (Attention projections use Column for backward compatibility)
    
    Usage:
      model = DistributedModelWrapper(base_model, world_size=4, rank=0)
      output = model(input)
    """
    
    def __init__(
        self,
        model: nn.Module,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        """
        Initialize distributed model wrapper.
        
        Args:
            model: Base model to parallelize
            world_size: Number of parallel processes (auto-detect if None)
            rank: Current rank (auto-detect if None)
        """
        super().__init__()
        
        # Auto-detect world size and rank
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.base_model = model
        self.world_size = world_size
        self.rank = rank
        self.parallelized_layers = []
        
        # Automatically parallelize layers
        self._parallelize_layers()
    
    def _parallelize_layers(self):
        """
        Traverse model and replace nn.Linear with parallel versions.
        """
        for name, module in list(self.base_model.named_modules()):
            if isinstance(module, nn.Linear) and name:  # Skip root module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                parent = self.base_model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                
                # Determine which type of parallelization to use
                is_output_projection = 'proj' in name.lower() or 'out' in name.lower()
                
                if is_output_projection or self.world_size == 1:
                    # Use row-parallel for output projections
                    parallel_layer = RowParallelLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        world_size=self.world_size,
                        rank=self.rank,
                        dtype=module.weight.dtype,
                    )
                else:
                    # Use column-parallel for others
                    parallel_layer = ColumnParallelLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        world_size=self.world_size,
                        rank=self.rank,
                        dtype=module.weight.dtype,
                    )
                
                # Copy weights if available
                if module.weight is not None:
                    with torch.no_grad():
                        if isinstance(parallel_layer, RowParallelLinear):
                            # Partition output dimension
                            out_idx = self.rank * module.out_features // self.world_size
                            parallel_layer.weight.copy_(
                                module.weight[out_idx:out_idx + parallel_layer.out_features_local, :]
                            )
                        else:
                            # Partition input dimension
                            in_idx = self.rank * module.in_features // self.world_size
                            parallel_layer.weight.copy_(
                                module.weight[:, in_idx:in_idx + parallel_layer.in_features_local]
                            )
                
                if module.bias is not None and parallel_layer.bias is not None:
                    with torch.no_grad():
                        if isinstance(parallel_layer, RowParallelLinear):
                            out_idx = self.rank * module.out_features // self.world_size
                            parallel_layer.bias.copy_(
                                module.bias[out_idx:out_idx + parallel_layer.out_features_local]
                            )
                        else:
                            parallel_layer.bias.copy_(module.bias)
                
                # Replace layer
                setattr(parent, child_name, parallel_layer)
                self.parallelized_layers.append(name)
                logger.info(f"Parallelized {name}: {type(parallel_layer).__name__}")
    
    def forward(self, *args, **kwargs):
        """Forward pass through parallelized model."""
        return self.base_model(*args, **kwargs)


# ============================================================================
# Utility Functions
# ============================================================================

def synchronize_across_ranks() -> None:
    """Synchronize all ranks at barrier."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def get_tensor_parallel_config() -> TensorParallelConfig:
    """
    Get tensor parallelism configuration from environment.
    
    Returns:
        TensorParallelConfig with auto-detected settings
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    debug = os.environ.get("TP_DEBUG", "0") == "1"
    
    return TensorParallelConfig(
        world_size=world_size,
        rank=rank,
        debug=debug,
    )


def init_tensor_parallel(backend: str = "nccl", **kwargs) -> TensorParallelConfig:
    """
    Initialize tensor parallelism system.
    
    Args:
        backend: Communication backend ("nccl", "gloo", etc.)
        **kwargs: Additional arguments passed to dist.init_process_group
        
    Returns:
        TensorParallelConfig
    """
    if not dist.is_available():
        logger.warning("Distributed package not available, using single GPU")
        return TensorParallelConfig(world_size=1, rank=0)
    
    if dist.is_initialized():
        logger.warning("Process group already initialized")
    else:
        dist.init_process_group(backend=backend, **kwargs)
    
    return get_tensor_parallel_config()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "TensorParallelConfig",
    "RowParallelLinear",
    "ColumnParallelLinear",
    "DistributedModelWrapper",
    "all_reduce_sum",
    "broadcast_tensor",
    "synchronize_across_ranks",
    "get_tensor_parallel_config",
    "init_tensor_parallel",
]
