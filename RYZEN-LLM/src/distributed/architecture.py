"""
Distributed Inference Architecture Interfaces

Defines core abstractions for distributed model inference with tensor parallelism.
This module establishes the API contracts that implementations must satisfy.

Key Abstractions:
    - DistributedConfig: Configuration for distributed setup
    - CommunicationHandler: Abstract communication interface
    - ParallelModelWrapper: Model wrapping for distributed inference
    - TensorParallelLayer: Base class for parallelized layers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn as nn


@dataclass
class DistributedConfig:
    """Configuration for distributed inference setup.
    
    Attributes:
        world_size: Total number of GPUs/ranks
        rank: Current process rank (0 to world_size-1)
        backend: Communication backend ('nccl', 'gloo', 'mpi')
        master_addr: Master node address for multi-node setup
        master_port: Master node port
        device: Device for this rank ('cuda:0', 'cuda:1', etc.)
        tensor_parallel_size: Size of tensor parallelism dimension
        pipeline_parallel_size: Size of pipeline parallelism dimension (future)
        use_checkpointing: Enable gradient checkpointing
        communication_timeout: Timeout for collective operations (seconds)
    """
    world_size: int
    rank: int
    backend: str = "nccl"
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    device: Optional[str] = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    use_checkpointing: bool = False
    communication_timeout: float = 30.0
    
    def __post_init__(self):
        """Validate configuration consistency."""
        assert self.world_size > 0, "world_size must be positive"
        assert 0 <= self.rank < self.world_size, f"rank must be in [0, {self.world_size})"
        assert self.backend in ["nccl", "gloo", "mpi"], f"Invalid backend: {self.backend}"
        assert self.tensor_parallel_size > 0, "tensor_parallel_size must be positive"
        assert self.pipeline_parallel_size > 0, "pipeline_parallel_size must be positive"
        assert (self.tensor_parallel_size * self.pipeline_parallel_size <= self.world_size), \
            "Product of parallelism sizes must not exceed world_size"


class CommunicationHandler(ABC):
    """Abstract base class for distributed communication.
    
    Handles point-to-point, collective, and broadcast operations across ranks.
    Implementations use NCCL, Gloo, or MPI backends.
    """
    
    @abstractmethod
    def init_process_group(self, config: DistributedConfig) -> None:
        """Initialize the process group for distributed communication.
        
        Args:
            config: Distributed configuration
            
        Raises:
            RuntimeError: If process group already initialized or setup fails
        """
        pass
    
    @abstractmethod
    def barrier(self) -> None:
        """Synchronize all ranks at a barrier point.
        
        All ranks block until this is called on all processes.
        """
        pass
    
    @abstractmethod
    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> None:
        """Reduce tensor across all ranks and scatter result.
        
        Args:
            tensor: Tensor to reduce (in-place operation)
            op: Reduction operation ('sum', 'mean', 'min', 'max')
            
        Raises:
            RuntimeError: If communication fails
        """
        pass
    
    @abstractmethod
    def broadcast(self, tensor: torch.Tensor, src_rank: int) -> None:
        """Broadcast tensor from source rank to all other ranks.
        
        Args:
            tensor: Tensor to broadcast (in-place operation)
            src_rank: Source rank (0 to world_size-1)
        """
        pass
    
    @abstractmethod
    def all_gather(self, output_tensors: List[torch.Tensor], 
                   input_tensor: torch.Tensor) -> None:
        """Gather tensors from all ranks.
        
        Args:
            output_tensors: List of output tensors (one per rank)
            input_tensor: Local tensor to gather
        """
        pass
    
    @abstractmethod
    def send(self, tensor: torch.Tensor, dst_rank: int, tag: int = 0) -> None:
        """Send tensor to destination rank (blocking).
        
        Args:
            tensor: Tensor to send
            dst_rank: Destination rank
            tag: Communication tag for multiplexing
        """
        pass
    
    @abstractmethod
    def recv(self, tensor: torch.Tensor, src_rank: int, tag: int = 0) -> None:
        """Receive tensor from source rank (blocking).
        
        Args:
            tensor: Tensor to receive into
            src_rank: Source rank
            tag: Communication tag for multiplexing
        """
        pass
    
    @abstractmethod
    def get_rank(self) -> int:
        """Get current process rank."""
        pass
    
    @abstractmethod
    def get_world_size(self) -> int:
        """Get total number of processes."""
        pass


class TensorParallelLayer(nn.Module, ABC):
    """Base class for tensor parallel layers.
    
    Subclasses implement specific parallelization strategies
    (row-wise, column-wise, head-wise for attention).
    """
    
    def __init__(self, comm_handler: CommunicationHandler, rank: int, world_size: int):
        """Initialize tensor parallel layer.
        
        Args:
            comm_handler: Communication handler for sync operations
            rank: Current rank
            world_size: Total number of ranks
        """
        super().__init__()
        self.comm_handler = comm_handler
        self.rank = rank
        self.world_size = world_size
        self.tp_size = world_size  # Default: full tensor parallelism
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with tensor parallelism.
        
        Args:
            x: Input tensor (assumed replicated across ranks)
            
        Returns:
            Output tensor (parallelization-specific layout)
        """
        pass


class ParallelModelWrapper(nn.Module, ABC):
    """Wraps a model for distributed inference with tensor parallelism.
    
    Handles:
    - Layer replacement with parallel versions
    - Forward pass orchestration
    - Gradient synchronization
    - Weight distribution
    """
    
    def __init__(self, model: nn.Module, config: DistributedConfig, 
                 comm_handler: CommunicationHandler):
        """Initialize model wrapper.
        
        Args:
            model: Original model to parallelize
            config: Distributed configuration
            comm_handler: Communication handler
        """
        super().__init__()
        self.model = model
        self.config = config
        self.comm_handler = comm_handler
    
    @abstractmethod
    def apply_tensor_parallelism(self) -> None:
        """Transform model layers to use tensor parallelism.
        
        This should replace linear layers, attention heads, etc.
        with their parallel counterparts.
        """
        pass
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward pass with distributed orchestration.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            **kwargs: Additional model-specific arguments
            
        Returns:
            Model output (logits, embeddings, etc.)
        """
        pass
    
    @abstractmethod
    def distribute_weights(self) -> None:
        """Distribute model weights across GPUs.
        
        Handles sharding of tensors according to parallelism strategy.
        """
        pass
    
    @abstractmethod
    def synchronize_gradients(self) -> None:
        """Synchronize gradients across ranks after backward pass.
        
        Performs all-reduce or other gradient synchronization operations.
        """
        pass
    
    def get_local_model(self) -> nn.Module:
        """Get the locally wrapped model.
        
        Returns:
            The underlying model (potentially with modified layers)
        """
        return self.model


# Configuration validation utilities
def validate_distributed_setup(config: DistributedConfig) -> None:
    """Validate distributed configuration for common issues.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    if config.tensor_parallel_size > config.world_size:
        raise ValueError(
            f"tensor_parallel_size ({config.tensor_parallel_size}) cannot exceed "
            f"world_size ({config.world_size})"
        )
    
    if config.world_size % config.tensor_parallel_size != 0:
        raise ValueError(
            f"world_size ({config.world_size}) must be divisible by "
            f"tensor_parallel_size ({config.tensor_parallel_size})"
        )


if __name__ == "__main__":
    # Example usage and validation
    config = DistributedConfig(
        world_size=4,
        rank=0,
        tensor_parallel_size=4,
        device="cuda:0"
    )
    print(f"Distributed config: {config}")
    validate_distributed_setup(config)
    print("✓ Configuration validated")
