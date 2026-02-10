"""
Distributed Inference Module for RYZEN-LLM Phase 3

Provides tensor parallelism, GPU orchestration, and distributed model loading
for scaling LLM inference across multiple GPUs.

Components:
    - architecture: Core interfaces and abstractions
    - tensor_parallel: Tensor parallelism implementations
    - orchestrator: Multi-GPU process management
    - model_loader: Distributed checkpoint loading
    - communication: NCCL optimization utilities
    - utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "RYZEN-LLM Team"

# Core interfaces (lazy imports to avoid circular dependencies)
from .architecture import (
    DistributedConfig,
    CommunicationHandler,
    ParallelModelWrapper,
)
from .bitnet_parallel import BitNetParallelModelWrapper

__all__ = [
    "__version__",
    "DistributedConfig",
    "CommunicationHandler",
    "ParallelModelWrapper",
    "BitNetParallelModelWrapper",
]
