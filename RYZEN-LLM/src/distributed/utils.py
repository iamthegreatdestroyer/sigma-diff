"""
Distributed Inference Utilities

Logging, debugging helpers, and utility functions.
"""

import logging
import torch
import os
from typing import Optional


def setup_distributed_logging(rank: int, log_level: str = "INFO") -> logging.Logger:
    """Setup logging for distributed training.
    
    All ranks log, but with rank prefix to distinguish messages.
    
    Args:
        rank: Current rank
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(__name__)
    
    if not logger.handlers:  # Avoid duplicate handlers
        formatter = logging.Formatter(
            f"[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(getattr(logging, log_level.upper()))
    return logger


def get_device_count() -> int:
    """Get number of available CUDA devices.
    
    Returns:
        Number of GPUs
    """
    return torch.cuda.device_count()


def get_device_name(device_id: int = 0) -> str:
    """Get GPU name.
    
    Args:
        device_id: GPU index
        
    Returns:
        GPU name (e.g., "A100-40GB", "RTX 4090")
    """
    return torch.cuda.get_device_name(device_id)


def check_nccl_availability() -> bool:
    """Check if NCCL backend is available.
    
    Returns:
        True if NCCL is available
    """
    try:
        # Try to access NCCL version
        import torch.distributed as dist
        return hasattr(dist, 'Backend') and 'nccl' in dir(dist.Backend)
    except:
        return False


def get_distributed_config_from_env() -> dict:
    """Parse distributed configuration from environment variables.
    
    Expected environment variables:
        RANK: Current rank (0 to world_size-1)
        WORLD_SIZE: Total number of ranks
        MASTER_ADDR: Master node address
        MASTER_PORT: Master node port
    
    Returns:
        Configuration dictionary
    """
    config = {
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "master_addr": os.environ.get("MASTER_ADDR", "127.0.0.1"),
        "master_port": int(os.environ.get("MASTER_PORT", "29500")),
    }
    return config


def synchronize_across_processes() -> None:
    """Simple barrier synchronization (only works if distributed initialized).
    
    For testing/debugging without torch.distributed.init_process_group.
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except:
        pass  # Not in distributed context, skip


def get_memory_stats(device: torch.device) -> dict:
    """Get GPU memory statistics.
    
    Args:
        device: CUDA device
        
    Returns:
        Dictionary with memory stats (allocated, reserved, free)
    """
    if device.type != 'cuda':
        return {}
    
    torch.cuda.reset_peak_memory_stats(device)
    
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
    }


if __name__ == "__main__":
    # Test utilities
    print(f"Available GPUs: {get_device_count()}")
    for i in range(get_device_count()):
        print(f"  GPU {i}: {get_device_name(i)}")
    print(f"NCCL available: {check_nccl_availability()}")
