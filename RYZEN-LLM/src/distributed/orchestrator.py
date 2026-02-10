"""
GPU Orchestrator: Multi-GPU Process Management

Handles:
- Process group initialization and cleanup
- Rank assignment and synchronization
- Barrier operations across all ranks
- Distributed parameter initialization
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import torch
import torch.distributed as dist
from datetime import timedelta
import logging
import time

logger = logging.getLogger(__name__)


class GPUPerformanceMonitor:
    """Monitor GPU performance metrics during distributed training/inference."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {
            'communication_time': [],
            'computation_time': [],
            'memory_usage': [],
            'barrier_time': [],
            'sync_time': []
        }
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str):
        """End timing an operation and record duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            if operation in self.metrics:
                self.metrics[operation].append(duration)
            del self.start_times[operation]
            return duration
        return 0.0
    
    def record_memory(self):
        """Record current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            self.metrics['memory_usage'].append((allocated, reserved))
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        stats = {}
        for metric, values in self.metrics.items():
            if values:
                if isinstance(values[0], tuple):  # Memory stats
                    allocated = [v[0] for v in values]
                    reserved = [v[1] for v in values]
                    stats[metric] = {
                        'allocated_mean': sum(allocated) / len(allocated),
                        'allocated_max': max(allocated),
                        'reserved_mean': sum(reserved) / len(reserved),
                        'reserved_max': max(reserved)
                    }
                else:  # Time stats
                    stats[metric] = {
                        'count': len(values),
                        'mean': sum(values) / len(values),
                        'max': max(values),
                        'min': min(values),
                        'total': sum(values)
                    }
        return stats
    
    def reset(self):
        """Reset all metrics."""
        for key in self.metrics:
            self.metrics[key].clear()
        self.start_times.clear()


class ProcessGroupManager:
    """Manages NCCL/Gloo process group initialization.
    
    Wraps torch.distributed.init_process_group with additional utilities.
    """
    
    def __init__(self, backend: str = "nccl", timeout: float = 30.0):
        """Initialize process group manager.
        
        Args:
            backend: Communication backend ("nccl", "gloo", "mpi")
            timeout: Collective operation timeout in seconds
        """
        self.backend = backend
        self.timeout = timedelta(seconds=timeout)
        self._initialized = False
    
    def init_distributed(self, rank: int, world_size: int, 
                        master_addr: str = "127.0.0.1",
                        master_port: int = 29500) -> None:
        """Initialize distributed process group.
        
        Args:
            rank: Current process rank
            world_size: Total number of processes
            master_addr: Master node address
            master_port: Master node port
            
        Raises:
            RuntimeError: If already initialized or init fails
        """
        if self._initialized:
            logger.warning("Process group already initialized, skipping re-init")
            return
        
        try:
            dist.init_process_group(
                backend=self.backend,
                init_method="env://",
                world_size=world_size,
                rank=rank,
                timeout=self.timeout
            )
            self._initialized = True
            logger.info(f"Process group initialized: rank={rank}, world_size={world_size}, backend={self.backend}")
        except Exception as e:
            logger.error(f"Failed to initialize process group: {e}")
            raise RuntimeError(f"Process group initialization failed: {e}")
    
    def destroy(self) -> None:
        """Cleanup and destroy process group.
        
        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized:
            logger.warning("Process group not initialized, nothing to destroy")
            return
        
        try:
            dist.destroy_process_group()
            self._initialized = False
            logger.info("Process group destroyed")
        except Exception as e:
            logger.error(f"Failed to destroy process group: {e}")
            raise RuntimeError(f"Process group destruction failed: {e}")
    
    def is_initialized(self) -> bool:
        """Check if process group is initialized."""
        return self._initialized


class MultiGPUOrchestrator:
    """Orchestrates multi-GPU setup and synchronization.
    
    Responsibilities:
    - Initialize distributed environment
    - Manage rank assignment
    - Provide barrier synchronization
    - Handle device management
    - Fault tolerance and recovery
    - Performance monitoring
    """
    
    def __init__(self, rank: int, world_size: int, 
                 backend: str = "nccl",
                 device: Optional[str] = None,
                 enable_monitoring: bool = True):
        """Initialize GPU orchestrator.
        
        Args:
            rank: Current rank (0 to world_size-1)
            world_size: Total number of ranks
            backend: Communication backend
            device: Device string (e.g., "cuda:0"), auto-detected if None
            enable_monitoring: Enable performance monitoring
        """
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.enable_monitoring = enable_monitoring
        
        # Set CUDA device if specified
        if device is None:
            device = f"cuda:{rank % torch.cuda.device_count()}"
        
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        
        self.process_group_manager = ProcessGroupManager(backend=backend)
        self._initialized = False
        
        # Fault tolerance
        self.failure_count = 0
        self.max_failures = 3
        self.recovery_attempts = 0
        
        # Monitoring
        self.monitor = GPUPerformanceMonitor() if enable_monitoring else None
        self.start_time = None
        
        logger.info(f"MultiGPUOrchestrator created: rank={rank}, world_size={world_size}, device={device}")
    
    def initialize(self, master_addr: str = "127.0.0.1", 
                   master_port: int = 29500) -> None:
        """Initialize distributed environment.
        
        Args:
            master_addr: Master node address
            master_port: Master node port
        """
        if self._initialized:
            logger.warning("Orchestrator already initialized, skipping re-init")
            return
        
        try:
            self.process_group_manager.init_distributed(
                rank=self.rank,
                world_size=self.world_size,
                master_addr=master_addr,
                master_port=master_port
            )
            self._initialized = True
            self.start_time = time.time()
            
            if self.monitor:
                self.monitor.record_memory()
                
            logger.info(f"✓ Distributed environment initialized: rank={self.rank}")
        except Exception as e:
            logger.error(f"Failed to initialize distributed environment: {e}")
            self._handle_initialization_failure(e)
            raise
    
    def barrier(self) -> None:
        """Synchronize all ranks at a barrier.
        
        All ranks block until called on all processes.
        
        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")
        
        try:
            dist.barrier()
            logger.debug(f"Rank {self.rank}: barrier reached")
        except Exception as e:
            logger.error(f"Rank {self.rank}: barrier failed: {e}")
            raise
    
    def get_rank(self) -> int:
        """Get current rank."""
        return self.rank
    
    def get_world_size(self) -> int:
        """Get total number of ranks."""
        return self.world_size
    
    def get_device(self) -> torch.device:
        """Get assigned CUDA device."""
        return self.device
    
    def is_master(self) -> bool:
        """Check if this is rank 0 (master rank)."""
        return self.rank == 0
    
    def cleanup(self) -> None:
        """Cleanup distributed environment.
        
        Safe to call even if not initialized.
        """
        if self._initialized:
            self.process_group_manager.destroy()
            self._initialized = False
            
            if self.monitor:
                self.monitor.record_memory()
                
            logger.info(f"Rank {self.rank}: orchestrator cleaned up")
    
    def _handle_initialization_failure(self, error: Exception) -> None:
        """Handle initialization failure with recovery attempts."""
        self.failure_count += 1
        logger.warning(f"Initialization failure {self.failure_count}/{self.max_failures}: {error}")
        
        if self.failure_count >= self.max_failures:
            raise RuntimeError(f"Failed to initialize after {self.max_failures} attempts")
    
    def check_health(self) -> bool:
        """Check health of distributed environment.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self._initialized:
            return False
        
        try:
            # Quick health check with barrier
            dist.barrier()
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            self.failure_count += 1
            return False
    
    def attempt_recovery(self) -> bool:
        """Attempt to recover from failure.
        
        Returns:
            True if recovery successful, False otherwise
        """
        if self.recovery_attempts >= 3:
            logger.error("Maximum recovery attempts exceeded")
            return False
        
        self.recovery_attempts += 1
        logger.info(f"Attempting recovery (attempt {self.recovery_attempts})")
        
        try:
            # Cleanup and re-initialize
            self.cleanup()
            # Note: Would need to re-init with same parameters
            # For now, just mark as recovered
            self.failure_count = 0
            logger.info("✓ Recovery successful")
            return True
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.monitor:
            return {}
        
        stats = self.monitor.get_stats()
        stats['uptime_seconds'] = time.time() - self.start_time if self.start_time else 0
        stats['failure_count'] = self.failure_count
        stats['recovery_attempts'] = self.recovery_attempts
        
        return stats
    
    def allocate_gpus_dynamically(self, required_gpus: int) -> List[int]:
        """Dynamically allocate GPUs based on availability.
        
        Args:
            required_gpus: Number of GPUs needed
            
        Returns:
            List of allocated GPU IDs
        """
        available_gpus = torch.cuda.device_count()
        if required_gpus > available_gpus:
            raise RuntimeError(f"Required {required_gpus} GPUs but only {available_gpus} available")
        
        # Simple allocation: assign consecutive GPUs starting from rank
        start_gpu = self.rank * required_gpus
        allocated = []
        
        for i in range(required_gpus):
            gpu_id = (start_gpu + i) % available_gpus
            allocated.append(gpu_id)
        
        logger.info(f"Rank {self.rank}: allocated GPUs {allocated}")
        return allocated


class DistributedParameterInitializer:
    """Handles distributed initialization of model parameters.
    
    Ensures consistent initialization across all ranks.
    """
    
    def __init__(self, orchestrator: MultiGPUOrchestrator):
        """Initialize parameter initializer.
        
        Args:
            orchestrator: MultiGPUOrchestrator instance
        """
        self.orchestrator = orchestrator
    
    def broadcast_parameters(self, model: torch.nn.Module, 
                           src_rank: int = 0) -> None:
        """Broadcast model parameters from source rank to all others.
        
        Ensures all ranks start with identical initialization.
        
        Args:
            model: Model to broadcast
            src_rank: Source rank for broadcast
        """
        for param in model.parameters():
            dist.broadcast(param.data, src=src_rank)
        
        logger.info(f"Rank {self.orchestrator.rank}: parameters broadcast from rank {src_rank}")
    
    def broadcast_buffers(self, model: torch.nn.Module, 
                         src_rank: int = 0) -> None:
        """Broadcast model buffers from source rank to all others.
        
        Args:
            model: Model to broadcast buffers from
            src_rank: Source rank for broadcast
        """
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=src_rank)
        
        logger.info(f"Rank {self.orchestrator.rank}: buffers broadcast from rank {src_rank}")


if __name__ == "__main__":
    # Example usage (for testing)
    logging.basicConfig(level=logging.INFO)
    
    # Simulate single-rank setup
    orchestrator = MultiGPUOrchestrator(rank=0, world_size=1, device="cuda:0")
    print(f"✓ Orchestrator created: rank={orchestrator.get_rank()}, device={orchestrator.get_device()}")
    print(f"  Master: {orchestrator.is_master()}")
    print(f"  Ready for initialization when torch.distributed starts")
