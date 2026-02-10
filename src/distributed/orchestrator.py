"""
Task 1.1.6: Multi-GPU Orchestrator Implementation

Manages distributed inference across multiple GPUs with process coordination,
resource allocation, health monitoring, and failure recovery.

Components:
  - ProcessGroupManager: Distributed process group lifecycle
  - ResourceAllocator: GPU memory and buffer management
  - HealthMonitor: Process and GPU health tracking
  - FailureRecoveryManager: Failure detection and recovery
  - MultiGPUOrchestrator: Main orchestration controller
  - OrchestratorConfig: Configuration dataclass

Performance Targets:
  - Orchestration overhead: <5ms per inference step
  - Health check overhead: <1ms per check
  - Memory efficiency: >90% utilization of allocated GPU memory
  - Recovery time: <100ms for automatic restart
"""

import os
import time
import logging
import threading
import collections
from typing import Any, Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

logger = logging.getLogger(__name__)


# ============================================================================
# Enums & Types
# ============================================================================

class ProcessStatus(Enum):
    """Status of a process."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"
    OFFLINE = "offline"


class FailureMode(Enum):
    """Types of failures that can occur."""
    GPU_OOM = "gpu_oom"
    COMMUNICATION_TIMEOUT = "communication_timeout"
    STALE_PROCESS = "stale_process"
    GPU_FAILURE = "gpu_failure"
    ALLREDUCE_ERROR = "allreduce_error"
    UNKNOWN = "unknown"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class OrchestratorConfig:
    """Configuration for multi-GPU orchestrator."""
    
    # Process group settings
    backend: str = "nccl"
    timeout_sec: float = 30.0
    
    # Resource settings
    memory_fraction: float = 0.9  # Use 90% of GPU memory
    
    # Monitoring settings
    health_check_interval_sec: float = 5.0
    heartbeat_timeout_sec: float = 10.0
    
    # Failure recovery
    enable_auto_restart: bool = True
    max_restart_attempts: int = 3
    
    # Load balancing
    batch_size: int = 32
    enable_dynamic_batching: bool = False
    
    # Logging
    log_level: str = "INFO"
    enable_profiling: bool = False


# ============================================================================
# ProcessGroupManager
# ============================================================================

class ProcessGroupManager:
    """
    Manages PyTorch distributed process groups and rank assignments.
    
    Responsibilities:
      - Rank and world size determination
      - Backend selection (NCCL for GPU, gloo for CPU)
      - Process group creation and lifecycle
      - Barrier synchronization
    """
    
    def __init__(self, backend: str = "nccl", timeout_sec: float = 30.0):
        """
        Initialize process group manager.
        
        Args:
            backend: Communication backend ("nccl" or "gloo")
            timeout_sec: Communication timeout in seconds
        """
        self.backend = backend
        self.timeout = timedelta(seconds=timeout_sec)
        self.rank = None
        self.world_size = None
        self.device = None
        self.initialized = False
    
    def initialize(self, rank: int, world_size: int,
                  master_addr: str = "localhost",
                  master_port: int = 29500):
        """
        Initialize distributed process group.
        
        Args:
            rank: Process rank (0 to world_size-1)
            world_size: Total number of processes
            master_addr: Master node address
            master_port: Master node port
            
        Raises:
            RuntimeError: If initialization fails
        """
        if self.initialized:
            logger.warning("Process group already initialized")
            return
        
        self.rank = rank
        self.world_size = world_size
        
        # Set environment variables for initialization
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        try:
            dist.init_process_group(
                backend=self.backend,
                rank=rank,
                world_size=world_size,
                timeout=self.timeout
            )
            self.initialized = True
            self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
            logger.info(f"Rank {rank}/{world_size} initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize process group: {e}")
            raise
    
    def barrier(self):
        """Synchronize all processes at barrier."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()
    
    def broadcast_tensor(self, tensor: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
        """
        Broadcast tensor from source rank to all.
        
        Args:
            tensor: Tensor to broadcast
            src_rank: Source rank for broadcast
            
        Returns:
            Broadcasted tensor
        """
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast(tensor, src=src_rank)
        return tensor
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """
        Perform all-reduce operation.
        
        Args:
            tensor: Tensor to reduce
            op: Operation ("sum", "mean", "max", "min")
            
        Returns:
            All-reduced tensor
        """
        if dist.is_initialized() and dist.get_world_size() > 1:
            reduce_op = {
                "sum": ReduceOp.SUM,
                "mean": ReduceOp.AVG,
                "max": ReduceOp.MAX,
                "min": ReduceOp.MIN,
            }.get(op, ReduceOp.SUM)
            dist.all_reduce(tensor, op=reduce_op)
        return tensor
    
    def finalize(self):
        """Cleanup and shutdown process group."""
        if self.initialized and dist.is_initialized():
            dist.destroy_process_group()
            self.initialized = False
            logger.info(f"Rank {self.rank} finalized")


# ============================================================================
# ResourceAllocator
# ============================================================================

class ResourceAllocator:
    """
    Allocates and manages GPU resources for distributed inference.
    
    Features:
      - Per-GPU memory budgeting
      - Automatic buffer pooling
      - Memory utilization tracking
      - Dynamic allocation/deallocation
    """
    
    def __init__(self, rank: int, device: torch.device,
                 memory_fraction: float = 0.9):
        """
        Initialize resource allocator.
        
        Args:
            rank: GPU rank
            device: CUDA device
            memory_fraction: Fraction of GPU memory to use
        """
        self.rank = rank
        self.device = device
        self.memory_fraction = memory_fraction
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            total_memory = props.total_memory
            self.max_memory = int(total_memory * memory_fraction)
        else:
            self.max_memory = int(16e9)  # Default 16GB for CPU
        
        self.allocated_memory = 0
        self.buffers = {}  # name -> tensor
        logger.info(f"Rank {rank}: Memory budget {self.max_memory / 1e9:.1f}GB")
    
    def allocate_tensor(self, name: str, shape: Tuple[int, ...],
                       dtype: torch.dtype) -> torch.Tensor:
        """
        Allocate named tensor with bounds checking.
        
        Args:
            name: Tensor identifier
            shape: Tensor shape
            dtype: Tensor data type
            
        Returns:
            Allocated tensor on device
            
        Raises:
            RuntimeError: If allocation would exceed memory budget
        """
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        required = tensor.numel() * torch.tensor(0, dtype=dtype).element_size()
        
        if self.allocated_memory + required > self.max_memory:
            raise RuntimeError(
                f"Rank {self.rank}: Allocation exceeds budget "
                f"({self.allocated_memory + required} > {self.max_memory})"
            )
        
        self.buffers[name] = tensor
        self.allocated_memory += required
        logger.debug(f"Rank {self.rank}: Allocated {name} {shape} ({required/1e6:.1f}MB)")
        return tensor
    
    def deallocate_tensor(self, name: str):
        """Deallocate named tensor."""
        if name in self.buffers:
            tensor = self.buffers.pop(name)
            required = tensor.numel() * torch.tensor(0, dtype=tensor.dtype).element_size()
            self.allocated_memory -= required
            logger.debug(f"Rank {self.rank}: Deallocated {name}")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
        else:
            allocated = self.allocated_memory
            reserved = self.allocated_memory
        
        return {
            "allocated_gb": allocated / 1e9,
            "reserved_gb": reserved / 1e9,
            "max_budget_gb": self.max_memory / 1e9,
            "utilization_percent": (allocated / reserved * 100) if reserved > 0 else 0,
        }
    
    def reset(self):
        """Clear all allocations and release memory."""
        self.buffers.clear()
        self.allocated_memory = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# HealthMonitor
# ============================================================================

class HealthMonitor:
    """
    Monitors GPU and process health during distributed inference.
    
    Tracks:
      - GPU memory utilization
      - Process responsiveness
      - Communication latency
      - Error accumulation
    """
    
    def __init__(self, rank: int, device: torch.device,
                 check_interval_sec: float = 5.0):
        """
        Initialize health monitor.
        
        Args:
            rank: GPU rank
            device: CUDA device
            check_interval_sec: Minimum seconds between checks
        """
        self.rank = rank
        self.device = device
        self.check_interval = check_interval_sec
        self.last_check = time.time()
        
        # Metrics history (rolling window of 100 samples)
        self.metrics_history = collections.deque(maxlen=100)
        self.error_count = 0
        self.last_heartbeat = time.time()
    
    def check_health(self) -> Dict[str, Any]:
        """
        Perform health check and return metrics.
        
        Returns:
            Dictionary with health metrics
        """
        now = time.time()
        if now - self.last_check < self.check_interval:
            return {}
        
        self.last_check = now
        
        # Collect metrics
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            mem_allocated = torch.cuda.memory_allocated(self.device)
            mem_reserved = torch.cuda.memory_reserved(self.device)
        else:
            mem_allocated = 0
            mem_reserved = 0
        
        metrics = {
            "timestamp": now,
            "rank": self.rank,
            "memory_allocated_mb": mem_allocated / 1e6,
            "memory_reserved_mb": mem_reserved / 1e6,
            "heartbeat_age_sec": now - self.last_heartbeat,
            "error_count": self.error_count,
            "status": ProcessStatus.HEALTHY.value,
        }
        
        # Determine status
        if self.error_count > 5:
            metrics["status"] = ProcessStatus.UNHEALTHY.value
        elif self.error_count > 0:
            metrics["status"] = ProcessStatus.DEGRADED.value
        
        self.metrics_history.append(metrics)
        return metrics
    
    def record_heartbeat(self):
        """Record that process is alive."""
        self.last_heartbeat = time.time()
        self.error_count = max(0, self.error_count - 1)  # Recover slowly
    
    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of recent health metrics."""
        if not self.metrics_history:
            return {"status": "unknown"}
        
        recent = list(self.metrics_history)[-10:]
        mem_allocated = [m["memory_allocated_mb"] for m in recent]
        
        return {
            "avg_memory_mb": sum(mem_allocated) / len(mem_allocated),
            "peak_memory_mb": max(mem_allocated),
            "error_count": self.error_count,
            "status": self.metrics_history[-1]["status"],
        }


# ============================================================================
# FailureRecoveryManager
# ============================================================================

class FailureRecoveryManager:
    """
    Handles detection and recovery from various failures.
    
    Supports:
      - GPU OOM recovery
      - Communication timeout recovery
      - Automatic restart
      - Graceful degradation
    """
    
    def __init__(self, max_retries: int = 3):
        """
        Initialize failure recovery manager.
        
        Args:
            max_retries: Maximum recovery attempts before giving up
        """
        self.max_retries = max_retries
        self.failure_count = 0
        self.failure_history = collections.deque(maxlen=50)
    
    def handle_gpu_oom(self, rank: int, batch_size: int) -> int:
        """
        Handle GPU out-of-memory condition.
        
        Args:
            rank: GPU rank
            batch_size: Current batch size
            
        Returns:
            New batch size (reduced)
        """
        logger.warning(f"Rank {rank}: GPU out of memory")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        new_batch_size = max(1, batch_size // 2)
        self.failure_history.append({
            "timestamp": time.time(),
            "type": FailureMode.GPU_OOM,
            "rank": rank,
        })
        
        logger.info(f"Rank {rank}: Reduced batch size {batch_size} → {new_batch_size}")
        return new_batch_size
    
    def handle_communication_timeout(self, rank: int) -> bool:
        """
        Handle communication timeout.
        
        Args:
            rank: Rank with timeout
            
        Returns:
            True if recovery possible, False otherwise
        """
        logger.error(f"Rank {rank}: Communication timeout")
        self.failure_count += 1
        
        self.failure_history.append({
            "timestamp": time.time(),
            "type": FailureMode.COMMUNICATION_TIMEOUT,
            "rank": rank,
        })
        
        if self.failure_count < self.max_retries:
            logger.info(f"Attempting recovery ({self.failure_count}/{self.max_retries})")
            return True
        else:
            logger.error("Max recovery attempts exceeded")
            return False
    
    def handle_gpu_failure(self, rank: int) -> bool:
        """
        Handle GPU hardware failure.
        
        Args:
            rank: Failed GPU rank
            
        Returns:
            True if can continue, False if must stop
        """
        logger.critical(f"Rank {rank}: GPU failure detected")
        self.failure_history.append({
            "timestamp": time.time(),
            "type": FailureMode.GPU_FAILURE,
            "rank": rank,
        })
        # GPU failure is fatal - cannot recover
        return False
    
    def reset(self):
        """Reset failure counter after successful step."""
        self.failure_count = max(0, self.failure_count - 1)
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get summary of recent failures."""
        if not self.failure_history:
            return {"failures": 0}
        
        failure_counts = {}
        for failure in self.failure_history:
            ftype = failure["type"].value
            failure_counts[ftype] = failure_counts.get(ftype, 0) + 1
        
        return {
            "total_failures": len(self.failure_history),
            "failure_types": failure_counts,
            "current_failure_count": self.failure_count,
        }


# ============================================================================
# MultiGPUOrchestrator
# ============================================================================

class MultiGPUOrchestrator:
    """
    Main orchestrator for distributed inference across multiple GPUs.
    
    Coordinates:
      - Process initialization and cleanup
      - Resource allocation and management
      - Health monitoring and failure detection
      - Distributed model execution
      - Load balancing and result aggregation
    """
    
    def __init__(self, config: OrchestratorConfig):
        """
        Initialize multi-GPU orchestrator.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config
        self.process_group_mgr = ProcessGroupManager(
            backend=config.backend,
            timeout_sec=config.timeout_sec
        )
        self.resource_allocator = None
        self.health_monitor = None
        self.recovery_manager = FailureRecoveryManager(
            max_retries=config.max_restart_attempts
        )
        
        self.model = None
        self.initialized = False
        self.step_count = 0
        self.inference_times = collections.deque(maxlen=100)
        
        # Setup logging
        logging.basicConfig(level=config.log_level)
    
    def initialize(self, rank: int, world_size: int,
                  master_addr: str = "localhost",
                  master_port: int = 29500) -> bool:
        """
        Initialize orchestrator for distributed inference.
        
        Args:
            rank: Process rank
            world_size: Total processes
            master_addr: Master node address
            master_port: Master node port
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize process group
            self.process_group_mgr.initialize(
                rank=rank,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port
            )
            
            # Setup resource allocator
            device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
            self.resource_allocator = ResourceAllocator(
                rank=rank,
                device=device,
                memory_fraction=self.config.memory_fraction
            )
            
            # Setup health monitor
            self.health_monitor = HealthMonitor(
                rank=rank,
                device=device,
                check_interval_sec=self.config.health_check_interval_sec
            )
            
            self.initialized = True
            logger.info(f"Rank {rank}: Orchestrator initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Rank {rank}: Initialization failed: {e}")
            return False
    
    def load_model(self, model: torch.nn.Module):
        """
        Load model for distributed inference.
        
        Args:
            model: PyTorch model to use
        """
        if not self.initialized:
            raise RuntimeError("Orchestrator not initialized")
        
        self.model = model
        logger.info(f"Rank {self.process_group_mgr.rank}: Model loaded")
    
    def inference_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute single inference step with orchestration.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Inference results
            
        Raises:
            RuntimeError: If inference fails unrecoverably
        """
        if not self.initialized or self.model is None:
            raise RuntimeError("Orchestrator not ready for inference")
        
        step_start = time.time()
        
        try:
            # Health check
            health = self.health_monitor.check_health()
            if health and health["status"] == ProcessStatus.UNHEALTHY.value:
                raise RuntimeError("GPU health check failed")
            
            # Record heartbeat
            self.health_monitor.record_heartbeat()
            
            # Model forward pass (distributed)
            with torch.no_grad():
                outputs = self.model(**batch)
            
            # Synchronize across GPUs
            if dist.is_initialized() and dist.get_world_size() > 1:
                self.process_group_mgr.barrier()
            
            # Record success
            self.recovery_manager.reset()
            self.step_count += 1
            
            step_time = time.time() - step_start
            self.inference_times.append(step_time)
            
            logger.debug(f"Inference step {self.step_count} completed in {step_time*1000:.1f}ms")
            
            return {
                "outputs": outputs,
                "step_time_ms": step_time * 1000,
                "status": "success",
            }
        
        except RuntimeError as e:
            self.health_monitor.record_error()
            logger.error(f"Inference step failed: {e}")
            
            if self.config.enable_auto_restart:
                can_recover = self.recovery_manager.handle_communication_timeout(
                    self.process_group_mgr.rank
                )
                if can_recover:
                    logger.info("Recovery initiated")
                    return {"status": "recovering"}
            
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        health_summary = self.health_monitor.get_health_summary()
        failure_summary = self.recovery_manager.get_failure_summary()
        
        if self.inference_times:
            avg_time = sum(self.inference_times) / len(self.inference_times)
        else:
            avg_time = 0
        
        return {
            "rank": self.process_group_mgr.rank,
            "world_size": self.process_group_mgr.world_size,
            "steps_completed": self.step_count,
            "avg_inference_time_ms": avg_time * 1000,
            "health": health_summary,
            "failures": failure_summary,
            "memory_stats": self.resource_allocator.get_memory_stats() if self.resource_allocator else {},
        }
    
    def cleanup(self):
        """Cleanup and shutdown orchestrator."""
        if self.initialized:
            logger.info(f"Rank {self.process_group_mgr.rank}: Cleaning up")
            
            if self.resource_allocator:
                self.resource_allocator.reset()
            
            self.process_group_mgr.finalize()
            self.initialized = False
            
            stats = self.get_stats()
            logger.info(f"Final stats: {stats}")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "OrchestratorConfig",
    "ProcessGroupManager",
    "ResourceAllocator",
    "HealthMonitor",
    "FailureRecoveryManager",
    "MultiGPUOrchestrator",
    "ProcessStatus",
    "FailureMode",
]
