# Task 1.1.3: Multi-GPU Orchestrator Components Design

**Document Type**: Architecture Design  
**Task ID**: 1.1.3  
**Assigned To**: @APEX, @ARCHITECT  
**Status**: IN PROGRESS  
**Duration**: 2 days (Jan 1-2, 2026)  
**Priority**: CRITICAL

---

## Executive Summary

This document defines the multi-GPU orchestrator architecture that manages distributed inference across multiple GPUs. The orchestrator handles process coordination, resource allocation, health monitoring, and failure recovery, providing a clean abstraction layer above the tensor parallelism primitives.

---

## 1. Orchestrator Architecture Overview

### 1.1 Core Responsibilities

```
┌────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                               │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PROCESS MANAGEMENT                                         │
│     ├─ Process group creation (per-GPU)                        │
│     ├─ Rank assignment and initialization                      │
│     ├─ Barrier synchronization                                 │
│     └─ Process shutdown and cleanup                            │
│                                                                 │
│  2. RESOURCE ALLOCATION                                        │
│     ├─ GPU assignment policy                                   │
│     ├─ Memory reservation per GPU                              │
│     ├─ Buffer pool management                                  │
│     └─ Dynamic allocation/deallocation                         │
│                                                                 │
│  3. COMMUNICATION INITIALIZATION                               │
│     ├─ NCCL rank configuration                                 │
│     ├─ Communication group setup                               │
│     ├─ Backend initialization (NCCL/gloo)                      │
│     └─ Timeout configuration                                   │
│                                                                 │
│  4. HEALTH MONITORING                                          │
│     ├─ Process heartbeat tracking                              │
│     ├─ GPU health checks                                       │
│     ├─ Memory utilization monitoring                           │
│     └─ Communication latency tracking                          │
│                                                                 │
│  5. FAILURE DETECTION & RECOVERY                               │
│     ├─ Stale process detection                                 │
│     ├─ Automatic restart triggers                              │
│     ├─ Graceful degradation                                    │
│     └─ Checkpoint/recovery mechanism                           │
│                                                                 │
│  6. INFERENCE COORDINATION                                     │
│     ├─ Request batching across GPUs                            │
│     ├─ Load balancing                                          │
│     ├─ Pipeline orchestration                                  │
│     └─ Result aggregation                                      │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture

### 2.1 ProcessGroupManager

**Purpose**: Manages PyTorch distributed process groups and rank assignments

```python
class ProcessGroupManager:
    """
    Manages initialization and lifecycle of PyTorch distributed groups.

    Responsibilities:
    - Rank and world size determination
    - Backend selection (NCCL for GPU, gloo for CPU)
    - Process group creation
    - Barrier synchronization
    """

    def __init__(self, backend: str = "nccl", timeout: float = 30.0):
        """
        Args:
            backend: "nccl" (GPU) or "gloo" (CPU)
            timeout: Communication timeout in seconds
        """
        self.backend = backend
        self.timeout = timeout
        self.rank = None
        self.world_size = None
        self.device = None

    def initialize(self, rank: int, world_size: int,
                  master_addr: str, master_port: int):
        """
        Initialize distributed process group.

        Args:
            rank: Process rank (0 to world_size-1)
            world_size: Total number of processes
            master_addr: Master node address
            master_port: Master node port
        """
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")

        dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://{master_addr}:{master_port}",
            timeout=timedelta(seconds=self.timeout)
        )

    def barrier(self):
        """Synchronize all processes."""
        if dist.is_initialized():
            dist.barrier()

    def broadcast_object(self, obj: Any, src_rank: int = 0):
        """Broadcast Python object from src_rank to all."""
        obj_list = [obj] if self.rank == src_rank else [None]
        dist.broadcast_object_list(obj_list, src=src_rank)
        return obj_list[0]

    def finalize(self):
        """Cleanup and shutdown process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
```

### 2.2 ResourceAllocator

**Purpose**: Manages GPU memory and tensor buffer allocation

```python
class ResourceAllocator:
    """
    Allocates and manages GPU resources for distributed inference.

    Features:
    - Per-GPU memory budgeting
    - Automatic buffer pooling
    - Memory fragmentation tracking
    - Dynamic allocation/deallocation
    """

    def __init__(self, rank: int, device: torch.device,
                 memory_fraction: float = 0.9):
        """
        Args:
            rank: GPU rank
            device: CUDA device
            memory_fraction: Fraction of GPU memory to use (safety margin)
        """
        self.rank = rank
        self.device = device
        self.memory_fraction = memory_fraction

        # Get available memory
        props = torch.cuda.get_device_properties(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        self.max_memory = int(total_memory * memory_fraction)
        self.allocated = 0

    def allocate_tensor(self, shape: Tuple[int, ...],
                       dtype: torch.dtype) -> torch.Tensor:
        """
        Allocate tensor with bounds checking.

        Args:
            shape: Tensor shape
            dtype: Tensor data type

        Returns:
            Allocated tensor on device

        Raises:
            RuntimeError: If allocation would exceed memory budget
        """
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        required = tensor.numel() * dtype.itemsize

        if self.allocated + required > self.max_memory:
            raise RuntimeError(
                f"GPU {self.rank}: Allocation would exceed memory "
                f"({self.allocated + required} > {self.max_memory})"
            )

        self.allocated += required
        return tensor

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage stats."""
        torch.cuda.synchronize(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        allocated = torch.cuda.memory_allocated(self.device)

        return {
            "allocated_mb": allocated / 1024 / 1024,
            "reserved_mb": reserved / 1024 / 1024,
            "max_budget_mb": self.max_memory / 1024 / 1024,
            "utilization_percent": (allocated / reserved) * 100 if reserved > 0 else 0
        }

    def reset(self):
        """Clear all allocations."""
        torch.cuda.empty_cache()
        self.allocated = 0
```

### 2.3 HealthMonitor

**Purpose**: Track process and GPU health, detect failures

```python
class HealthMonitor:
    """
    Monitors GPU and process health during distributed inference.

    Metrics:
    - GPU temperature, clock rate, power draw
    - Memory utilization trends
    - All-reduce latency and bandwidth
    - Process responsiveness (heartbeat)
    """

    def __init__(self, rank: int, device: torch.device,
                 check_interval_sec: float = 5.0):
        self.rank = rank
        self.device = device
        self.check_interval = check_interval_sec
        self.last_check = time.time()

        # Metrics storage (rolling window)
        self.metrics_history = collections.deque(maxlen=100)

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
        torch.cuda.synchronize(self.device)
        mem_info = torch.cuda.memory_stats(self.device)
        props = torch.cuda.get_device_properties(self.device)

        metrics = {
            "timestamp": now,
            "rank": self.rank,
            "memory_allocated_mb": torch.cuda.memory_allocated(self.device) / 1024 / 1024,
            "memory_reserved_mb": torch.cuda.memory_reserved(self.device) / 1024 / 1024,
            "gpu_utilization_percent": self._get_gpu_utilization(),
            "processes_healthy": True,
        }

        self.metrics_history.append(metrics)
        return metrics

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of recent health metrics."""
        if not self.metrics_history:
            return {}

        recent = list(self.metrics_history)[-10:]
        mem_allocated = [m["memory_allocated_mb"] for m in recent]

        return {
            "avg_memory_mb": sum(mem_allocated) / len(mem_allocated),
            "peak_memory_mb": max(mem_allocated),
            "status": "healthy" if all(m["processes_healthy"] for m in recent) else "unhealthy"
        }

    def _get_gpu_utilization(self) -> float:
        """Get approximate GPU utilization (requires nvidia-ml-py)."""
        try:
            from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetHandleByIndex
            handle = nvmlDeviceGetHandleByIndex(self.device.index)
            util = nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return -1.0  # Unavailable
```

### 2.4 MultiGPUOrchestrator

**Purpose**: Main orchestrator coordinating all components

```python
class MultiGPUOrchestrator:
    """
    Main orchestrator for distributed inference across multiple GPUs.

    Coordinates:
    - Process initialization and cleanup
    - Resource allocation
    - Health monitoring
    - Model loading and distribution
    - Request batching and load balancing
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.process_group_mgr = ProcessGroupManager(
            backend=config.backend,
            timeout=config.timeout_sec
        )
        self.resource_allocator = None
        self.health_monitor = None
        self.model = None
        self.initialized = False

    def initialize(self, rank: int, world_size: int,
                  master_addr: str, master_port: int):
        """
        Initialize orchestrator for distributed training.

        Args:
            rank: Process rank
            world_size: Total processes
            master_addr: Master node address
            master_port: Master node port
        """
        # Initialize process group
        self.process_group_mgr.initialize(
            rank=rank,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port
        )

        # Setup resource allocator
        device = torch.device(f"cuda:{rank}")
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

    def load_distributed_model(self, model_name: str):
        """
        Load model with weights distributed across GPUs.

        Args:
            model_name: Model identifier
        """
        if not self.initialized:
            raise RuntimeError("Orchestrator not initialized")

        # Model loading delegated to ModelLoader (task 1.1.6)
        # Orchestrator ensures proper GPU placement and synchronization
        pass

    def run_inference_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Run single inference step with orchestration.

        Args:
            batch: Input batch dictionary

        Returns:
            Inference results
        """
        # Health check
        health = self.health_monitor.check_health()
        if health and not health.get("processes_healthy", True):
            raise RuntimeError("GPU health check failed")

        # Model forward pass (distributed)
        with torch.no_grad():
            outputs = self.model(**batch)

        # Synchronize across GPUs
        self.process_group_mgr.barrier()

        return outputs

    def cleanup(self):
        """Cleanup and shutdown."""
        if self.initialized:
            self.process_group_mgr.finalize()
            self.initialized = False
```

### 2.5 OrchestratorConfig

```python
@dataclass
class OrchestratorConfig:
    """Configuration for multi-GPU orchestrator."""

    # Process group settings
    backend: str = "nccl"  # "nccl" or "gloo"
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
```

---

## 3. Orchestration Flow

### 3.1 Startup Sequence

```
1. RANK 0 PROCESS
   └─ Start training harness
      ├─ Parse config and arguments
      ├─ Determine world_size from GPU count
      ├─ Start N worker processes (one per GPU)
      └─ Wait for all to connect

2. WORKER PROCESSES (RANKS 1 to N)
   └─ For each rank in 1..N:
      ├─ Wait for RANK 0 signal
      ├─ Initialize ProcessGroupManager
      ├─ Initialize ResourceAllocator
      ├─ Initialize HealthMonitor
      ├─ Wait at barrier for all ranks
      └─ Ready for inference

3. SYNCHRONIZATION
   └─ All ranks at barrier
      └─ Begin distributed inference
```

### 3.2 Inference Execution Sequence

```
┌─ BATCH ARRIVES ─┐
│                 ↓
│         INPUT VALIDATION
│         (All ranks check consistency)
│                 ↓
│         HEALTH CHECK
│         (Monitor: GPU health OK?)
│                 ↓
│         RESOURCE ALLOCATION
│         (Allocate buffers on each GPU)
│                 ↓
├─ LOAD BALANCING ─┤
│ (Orchestrator    │
│  routes to       │ LOAD DISTRIBUTION
│  least busy GPU) │ (Scatter inputs to GPUs)
│                 ↓
│        DISTRIBUTED MODEL FORWARD
│        (Tensor parallel execution)
│                 ↓
│        OUTPUT AGGREGATION
│        (Gather results from all GPUs)
│                 ↓
│        BARRIER SYNCHRONIZATION
│        (Wait for all GPUs to complete)
│                 ↓
└─ RETURN RESULTS ─┘
```

---

## 4. Error Handling & Recovery

### 4.1 Failure Modes

| Failure Mode          | Detection              | Recovery                       |
| --------------------- | ---------------------- | ------------------------------ |
| GPU OOM               | Exception during alloc | Clear cache, reduce batch size |
| Communication timeout | Timeout exception      | Restart failed rank            |
| Stale process         | Heartbeat miss         | Automatic restart              |
| GPU failure           | Health check fail      | Degrade to fewer GPUs          |
| All-reduce error      | Collective error       | Checkpoint and restart         |

### 4.2 Recovery Strategy

```python
class FailureRecoveryManager:
    """
    Handles detection and recovery from various failures.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.failure_count = 0

    def handle_gpu_oom(self, rank: int):
        """Handle out-of-memory on GPU."""
        logger.warning(f"GPU {rank}: Out of memory")

        # Clear cache
        torch.cuda.empty_cache()

        # Reduce batch size
        self.batch_size = max(1, self.batch_size // 2)

        logger.info(f"Reduced batch size to {self.batch_size}")

    def handle_communication_timeout(self, rank: int):
        """Handle communication timeout."""
        logger.error(f"Rank {rank}: Communication timeout")

        if self.failure_count < self.max_retries:
            self.failure_count += 1
            logger.info(f"Attempting recovery ({self.failure_count}/{self.max_retries})")
            # Restart rank
        else:
            raise RuntimeError("Max recovery attempts exceeded")

    def reset(self):
        """Reset failure counter after successful step."""
        self.failure_count = 0
```

---

## 5. Monitoring & Metrics

### 5.1 Exported Metrics

```python
# Prometheus-style metrics
distributed_inference_throughput = Counter(
    "distributed_inference_steps_total",
    "Total inference steps completed"
)

distributed_inference_latency = Histogram(
    "distributed_inference_step_seconds",
    "Latency of single inference step"
)

distributed_communication_latency = Histogram(
    "distributed_communication_latency_seconds",
    "Latency of all-reduce operations"
)

gpu_memory_usage = Gauge(
    "gpu_memory_usage_bytes",
    "GPU memory usage by rank",
    ["rank"]
)

distributed_scaling_efficiency = Gauge(
    "distributed_scaling_efficiency_percent",
    "Efficiency relative to single GPU"
)
```

### 5.2 Observability Integration

- **Logging**: Structured logging with rank information
- **Tracing**: Distributed tracing of inference flow
- **Metrics**: Prometheus metrics export
- **Profiling**: Optional detailed profiling with PyTorch Profiler

---

## 6. Design Decisions

### Decision 1: Synchronous All-reduce

**Rationale**: Simpler to implement and debug than async, sufficient for inference workloads

### Decision 2: Per-GPU Process Model

**Rationale**: Cleaner separation of concerns, easier failure isolation

### Decision 3: Explicit Barrier Synchronization

**Rationale**: Ensures deterministic behavior, simplifies debugging

### Decision 4: Health Monitoring

**Rationale**: Proactive failure detection enables fast recovery

---

## 7. Integration with Phase 3 Components

```
Task 1.1.3 (This Design) ← You are here
    ↓ Uses
Task 1.1.2: Tensor Parallelism (for computation model)
    ↓ Coordinates with
Task 1.1.4: NCCL Backend (for communication)
    ↓ Enables
Task 1.1.5: Tensor Parallel Implementation
    ↓ Manages
Task 1.1.6: Model Loading
```

---

## 8. Validation Criteria

- [ ] Orchestrator successfully initializes with 2 GPUs
- [ ] Process group creation verified via barriers
- [ ] Resource allocation enforces memory bounds
- [ ] Health monitoring produces valid metrics
- [ ] Error recovery works for OOM scenario
- [ ] Distributed model forward pass executes
- [ ] Metrics exported correctly

---

## 9. Next Steps

1. **Task 1.1.4**: Select and configure NCCL communication backend (depends on this design)
2. **Task 1.1.5**: Implement orchestrator (implementation phase)
3. **Task 1.1.6**: Implement model loader integration

---

## References

- PyTorch distributed documentation
- Horovod distributed training orchestration
- NVIDIA Collective Communications Library (NCCL)
- Google Borg resource management
