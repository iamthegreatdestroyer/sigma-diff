"""
GPU Coordinator for Multi-GPU Inference Orchestration.

This module provides centralized coordination for multi-GPU inference workloads:
- Device discovery and topology detection
- Health monitoring and heartbeat tracking
- Workload balancing and scheduling
- Failure detection and recovery
- Resource allocation and management

Integration Points:
- Works with tensor_parallelism.py for model distribution
- Works with pipeline_parallelism.py for pipeline orchestration
- Works with multi_gpu_cache.py for cache coordination

Author: Ryzanstein Team
Date: 2025
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist
import torch.cuda as cuda

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class GPUState(Enum):
    """State of a GPU device in the cluster."""
    UNKNOWN = auto()
    IDLE = auto()           # GPU is available and idle
    BUSY = auto()           # GPU is actively processing
    INITIALIZING = auto()
    HEALTHY = auto()
    DEGRADED = auto()
    OVERLOADED = auto()
    FAILING = auto()
    FAILED = auto()
    RECOVERING = auto()
    MAINTENANCE = auto()


class WorkloadPriority(Enum):
    """Priority levels for workload scheduling."""
    CRITICAL = 0      # Real-time, latency-sensitive
    HIGH = 1          # Interactive requests
    NORMAL = 2        # Standard batch processing
    LOW = 3           # Background tasks
    PREEMPTIBLE = 4   # Can be interrupted


class SchedulingPolicy(Enum):
    """Workload scheduling policies."""
    ROUND_ROBIN = auto()           # Simple rotation
    LEAST_LOADED = auto()          # Send to least busy GPU
    MEMORY_AWARE = auto()          # Consider memory availability
    AFFINITY_BASED = auto()        # Prefer same GPU for related work
    POWER_EFFICIENT = auto()       # Minimize active GPUs
    LATENCY_OPTIMIZED = auto()     # Minimize queue depth
    THROUGHPUT_OPTIMIZED = auto()  # Maximize utilization


class TopologyType(Enum):
    """GPU interconnect topology types."""
    UNKNOWN = auto()
    NVLINK = auto()           # NVLink connections
    PCIE = auto()             # PCIe connections
    NVSWITCH = auto()         # NVSwitch fabric
    MIXED = auto()            # Combination
    RDMA = auto()             # Remote GPUs via RDMA


class FailureType(Enum):
    """Types of GPU failures."""
    NONE = auto()
    MEMORY_ERROR = auto()
    COMPUTE_ERROR = auto()
    DRIVER_ERROR = auto()
    THERMAL_THROTTLE = auto()
    POWER_LIMIT = auto()
    HANG_DETECTED = auto()
    COMMUNICATION_FAILURE = auto()
    OUT_OF_MEMORY = auto()


class RecoveryAction(Enum):
    """Recovery actions for failed GPUs."""
    NONE = auto()
    RETRY = auto()
    SOFT_RESET = auto()
    HARD_RESET = auto()
    REASSIGN_WORK = auto()
    ISOLATE = auto()
    ESCALATE = auto()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class HealthMonitorConfig:
    """Configuration for GPU health monitoring."""
    # Heartbeat settings
    heartbeat_interval_ms: float = 100.0
    heartbeat_timeout_ms: float = 1000.0
    missed_heartbeats_threshold: int = 3
    
    # Resource thresholds
    memory_warning_threshold: float = 0.85
    memory_critical_threshold: float = 0.95
    utilization_high_threshold: float = 0.90
    utilization_critical_threshold: float = 0.98
    temperature_warning_c: float = 80.0
    temperature_critical_c: float = 90.0
    
    # Error tracking
    error_count_window_s: float = 60.0
    max_errors_before_degraded: int = 5
    max_errors_before_failing: int = 10
    
    # Recovery settings
    auto_recovery_enabled: bool = True
    recovery_cooldown_s: float = 30.0
    max_recovery_attempts: int = 3


@dataclass
class SchedulerConfig:
    """Configuration for workload scheduling."""
    # Scheduling policy
    policy: SchedulingPolicy = SchedulingPolicy.LEAST_LOADED
    
    # Queue settings
    max_queue_depth_per_gpu: int = 32
    queue_timeout_ms: float = 5000.0
    preemption_enabled: bool = True
    
    # Load balancing
    load_update_interval_ms: float = 50.0
    load_smoothing_factor: float = 0.3
    rebalance_threshold: float = 0.2
    
    # Affinity
    affinity_decay_ms: float = 10000.0
    min_affinity_strength: float = 0.1
    
    # Batching
    dynamic_batching_enabled: bool = True
    max_batch_wait_ms: float = 10.0
    preferred_batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])


@dataclass
class CoordinatorConfig:
    """Configuration for GPU coordinator."""
    # Device settings
    device_ids: Optional[List[int]] = None  # None = auto-detect all
    master_device_id: int = 0
    
    # Monitoring
    health_config: HealthMonitorConfig = field(default_factory=HealthMonitorConfig)
    
    # Scheduling
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Topology
    auto_detect_topology: bool = True
    prefer_nvlink: bool = True
    
    # Distributed settings
    distributed_backend: str = "nccl"
    init_method: Optional[str] = None
    
    # Logging
    metrics_log_interval_s: float = 10.0
    verbose_logging: bool = False


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class GPUInfo:
    """Information about a single GPU device."""
    device_id: int
    device_name: str
    compute_capability: Tuple[int, int]
    total_memory_bytes: int
    
    # Dynamic state
    state: GPUState = GPUState.UNKNOWN
    available_memory_bytes: int = 0
    utilization_percent: float = 0.0
    temperature_c: float = 0.0
    power_watts: float = 0.0
    
    # Health tracking
    last_heartbeat_time: float = 0.0
    missed_heartbeats: int = 0
    error_count: int = 0
    last_error_time: float = 0.0
    failure_type: FailureType = FailureType.NONE
    
    # Workload tracking
    active_tasks: int = 0
    queued_tasks: int = 0
    completed_tasks: int = 0
    
    @property
    def memory_used_percent(self) -> float:
        """Calculate memory usage percentage."""
        if self.total_memory_bytes == 0:
            return 0.0
        used = self.total_memory_bytes - self.available_memory_bytes
        return used / self.total_memory_bytes
    
    @property
    def is_available(self) -> bool:
        """Check if GPU is available for work."""
        return self.state in (GPUState.HEALTHY, GPUState.DEGRADED)


@dataclass
class TopologyInfo:
    """GPU interconnect topology information."""
    topology_type: TopologyType
    
    # Adjacency matrix: bandwidth in GB/s between GPU pairs
    bandwidth_matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    # NVLink connections
    nvlink_connections: Dict[Tuple[int, int], int] = field(default_factory=dict)
    
    # Optimal transfer paths
    optimal_paths: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    
    def get_bandwidth(self, src_gpu: int, dst_gpu: int) -> float:
        """Get bandwidth between two GPUs in GB/s."""
        if src_gpu == dst_gpu:
            return float('inf')
        return self.bandwidth_matrix.get((src_gpu, dst_gpu), 0.0)
    
    def get_latency_rank(self, src_gpu: int, dst_gpu: int) -> int:
        """Get relative latency rank (lower is better)."""
        if src_gpu == dst_gpu:
            return 0
        nvlinks = self.nvlink_connections.get((src_gpu, dst_gpu), 0)
        if nvlinks > 0:
            return 1
        return 2  # PCIe


@dataclass
class WorkloadTask:
    """Represents a workload task to be scheduled."""
    task_id: str
    priority: WorkloadPriority
    
    # Resource requirements
    estimated_memory_bytes: int
    estimated_compute_ms: float
    
    # Execution info
    assigned_gpu: Optional[int] = None
    submit_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Affinity
    preferred_gpu: Optional[int] = None
    sequence_id: Optional[int] = None  # For affinity with existing cache
    
    # State
    is_completed: bool = False
    is_failed: bool = False
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    @property
    def wait_time_ms(self) -> float:
        """Time spent waiting in queue."""
        if self.start_time is None:
            return (time.time() - self.submit_time) * 1000
        return (self.start_time - self.submit_time) * 1000
    
    @property
    def execution_time_ms(self) -> Optional[float]:
        """Time spent executing."""
        if self.start_time is None or self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


@dataclass
class CoordinatorStatistics:
    """Statistics for the GPU coordinator."""
    # Device stats
    total_gpus: int = 0
    healthy_gpus: int = 0
    degraded_gpus: int = 0
    failed_gpus: int = 0
    
    # Workload stats
    total_tasks_submitted: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    tasks_in_queue: int = 0
    tasks_in_progress: int = 0
    
    # Timing stats
    avg_queue_wait_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Resource stats
    total_memory_bytes: int = 0
    used_memory_bytes: int = 0
    avg_utilization: float = 0.0
    
    # Health stats
    total_heartbeats: int = 0
    missed_heartbeats: int = 0
    total_recoveries: int = 0
    
    @property
    def memory_utilization(self) -> float:
        """Overall memory utilization."""
        if self.total_memory_bytes == 0:
            return 0.0
        return self.used_memory_bytes / self.total_memory_bytes


# =============================================================================
# Topology Detection
# =============================================================================


class TopologyDetector:
    """Detects GPU interconnect topology."""
    
    def __init__(self, device_ids: List[int]):
        self.device_ids = device_ids
        self._topology: Optional[TopologyInfo] = None
    
    def detect_topology(self) -> TopologyInfo:
        """Detect GPU topology and interconnect bandwidth."""
        if not cuda.is_available():
            return self._create_empty_topology()
        
        topology = TopologyInfo(topology_type=TopologyType.UNKNOWN)
        n_gpus = len(self.device_ids)
        
        # Detect NVLink connections
        has_nvlink = False
        has_pcie = False
        
        for i, src_id in enumerate(self.device_ids):
            for j, dst_id in enumerate(self.device_ids):
                if src_id == dst_id:
                    continue
                
                # Query NVLink info
                try:
                    can_access = cuda.can_device_access_peer(src_id, dst_id)
                    if can_access:
                        # Estimate bandwidth based on architecture
                        props = cuda.get_device_properties(src_id)
                        major, minor = props.major, props.minor
                        
                        # Rough bandwidth estimates
                        if major >= 8:  # Ampere+
                            bandwidth = 300.0  # GB/s per NVLink
                            nvlinks = 4
                        elif major >= 7:  # Volta/Turing
                            bandwidth = 150.0
                            nvlinks = 2
                        else:
                            bandwidth = 20.0  # PCIe fallback
                            nvlinks = 0
                        
                        if nvlinks > 0:
                            has_nvlink = True
                            topology.nvlink_connections[(src_id, dst_id)] = nvlinks
                            topology.bandwidth_matrix[(src_id, dst_id)] = bandwidth * nvlinks
                        else:
                            has_pcie = True
                            topology.bandwidth_matrix[(src_id, dst_id)] = bandwidth
                    else:
                        has_pcie = True
                        topology.bandwidth_matrix[(src_id, dst_id)] = 12.0  # PCIe 3.0 x16
                        
                except Exception as e:
                    logger.warning(f"Failed to query peer access {src_id}->{dst_id}: {e}")
                    topology.bandwidth_matrix[(src_id, dst_id)] = 12.0
        
        # Determine topology type
        if has_nvlink and has_pcie:
            topology.topology_type = TopologyType.MIXED
        elif has_nvlink:
            topology.topology_type = TopologyType.NVLINK
        elif has_pcie:
            topology.topology_type = TopologyType.PCIE
        else:
            topology.topology_type = TopologyType.UNKNOWN
        
        # Compute optimal transfer paths (Floyd-Warshall)
        self._compute_optimal_paths(topology)
        
        self._topology = topology
        return topology
    
    def _compute_optimal_paths(self, topology: TopologyInfo) -> None:
        """Compute optimal transfer paths using Floyd-Warshall."""
        n = len(self.device_ids)
        id_to_idx = {gid: i for i, gid in enumerate(self.device_ids)}
        
        # Initialize distance matrix (use 1/bandwidth as weight)
        INF = float('inf')
        dist = [[INF] * n for _ in range(n)]
        next_hop = [[None] * n for _ in range(n)]
        
        for i in range(n):
            dist[i][i] = 0
        
        for (src, dst), bw in topology.bandwidth_matrix.items():
            if bw > 0:
                i, j = id_to_idx[src], id_to_idx[dst]
                dist[i][j] = 1.0 / bw
                next_hop[i][j] = j
        
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_hop[i][j] = next_hop[i][k]
        
        # Reconstruct paths
        for src in self.device_ids:
            for dst in self.device_ids:
                if src != dst:
                    path = self._reconstruct_path(
                        id_to_idx[src],
                        id_to_idx[dst],
                        next_hop,
                        self.device_ids
                    )
                    if path:
                        topology.optimal_paths[(src, dst)] = path
    
    def _reconstruct_path(
        self,
        src_idx: int,
        dst_idx: int,
        next_hop: List[List[Optional[int]]],
        device_ids: List[int]
    ) -> List[int]:
        """Reconstruct path from next_hop matrix."""
        if next_hop[src_idx][dst_idx] is None:
            return []
        
        path = [device_ids[src_idx]]
        current = src_idx
        while current != dst_idx:
            current = next_hop[current][dst_idx]
            if current is None:
                return []
            path.append(device_ids[current])
        return path
    
    def _create_empty_topology(self) -> TopologyInfo:
        """Create empty topology for non-GPU environments."""
        return TopologyInfo(topology_type=TopologyType.UNKNOWN)


# =============================================================================
# Health Monitor
# =============================================================================


class HealthMonitor:
    """Monitors GPU health and detects failures."""
    
    def __init__(
        self,
        device_infos: Dict[int, GPUInfo],
        config: HealthMonitorConfig,
        on_state_change: Optional[Callable[[int, GPUState, GPUState], None]] = None
    ):
        self.device_infos = device_infos
        self.config = config
        self.on_state_change = on_state_change
        
        # Tracking
        self._error_history: Dict[int, List[float]] = defaultdict(list)
        self._recovery_attempts: Dict[int, int] = defaultdict(int)
        self._last_recovery_time: Dict[int, float] = defaultdict(float)
        
        # Threading
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Statistics
        self.total_heartbeats = 0
        self.missed_heartbeats = 0
    
    def start(self) -> None:
        """Start background monitoring."""
        if self._monitor_thread is not None:
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="GPUHealthMonitor"
        )
        self._monitor_thread.start()
        logger.info("GPU health monitoring started")
    
    def stop(self) -> None:
        """Stop background monitoring."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        logger.info("GPU health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        interval = self.config.heartbeat_interval_ms / 1000.0
        
        while not self._stop_event.is_set():
            try:
                self._check_all_gpus()
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
            
            self._stop_event.wait(interval)
    
    def _check_all_gpus(self) -> None:
        """Check health of all GPUs."""
        current_time = time.time()
        
        for device_id, info in self.device_infos.items():
            self._check_gpu(device_id, info, current_time)
    
    def _check_gpu(self, device_id: int, info: GPUInfo, current_time: float) -> None:
        """Check health of a single GPU."""
        old_state = info.state
        
        # Update device metrics
        try:
            self._update_device_metrics(device_id, info)
            self.total_heartbeats += 1
            info.last_heartbeat_time = current_time
            info.missed_heartbeats = 0
        except Exception as e:
            info.missed_heartbeats += 1
            self.missed_heartbeats += 1
            logger.warning(f"Failed to query GPU {device_id}: {e}")
        
        # Determine new state
        new_state = self._determine_state(info, current_time)
        
        if new_state != old_state:
            info.state = new_state
            logger.info(f"GPU {device_id} state: {old_state.name} -> {new_state.name}")
            
            if self.on_state_change:
                self.on_state_change(device_id, old_state, new_state)
            
            # Auto-recovery if enabled
            if new_state == GPUState.FAILED and self.config.auto_recovery_enabled:
                self._attempt_recovery(device_id, info)
    
    def _update_device_metrics(self, device_id: int, info: GPUInfo) -> None:
        """Update GPU metrics from CUDA."""
        if not cuda.is_available():
            return
        
        try:
            # Memory info
            with cuda.device(device_id):
                mem_info = cuda.mem_get_info()
                info.available_memory_bytes = mem_info[0]
            
            # Get device properties for utilization (if available via pynvml)
            # For now, estimate from memory pressure
            info.utilization_percent = info.memory_used_percent * 100
            
        except cuda.CudaError as e:
            self._record_error(device_id, info, FailureType.DRIVER_ERROR)
            raise
    
    def _determine_state(self, info: GPUInfo, current_time: float) -> GPUState:
        """Determine GPU state based on metrics."""
        # Check heartbeat
        if info.missed_heartbeats >= self.config.missed_heartbeats_threshold:
            info.failure_type = FailureType.HANG_DETECTED
            return GPUState.FAILED
        
        # Check errors in window
        recent_errors = self._count_recent_errors(
            info.device_id,
            current_time - self.config.error_count_window_s
        )
        
        if recent_errors >= self.config.max_errors_before_failing:
            return GPUState.FAILING
        
        if recent_errors >= self.config.max_errors_before_degraded:
            return GPUState.DEGRADED
        
        # Check resources
        if info.memory_used_percent >= self.config.memory_critical_threshold:
            return GPUState.OVERLOADED
        
        if info.memory_used_percent >= self.config.memory_warning_threshold:
            return GPUState.DEGRADED
        
        if info.utilization_percent >= self.config.utilization_critical_threshold:
            return GPUState.OVERLOADED
        
        # Check temperature
        if info.temperature_c >= self.config.temperature_critical_c:
            return GPUState.DEGRADED
        
        return GPUState.HEALTHY
    
    def _record_error(
        self,
        device_id: int,
        info: GPUInfo,
        failure_type: FailureType
    ) -> None:
        """Record an error for a GPU."""
        current_time = time.time()
        info.error_count += 1
        info.last_error_time = current_time
        info.failure_type = failure_type
        
        with self._lock:
            self._error_history[device_id].append(current_time)
            
            # Prune old errors
            cutoff = current_time - self.config.error_count_window_s
            self._error_history[device_id] = [
                t for t in self._error_history[device_id] if t > cutoff
            ]
    
    def _count_recent_errors(self, device_id: int, since: float) -> int:
        """Count errors since a given time."""
        with self._lock:
            return sum(
                1 for t in self._error_history.get(device_id, [])
                if t >= since
            )
    
    def _attempt_recovery(self, device_id: int, info: GPUInfo) -> None:
        """Attempt to recover a failed GPU."""
        current_time = time.time()
        
        # Check cooldown
        last_recovery = self._last_recovery_time.get(device_id, 0)
        if current_time - last_recovery < self.config.recovery_cooldown_s:
            return
        
        # Check max attempts
        attempts = self._recovery_attempts.get(device_id, 0)
        if attempts >= self.config.max_recovery_attempts:
            logger.error(f"GPU {device_id} exceeded max recovery attempts")
            return
        
        logger.info(f"Attempting recovery for GPU {device_id} (attempt {attempts + 1})")
        
        info.state = GPUState.RECOVERING
        self._recovery_attempts[device_id] = attempts + 1
        self._last_recovery_time[device_id] = current_time
        
        try:
            # Attempt soft reset
            with cuda.device(device_id):
                cuda.empty_cache()
                cuda.synchronize()
            
            # Clear error history
            with self._lock:
                self._error_history[device_id].clear()
            
            info.error_count = 0
            info.failure_type = FailureType.NONE
            info.state = GPUState.HEALTHY
            logger.info(f"GPU {device_id} recovery successful")
            
        except Exception as e:
            logger.error(f"GPU {device_id} recovery failed: {e}")
            info.state = GPUState.FAILED


# =============================================================================
# Workload Scheduler
# =============================================================================


class WorkloadScheduler:
    """Schedules workloads across GPUs."""
    
    def __init__(
        self,
        device_infos: Dict[int, GPUInfo],
        topology: TopologyInfo,
        config: SchedulerConfig
    ):
        self.device_infos = device_infos
        self.topology = topology
        self.config = config
        
        # Task queues
        self._task_queues: Dict[int, List[WorkloadTask]] = {
            gid: [] for gid in device_infos
        }
        self._pending_tasks: List[WorkloadTask] = []
        self._active_tasks: Dict[str, WorkloadTask] = {}
        
        # Load tracking
        self._gpu_loads: Dict[int, float] = {gid: 0.0 for gid in device_infos}
        
        # Affinity tracking
        self._sequence_affinity: Dict[int, Tuple[int, float]] = {}  # seq_id -> (gpu, strength)
        
        # Statistics
        self._latencies: List[float] = []
        self._max_latency_samples = 1000
        
        # Threading
        self._lock = threading.Lock()
    
    def submit_task(self, task: WorkloadTask) -> bool:
        """Submit a task for scheduling."""
        with self._lock:
            # Find best GPU
            assigned_gpu = self._select_gpu(task)
            
            if assigned_gpu is None:
                # Queue for later
                self._pending_tasks.append(task)
                return False
            
            task.assigned_gpu = assigned_gpu
            queue = self._task_queues[assigned_gpu]
            
            if len(queue) >= self.config.max_queue_depth_per_gpu:
                if self.config.preemption_enabled and task.priority.value < WorkloadPriority.NORMAL.value:
                    # Preempt lower priority task
                    self._preempt_task(assigned_gpu, task)
                else:
                    self._pending_tasks.append(task)
                    return False
            
            # Insert by priority
            insert_idx = 0
            for i, queued_task in enumerate(queue):
                if task.priority.value < queued_task.priority.value:
                    break
                insert_idx = i + 1
            
            queue.insert(insert_idx, task)
            self.device_infos[assigned_gpu].queued_tasks += 1
            
            # Update affinity
            if task.sequence_id is not None:
                self._update_affinity(task.sequence_id, assigned_gpu)
            
            return True
    
    def get_next_task(self, device_id: int) -> Optional[WorkloadTask]:
        """Get next task for a device to execute."""
        with self._lock:
            queue = self._task_queues.get(device_id)
            if not queue:
                return None
            
            task = queue.pop(0)
            task.start_time = time.time()
            self._active_tasks[task.task_id] = task
            
            info = self.device_infos[device_id]
            info.queued_tasks -= 1
            info.active_tasks += 1
            
            return task
    
    def complete_task(self, task_id: str, success: bool, error: Optional[str] = None) -> None:
        """Mark a task as completed."""
        with self._lock:
            task = self._active_tasks.pop(task_id, None)
            if task is None:
                return
            
            task.end_time = time.time()
            task.is_completed = success
            task.is_failed = not success
            task.error_message = error
            
            if task.assigned_gpu is not None:
                info = self.device_infos[task.assigned_gpu]
                info.active_tasks -= 1
                if success:
                    info.completed_tasks += 1
                
                # Update load estimate
                self._update_load(task.assigned_gpu)
            
            # Track latency
            if task.execution_time_ms is not None:
                total_latency = task.wait_time_ms + task.execution_time_ms
                self._latencies.append(total_latency)
                if len(self._latencies) > self._max_latency_samples:
                    self._latencies.pop(0)
            
            # Retry failed tasks
            if not success and task.retry_count < task.max_retries:
                task.retry_count += 1
                task.start_time = None
                task.end_time = None
                task.is_failed = False
                self.submit_task(task)
    
    def _select_gpu(self, task: WorkloadTask) -> Optional[int]:
        """Select best GPU for a task."""
        policy = self.config.policy
        
        # Filter available GPUs
        available = [
            gid for gid, info in self.device_infos.items()
            if info.is_available and info.available_memory_bytes >= task.estimated_memory_bytes
        ]
        
        if not available:
            return None
        
        # Check affinity first
        if task.sequence_id is not None and task.sequence_id in self._sequence_affinity:
            affinity_gpu, strength = self._sequence_affinity[task.sequence_id]
            if affinity_gpu in available and strength >= self.config.min_affinity_strength:
                return affinity_gpu
        
        if task.preferred_gpu is not None and task.preferred_gpu in available:
            return task.preferred_gpu
        
        # Apply scheduling policy
        if policy == SchedulingPolicy.ROUND_ROBIN:
            return self._select_round_robin(available)
        elif policy == SchedulingPolicy.LEAST_LOADED:
            return self._select_least_loaded(available)
        elif policy == SchedulingPolicy.MEMORY_AWARE:
            return self._select_memory_aware(available, task)
        elif policy == SchedulingPolicy.LATENCY_OPTIMIZED:
            return self._select_latency_optimized(available)
        elif policy == SchedulingPolicy.THROUGHPUT_OPTIMIZED:
            return self._select_throughput_optimized(available)
        else:
            return self._select_least_loaded(available)
    
    def _select_round_robin(self, available: List[int]) -> int:
        """Round-robin selection."""
        # Select GPU with least total tasks scheduled
        return min(
            available,
            key=lambda g: self.device_infos[g].completed_tasks
        )
    
    def _select_least_loaded(self, available: List[int]) -> int:
        """Select least loaded GPU."""
        return min(
            available,
            key=lambda g: self._gpu_loads.get(g, 0.0)
        )
    
    def _select_memory_aware(self, available: List[int], task: WorkloadTask) -> int:
        """Select GPU with most available memory."""
        return max(
            available,
            key=lambda g: self.device_infos[g].available_memory_bytes
        )
    
    def _select_latency_optimized(self, available: List[int]) -> int:
        """Select GPU with shortest queue."""
        return min(
            available,
            key=lambda g: len(self._task_queues[g])
        )
    
    def _select_throughput_optimized(self, available: List[int]) -> int:
        """Select GPU to maximize throughput."""
        # Prefer GPUs that aren't too loaded but have some work
        def score(gid: int) -> float:
            load = self._gpu_loads.get(gid, 0.0)
            if load < 0.5:
                return load  # Underutilized
            elif load < 0.85:
                return 0.5  # Sweet spot
            else:
                return 1.0 - load  # Overloaded
        
        return max(available, key=score)
    
    def _update_load(self, device_id: int) -> None:
        """Update load estimate for a GPU."""
        info = self.device_infos[device_id]
        queue_len = len(self._task_queues[device_id])
        
        # Combine active tasks and queue depth
        new_load = (info.active_tasks + queue_len * 0.5) / max(
            self.config.max_queue_depth_per_gpu, 1
        )
        
        # Exponential smoothing
        alpha = self.config.load_smoothing_factor
        self._gpu_loads[device_id] = alpha * new_load + (1 - alpha) * self._gpu_loads.get(device_id, 0)
    
    def _update_affinity(self, sequence_id: int, gpu_id: int) -> None:
        """Update sequence affinity to GPU."""
        current_time = time.time()
        
        if sequence_id in self._sequence_affinity:
            old_gpu, old_strength = self._sequence_affinity[sequence_id]
            if old_gpu == gpu_id:
                # Strengthen affinity
                new_strength = min(1.0, old_strength + 0.1)
            else:
                # Weaken and maybe switch
                new_strength = 0.5
                gpu_id = gpu_id if old_strength < 0.5 else old_gpu
        else:
            new_strength = 0.5
        
        self._sequence_affinity[sequence_id] = (gpu_id, new_strength)
    
    def _preempt_task(self, device_id: int, high_priority_task: WorkloadTask) -> None:
        """Preempt a lower priority task."""
        queue = self._task_queues[device_id]
        
        # Find lowest priority task
        if not queue:
            return
        
        lowest_idx = -1
        lowest_priority = WorkloadPriority.CRITICAL
        
        for i, task in enumerate(queue):
            if task.priority.value > lowest_priority.value:
                lowest_priority = task.priority
                lowest_idx = i
        
        if lowest_idx >= 0 and lowest_priority.value > high_priority_task.priority.value:
            preempted = queue.pop(lowest_idx)
            self._pending_tasks.append(preempted)
            queue.insert(0, high_priority_task)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self._lock:
            sorted_latencies = sorted(self._latencies) if self._latencies else [0.0]
            p50_idx = len(sorted_latencies) // 2
            p99_idx = int(len(sorted_latencies) * 0.99)
            
            return {
                "pending_tasks": len(self._pending_tasks),
                "active_tasks": len(self._active_tasks),
                "queued_tasks": sum(len(q) for q in self._task_queues.values()),
                "avg_latency_ms": sum(self._latencies) / max(len(self._latencies), 1),
                "p50_latency_ms": sorted_latencies[p50_idx],
                "p99_latency_ms": sorted_latencies[p99_idx],
                "gpu_loads": dict(self._gpu_loads),
            }


# =============================================================================
# GPU Coordinator
# =============================================================================


class GPUCoordinator:
    """
    Central coordinator for multi-GPU inference.
    
    Features:
    - Device discovery and topology detection
    - Health monitoring with automatic recovery
    - Workload scheduling with multiple policies
    - Statistics and metrics collection
    """
    
    def __init__(self, config: Optional[CoordinatorConfig] = None):
        """
        Initialize GPU coordinator.
        
        Args:
            config: Coordinator configuration
        """
        self.config = config or CoordinatorConfig()
        
        # Device management
        self.device_infos: Dict[int, GPUInfo] = {}
        self.topology: Optional[TopologyInfo] = None
        
        # Components
        self.health_monitor: Optional[HealthMonitor] = None
        self.scheduler: Optional[WorkloadScheduler] = None
        self.topology_detector: Optional[TopologyDetector] = None
        
        # State
        self._initialized = False
        self._running = False
        self._lock = threading.Lock()
        
        # Statistics
        self._statistics = CoordinatorStatistics()
        self._start_time: Optional[float] = None
        
        # Callbacks
        self._state_change_callbacks: List[Callable[[int, GPUState, GPUState], None]] = []
    
    def initialize(self) -> None:
        """Initialize coordinator and discover GPUs."""
        if self._initialized:
            return
        
        logger.info("Initializing GPU coordinator")
        
        # Discover devices
        device_ids = self._discover_devices()
        if not device_ids:
            logger.warning("No GPUs discovered")
            self._initialized = True
            return
        
        logger.info(f"Discovered {len(device_ids)} GPU(s): {device_ids}")
        
        # Initialize device infos
        for device_id in device_ids:
            info = self._create_device_info(device_id)
            self.device_infos[device_id] = info
        
        # Detect topology
        self.topology_detector = TopologyDetector(device_ids)
        self.topology = self.topology_detector.detect_topology()
        logger.info(f"Detected topology: {self.topology.topology_type.name}")
        
        # Initialize health monitor
        self.health_monitor = HealthMonitor(
            device_infos=self.device_infos,
            config=self.config.health_config,
            on_state_change=self._on_gpu_state_change
        )
        
        # Initialize scheduler
        self.scheduler = WorkloadScheduler(
            device_infos=self.device_infos,
            topology=self.topology,
            config=self.config.scheduler_config
        )
        
        # Update statistics
        self._statistics.total_gpus = len(device_ids)
        self._statistics.total_memory_bytes = sum(
            info.total_memory_bytes for info in self.device_infos.values()
        )
        
        self._initialized = True
        logger.info("GPU coordinator initialized")
    
    def start(self) -> None:
        """Start coordinator services."""
        if not self._initialized:
            self.initialize()
        
        if self._running:
            return
        
        self._running = True
        self._start_time = time.time()
        
        if self.health_monitor:
            self.health_monitor.start()
        
        logger.info("GPU coordinator started")
    
    def stop(self) -> None:
        """Stop coordinator services."""
        if not self._running:
            return
        
        self._running = False
        
        if self.health_monitor:
            self.health_monitor.stop()
        
        logger.info("GPU coordinator stopped")
    
    def _discover_devices(self) -> List[int]:
        """Discover available GPU devices."""
        if not cuda.is_available():
            return []
        
        if self.config.device_ids is not None:
            return self.config.device_ids
        
        return list(range(cuda.device_count()))
    
    def _create_device_info(self, device_id: int) -> GPUInfo:
        """Create device info for a GPU."""
        if not cuda.is_available():
            return GPUInfo(
                device_id=device_id,
                device_name="Unknown",
                compute_capability=(0, 0),
                total_memory_bytes=0,
                state=GPUState.UNKNOWN
            )
        
        props = cuda.get_device_properties(device_id)
        mem_info = cuda.mem_get_info(device_id)
        
        return GPUInfo(
            device_id=device_id,
            device_name=props.name,
            compute_capability=(props.major, props.minor),
            total_memory_bytes=props.total_memory,
            available_memory_bytes=mem_info[0],
            state=GPUState.INITIALIZING
        )
    
    def _on_gpu_state_change(
        self,
        device_id: int,
        old_state: GPUState,
        new_state: GPUState
    ) -> None:
        """Handle GPU state changes."""
        # Update statistics
        if new_state == GPUState.HEALTHY:
            self._statistics.healthy_gpus += 1
            if old_state == GPUState.DEGRADED:
                self._statistics.degraded_gpus -= 1
            elif old_state == GPUState.FAILED:
                self._statistics.failed_gpus -= 1
                self._statistics.total_recoveries += 1
        elif new_state == GPUState.DEGRADED:
            self._statistics.degraded_gpus += 1
            if old_state == GPUState.HEALTHY:
                self._statistics.healthy_gpus -= 1
        elif new_state == GPUState.FAILED:
            self._statistics.failed_gpus += 1
            if old_state == GPUState.HEALTHY:
                self._statistics.healthy_gpus -= 1
            elif old_state == GPUState.DEGRADED:
                self._statistics.degraded_gpus -= 1
        
        # Notify callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(device_id, old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    def register_state_callback(
        self,
        callback: Callable[[int, GPUState, GPUState], None]
    ) -> None:
        """Register callback for GPU state changes."""
        self._state_change_callbacks.append(callback)
    
    # -------------------------------------------------------------------------
    # Task Management
    # -------------------------------------------------------------------------
    
    def submit_task(
        self,
        task_id: str,
        estimated_memory_bytes: int = 0,
        estimated_compute_ms: float = 0.0,
        priority: WorkloadPriority = WorkloadPriority.NORMAL,
        preferred_gpu: Optional[int] = None,
        sequence_id: Optional[int] = None
    ) -> bool:
        """
        Submit a task for scheduling.
        
        Args:
            task_id: Unique task identifier
            estimated_memory_bytes: Estimated memory requirement
            estimated_compute_ms: Estimated compute time
            priority: Task priority
            preferred_gpu: Preferred GPU device ID
            sequence_id: Sequence ID for affinity
            
        Returns:
            True if task was immediately scheduled
        """
        if self.scheduler is None:
            raise RuntimeError("Coordinator not initialized")
        
        task = WorkloadTask(
            task_id=task_id,
            priority=priority,
            estimated_memory_bytes=estimated_memory_bytes,
            estimated_compute_ms=estimated_compute_ms,
            preferred_gpu=preferred_gpu,
            sequence_id=sequence_id
        )
        
        self._statistics.total_tasks_submitted += 1
        return self.scheduler.submit_task(task)
    
    def get_next_task(self, device_id: int) -> Optional[WorkloadTask]:
        """Get next task for a device."""
        if self.scheduler is None:
            return None
        return self.scheduler.get_next_task(device_id)
    
    def complete_task(
        self,
        task_id: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Mark a task as completed."""
        if self.scheduler is None:
            return
        
        if success:
            self._statistics.total_tasks_completed += 1
        else:
            self._statistics.total_tasks_failed += 1
        
        self.scheduler.complete_task(task_id, success, error)
    
    # -------------------------------------------------------------------------
    # Device Queries
    # -------------------------------------------------------------------------
    
    def get_device_info(self, device_id: int) -> Optional[GPUInfo]:
        """Get info for a specific device."""
        return self.device_infos.get(device_id)
    
    def get_all_devices(self) -> Dict[int, GPUInfo]:
        """Get all device infos."""
        return dict(self.device_infos)
    
    def get_available_devices(self) -> List[int]:
        """Get list of available device IDs."""
        return [
            gid for gid, info in self.device_infos.items()
            if info.is_available
        ]
    
    def get_healthy_devices(self) -> List[int]:
        """Get list of healthy device IDs."""
        return [
            gid for gid, info in self.device_infos.items()
            if info.state == GPUState.HEALTHY
        ]
    
    def get_best_device_for_memory(self, required_bytes: int) -> Optional[int]:
        """Get device with most available memory meeting requirement."""
        candidates = [
            gid for gid, info in self.device_infos.items()
            if info.is_available and info.available_memory_bytes >= required_bytes
        ]
        
        if not candidates:
            return None
        
        return max(
            candidates,
            key=lambda g: self.device_infos[g].available_memory_bytes
        )
    
    def get_optimal_transfer_path(
        self,
        src_gpu: int,
        dst_gpu: int
    ) -> List[int]:
        """Get optimal transfer path between GPUs."""
        if self.topology is None:
            return [src_gpu, dst_gpu] if src_gpu != dst_gpu else [src_gpu]
        
        return self.topology.optimal_paths.get(
            (src_gpu, dst_gpu),
            [src_gpu, dst_gpu] if src_gpu != dst_gpu else [src_gpu]
        )
    
    def get_bandwidth(self, src_gpu: int, dst_gpu: int) -> float:
        """Get bandwidth between GPUs in GB/s."""
        if self.topology is None:
            return 12.0  # Default PCIe
        return self.topology.get_bandwidth(src_gpu, dst_gpu)
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def get_statistics(self) -> CoordinatorStatistics:
        """Get coordinator statistics."""
        stats = self._statistics
        
        # Update real-time stats
        if self.scheduler:
            scheduler_stats = self.scheduler.get_statistics()
            stats.tasks_in_queue = scheduler_stats["queued_tasks"]
            stats.tasks_in_progress = scheduler_stats["active_tasks"]
            stats.avg_queue_wait_ms = scheduler_stats["avg_latency_ms"]
            stats.p50_latency_ms = scheduler_stats["p50_latency_ms"]
            stats.p99_latency_ms = scheduler_stats["p99_latency_ms"]
        
        # Update memory stats
        stats.used_memory_bytes = sum(
            info.total_memory_bytes - info.available_memory_bytes
            for info in self.device_infos.values()
        )
        
        # Update utilization
        if self.device_infos:
            stats.avg_utilization = sum(
                info.utilization_percent for info in self.device_infos.values()
            ) / len(self.device_infos)
        
        # Update health stats
        if self.health_monitor:
            stats.total_heartbeats = self.health_monitor.total_heartbeats
            stats.missed_heartbeats = self.health_monitor.missed_heartbeats
        
        return stats
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary for logging/monitoring."""
        stats = self.get_statistics()
        
        return {
            "running": self._running,
            "uptime_s": time.time() - self._start_time if self._start_time else 0,
            "gpus": {
                "total": stats.total_gpus,
                "healthy": stats.healthy_gpus,
                "degraded": stats.degraded_gpus,
                "failed": stats.failed_gpus,
            },
            "tasks": {
                "submitted": stats.total_tasks_submitted,
                "completed": stats.total_tasks_completed,
                "failed": stats.total_tasks_failed,
                "in_queue": stats.tasks_in_queue,
                "in_progress": stats.tasks_in_progress,
            },
            "memory": {
                "total_gb": stats.total_memory_bytes / (1024**3),
                "used_gb": stats.used_memory_bytes / (1024**3),
                "utilization": stats.memory_utilization,
            },
            "latency": {
                "avg_ms": stats.avg_queue_wait_ms,
                "p50_ms": stats.p50_latency_ms,
                "p99_ms": stats.p99_latency_ms,
            },
            "health": {
                "total_heartbeats": stats.total_heartbeats,
                "missed_heartbeats": stats.missed_heartbeats,
                "total_recoveries": stats.total_recoveries,
            },
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_coordinator(
    device_ids: Optional[List[int]] = None,
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.LEAST_LOADED,
    auto_start: bool = True,
    **kwargs
) -> GPUCoordinator:
    """
    Create and optionally start a GPU coordinator.
    
    Args:
        device_ids: Specific GPU device IDs (None for auto-detect)
        scheduling_policy: Workload scheduling policy
        auto_start: Whether to start services immediately
        **kwargs: Additional configuration options
        
    Returns:
        Initialized GPUCoordinator
    """
    scheduler_config = SchedulerConfig(policy=scheduling_policy)
    
    config = CoordinatorConfig(
        device_ids=device_ids,
        scheduler_config=scheduler_config,
        **kwargs
    )
    
    coordinator = GPUCoordinator(config)
    coordinator.initialize()
    
    if auto_start:
        coordinator.start()
    
    return coordinator


def get_gpu_cluster_info() -> Dict[str, Any]:
    """
    Get information about available GPU cluster.
    
    Returns:
        Dictionary with cluster information
    """
    if not cuda.is_available():
        return {
            "available": False,
            "cuda_version": None,
            "device_count": 0,
            "devices": [],
        }
    
    devices = []
    for i in range(cuda.device_count()):
        props = cuda.get_device_properties(i)
        mem_info = cuda.mem_get_info(i)
        
        devices.append({
            "id": i,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / (1024**3),
            "available_memory_gb": mem_info[0] / (1024**3),
            "multiprocessor_count": props.multi_processor_count,
        })
    
    return {
        "available": True,
        "cuda_version": cuda.runtime.cuda.runtimeGetVersion() if hasattr(cuda, 'runtime') else None,
        "device_count": cuda.device_count(),
        "devices": devices,
    }


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Enums
    "GPUState",
    "WorkloadPriority",
    "SchedulingPolicy",
    "TopologyType",
    "FailureType",
    "RecoveryAction",
    # Config
    "HealthMonitorConfig",
    "SchedulerConfig",
    "CoordinatorConfig",
    # Data structures
    "GPUInfo",
    "TopologyInfo",
    "WorkloadTask",
    "CoordinatorStatistics",
    # Components
    "TopologyDetector",
    "HealthMonitor",
    "WorkloadScheduler",
    "GPUCoordinator",
    # Factory functions
    "create_coordinator",
    "get_gpu_cluster_info",
]
