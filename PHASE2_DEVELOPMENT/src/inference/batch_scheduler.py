"""
Batch Scheduler for Ryzanstein LLM Serving System
==================================================

Advanced batch scheduling engine that determines optimal execution timing
based on multiple triggers (size, time, priority) and system conditions.

This module bridges request ingestion with batch execution, implementing
sophisticated scheduling policies that balance throughput and latency.

Cross-Domain Synthesis:
- Operating Systems: Real-time scheduling (EDF, Rate Monotonic)
- Database: Query scheduling and execution planning
- Network: QoS-aware packet scheduling
- Industrial: Just-In-Time manufacturing principles

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │              Batch Scheduler                        │
    ├─────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
    │  │  Trigger    │  │  Policy     │  │  Executor   │ │
    │  │  Manager    │──│  Engine     │──│  Interface  │ │
    │  └─────────────┘  └─────────────┘  └─────────────┘ │
    │         │               │                │         │
    │         ▼               ▼                ▼         │
    │  ┌─────────────────────────────────────────────┐  │
    │  │           Scheduling Decision               │  │
    │  │  (when, what size, which requests)          │  │
    │  └─────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘

Author: Ryzanstein LLM Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import heapq
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Coroutine,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEDULING POLICIES
# =============================================================================

class SchedulingPolicy(Enum):
    """Scheduling policy selection.
    
    Synthesis of multiple scheduling paradigms:
    - FIFO: Fair ordering, simple implementation (database query queues)
    - SIZE_OPTIMAL: Pack similar sizes (bin packing from operations research)
    - DEADLINE_DRIVEN: EDF from real-time systems theory
    - PRIORITY_WEIGHTED: QoS-aware (network packet scheduling)
    - ADAPTIVE: ML-based policy selection (reinforcement learning)
    """
    FIFO = auto()                # First In, First Out
    SIZE_OPTIMAL = auto()        # Group similar sequence lengths
    DEADLINE_DRIVEN = auto()     # Earliest Deadline First (EDF)
    PRIORITY_WEIGHTED = auto()   # Priority with fairness
    ADAPTIVE = auto()            # Dynamic policy switching


class TriggerType(Enum):
    """Batch execution trigger types.
    
    Multi-signal triggering inspired by event-driven architectures
    and industrial process control.
    """
    SIZE_THRESHOLD = auto()      # Batch reaches target size
    TIME_DEADLINE = auto()       # Maximum wait time exceeded
    PRIORITY_URGENT = auto()     # High-priority request received
    MEMORY_PRESSURE = auto()     # Memory approaching limits
    LOAD_SHEDDING = auto()       # System overload response
    MANUAL = auto()              # Explicit flush request


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ScheduledRequest:
    """A request scheduled for batch execution.
    
    Priority ordering follows real-time systems conventions:
    lower number = higher priority (like nice values in Unix).
    
    Comparison uses (priority, arrival_time) tuple for stable FIFO ordering
    within the same priority level.
    """
    priority: int
    arrival_time: float = field(default=0.0)
    deadline: Optional[float] = field(default=None)
    request_id: str = field(default="")
    sequence_length: int = field(default=0)
    max_tokens: int = field(default=0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def _comparison_key(self) -> Tuple[int, float]:
        """Return comparison key: (priority, arrival_time)."""
        return (self.priority, self.arrival_time)
    
    def __lt__(self, other: 'ScheduledRequest') -> bool:
        return self._comparison_key() < other._comparison_key()
    
    def __le__(self, other: 'ScheduledRequest') -> bool:
        return self._comparison_key() <= other._comparison_key()
    
    def __gt__(self, other: 'ScheduledRequest') -> bool:
        return self._comparison_key() > other._comparison_key()
    
    def __ge__(self, other: 'ScheduledRequest') -> bool:
        return self._comparison_key() >= other._comparison_key()
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScheduledRequest):
            return NotImplemented
        return self._comparison_key() == other._comparison_key()
    
    def __hash__(self) -> int:
        return hash((self.priority, self.arrival_time, self.request_id))
    
    @property
    def urgency(self) -> float:
        """Compute urgency score (higher = more urgent).
        
        Combines slack time and priority using weighted sum.
        Inspired by Earliest Deadline First (EDF) scheduling.
        
        Requests with deadlines always have higher urgency than those without,
        ensuring deadline-sensitive requests are prioritized.
        """
        if self.deadline is None:
            return float(100 - self.priority)  # Priority-based only (0-100)
        
        slack = self.deadline - time.time()
        if slack <= 0:
            return float('inf')  # Already overdue
        
        # Deadline requests get base urgency of 1000+ to always beat non-deadline
        # Urgency increases as deadline approaches
        # Priority provides secondary ordering
        return 1000.0 + (1.0 / max(slack, 0.001)) + (100 - self.priority) * 0.1


@dataclass
class SchedulerConfig:
    """Scheduler configuration parameters.
    
    Tunable parameters for different deployment scenarios:
    - Latency-sensitive: Low max_wait_time, smaller batch_size_target
    - Throughput-focused: Higher batch_size_target, longer max_wait_time
    """
    # Batch size triggers
    min_batch_size: int = 1
    max_batch_size: int = 64
    batch_size_target: int = 32
    
    # Time triggers (seconds)
    max_wait_time: float = 0.1       # Maximum wait before forced flush
    check_interval: float = 0.01     # Scheduling loop interval
    
    # Priority settings
    priority_levels: int = 10        # Number of priority levels (0-9)
    urgent_priority_threshold: int = 2  # Priority <= this triggers immediate
    
    # Memory settings
    memory_headroom_mb: float = 512  # Reserve memory for safety
    
    # Policy settings
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.ADAPTIVE
    enable_preemption: bool = False  # Allow preempting lower priority
    
    # Fairness settings
    starvation_prevention_time: float = 1.0  # Boost priority after this time
    fairness_window: int = 100       # Requests window for fairness tracking


@dataclass
class SchedulerStats:
    """Scheduler performance statistics."""
    total_requests_scheduled: int = 0
    total_batches_executed: int = 0
    total_wait_time: float = 0.0
    avg_batch_size: float = 0.0
    avg_wait_time: float = 0.0
    trigger_counts: Dict[TriggerType, int] = field(default_factory=dict)
    policy_usage: Dict[SchedulingPolicy, int] = field(default_factory=dict)
    deadline_misses: int = 0
    starvation_preventions: int = 0


# =============================================================================
# TRIGGER SYSTEM
# =============================================================================

class SchedulingTrigger(ABC):
    """Abstract base class for scheduling triggers.
    
    Follows the Strategy Pattern - each trigger encapsulates
    a specific decision criterion.
    """
    
    @abstractmethod
    def check(self, queue_state: 'QueueState') -> Tuple[bool, TriggerType]:
        """Check if trigger condition is met.
        
        Args:
            queue_state: Current state of the request queue
            
        Returns:
            Tuple of (should_trigger, trigger_type)
        """
        pass


class SizeThresholdTrigger(SchedulingTrigger):
    """Triggers when batch size reaches target."""
    
    def __init__(self, target_size: int):
        self.target_size = target_size
    
    def check(self, queue_state: 'QueueState') -> Tuple[bool, TriggerType]:
        if queue_state.pending_count >= self.target_size:
            return True, TriggerType.SIZE_THRESHOLD
        return False, TriggerType.SIZE_THRESHOLD


class TimeDeadlineTrigger(SchedulingTrigger):
    """Triggers when oldest request exceeds max wait time."""
    
    def __init__(self, max_wait_time: float):
        self.max_wait_time = max_wait_time
    
    def check(self, queue_state: 'QueueState') -> Tuple[bool, TriggerType]:
        if queue_state.oldest_arrival is None:
            return False, TriggerType.TIME_DEADLINE
        
        wait_time = time.time() - queue_state.oldest_arrival
        if wait_time >= self.max_wait_time:
            return True, TriggerType.TIME_DEADLINE
        return False, TriggerType.TIME_DEADLINE


class PriorityUrgentTrigger(SchedulingTrigger):
    """Triggers when high-priority request is present."""
    
    def __init__(self, urgent_threshold: int):
        self.urgent_threshold = urgent_threshold
    
    def check(self, queue_state: 'QueueState') -> Tuple[bool, TriggerType]:
        if queue_state.highest_priority <= self.urgent_threshold:
            return True, TriggerType.PRIORITY_URGENT
        return False, TriggerType.PRIORITY_URGENT


class MemoryPressureTrigger(SchedulingTrigger):
    """Triggers when memory usage is high to prevent OOM."""
    
    def __init__(self, memory_threshold_mb: float):
        self.memory_threshold_mb = memory_threshold_mb
    
    def check(self, queue_state: 'QueueState') -> Tuple[bool, TriggerType]:
        if queue_state.estimated_memory_mb >= self.memory_threshold_mb:
            return True, TriggerType.MEMORY_PRESSURE
        return False, TriggerType.MEMORY_PRESSURE


@dataclass
class QueueState:
    """Snapshot of queue state for trigger evaluation."""
    pending_count: int = 0
    oldest_arrival: Optional[float] = None
    highest_priority: int = 100
    estimated_memory_mb: float = 0.0
    total_tokens: int = 0
    sequence_length_distribution: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# SCHEDULING POLICIES
# =============================================================================

class BatchFormationPolicy(ABC):
    """Abstract policy for forming batches from pending requests.
    
    Implements the Strategy Pattern for different batching strategies.
    """
    
    @abstractmethod
    def form_batch(
        self,
        pending: List[ScheduledRequest],
        max_size: int,
        config: SchedulerConfig
    ) -> List[ScheduledRequest]:
        """Form an optimal batch from pending requests.
        
        Args:
            pending: List of pending requests
            max_size: Maximum batch size
            config: Scheduler configuration
            
        Returns:
            List of requests to include in the batch
        """
        pass


class FIFOPolicy(BatchFormationPolicy):
    """Simple first-in-first-out batching."""
    
    def form_batch(
        self,
        pending: List[ScheduledRequest],
        max_size: int,
        config: SchedulerConfig
    ) -> List[ScheduledRequest]:
        # Sort by arrival time, take up to max_size
        sorted_requests = sorted(pending, key=lambda r: r.arrival_time)
        return sorted_requests[:max_size]


class SizeOptimalPolicy(BatchFormationPolicy):
    """Group similar sequence lengths for padding efficiency.
    
    Inspired by bin-packing algorithms from operations research.
    Groups sequences with similar lengths to minimize padding waste.
    """
    
    def __init__(self, length_bucket_size: int = 32):
        self.bucket_size = length_bucket_size
    
    def form_batch(
        self,
        pending: List[ScheduledRequest],
        max_size: int,
        config: SchedulerConfig
    ) -> List[ScheduledRequest]:
        if not pending:
            return []
        
        # Bucket requests by sequence length
        buckets: Dict[int, List[ScheduledRequest]] = {}
        for req in pending:
            bucket_key = req.sequence_length // self.bucket_size
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(req)
        
        # Find largest bucket that can form a batch
        best_bucket = max(buckets.keys(), key=lambda k: len(buckets[k]))
        bucket_requests = buckets[best_bucket]
        
        # Sort by arrival within bucket, take up to max_size
        bucket_requests.sort(key=lambda r: r.arrival_time)
        return bucket_requests[:max_size]


class DeadlineDrivenPolicy(BatchFormationPolicy):
    """Earliest Deadline First (EDF) policy.
    
    From real-time systems theory - optimal for meeting deadlines
    when system is not overloaded.
    """
    
    def form_batch(
        self,
        pending: List[ScheduledRequest],
        max_size: int,
        config: SchedulerConfig
    ) -> List[ScheduledRequest]:
        # Sort by urgency (considers both deadline and priority)
        sorted_by_urgency = sorted(pending, key=lambda r: -r.urgency)
        return sorted_by_urgency[:max_size]


class PriorityWeightedPolicy(BatchFormationPolicy):
    """Priority-based with starvation prevention.
    
    Combines strict priority ordering with fairness mechanisms
    to prevent low-priority starvation (from QoS networking).
    """
    
    def form_batch(
        self,
        pending: List[ScheduledRequest],
        max_size: int,
        config: SchedulerConfig
    ) -> List[ScheduledRequest]:
        now = time.time()
        
        # Boost priority for starved requests
        adjusted_requests = []
        for req in pending:
            wait_time = now - req.arrival_time
            priority_boost = int(wait_time / config.starvation_prevention_time)
            adjusted_priority = max(0, req.priority - priority_boost)
            adjusted_requests.append((adjusted_priority, req.arrival_time, req))
        
        # Sort by adjusted priority, then arrival time
        adjusted_requests.sort()
        return [r[2] for r in adjusted_requests[:max_size]]


class AdaptivePolicy(BatchFormationPolicy):
    """Dynamically selects best policy based on conditions.
    
    Uses simple heuristics to choose between policies:
    - High variance in sequence lengths → SizeOptimal
    - Many urgent deadlines → DeadlineDriven
    - Significant priority spread → PriorityWeighted
    - Otherwise → FIFO
    """
    
    def __init__(self):
        self.fifo = FIFOPolicy()
        self.size_optimal = SizeOptimalPolicy()
        self.deadline = DeadlineDrivenPolicy()
        self.priority = PriorityWeightedPolicy()
    
    def form_batch(
        self,
        pending: List[ScheduledRequest],
        max_size: int,
        config: SchedulerConfig
    ) -> List[ScheduledRequest]:
        if not pending:
            return []
        
        # Analyze queue characteristics
        seq_lengths = [r.sequence_length for r in pending]
        priorities = [r.priority for r in pending]
        has_deadlines = any(r.deadline is not None for r in pending)
        
        # Compute variance metrics
        if len(seq_lengths) > 1:
            mean_len = sum(seq_lengths) / len(seq_lengths)
            length_variance = sum((l - mean_len) ** 2 for l in seq_lengths) / len(seq_lengths)
            length_cv = (length_variance ** 0.5) / max(mean_len, 1)  # Coefficient of variation
        else:
            length_cv = 0
        
        priority_spread = max(priorities) - min(priorities) if priorities else 0
        
        # Policy selection heuristics
        if has_deadlines and any(r.urgency > 10 for r in pending):
            return self.deadline.form_batch(pending, max_size, config)
        elif length_cv > 0.5:  # High variance in lengths
            return self.size_optimal.form_batch(pending, max_size, config)
        elif priority_spread >= 3:  # Significant priority differences
            return self.priority.form_batch(pending, max_size, config)
        else:
            return self.fifo.form_batch(pending, max_size, config)


# =============================================================================
# MAIN SCHEDULER
# =============================================================================

T = TypeVar('T')


class BatchScheduler(Generic[T]):
    """Main batch scheduler implementation.
    
    Orchestrates the scheduling lifecycle:
    1. Request admission
    2. Trigger evaluation
    3. Batch formation
    4. Execution dispatch
    
    Thread-safe implementation supporting both sync and async usage.
    """
    
    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        executor: Optional[Callable[[List[ScheduledRequest]], Coroutine]] = None
    ):
        self.config = config or SchedulerConfig()
        self.executor = executor
        
        # Request storage
        self._pending: List[ScheduledRequest] = []
        self._pending_lock = threading.Lock()
        
        # Scheduling state
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # Triggers
        self._triggers: List[SchedulingTrigger] = self._create_triggers()
        
        # Policy
        self._policy = self._create_policy()
        
        # Statistics
        self.stats = SchedulerStats()
        
        # Request tracking
        self._request_futures: Dict[str, asyncio.Future] = {}
        
        logger.info(f"BatchScheduler initialized with policy: {self.config.scheduling_policy}")
    
    def _create_triggers(self) -> List[SchedulingTrigger]:
        """Create trigger chain based on configuration."""
        return [
            SizeThresholdTrigger(self.config.batch_size_target),
            TimeDeadlineTrigger(self.config.max_wait_time),
            PriorityUrgentTrigger(self.config.urgent_priority_threshold),
            MemoryPressureTrigger(
                self.config.memory_headroom_mb * 0.8  # Trigger at 80% of headroom
            ),
        ]
    
    def _create_policy(self) -> BatchFormationPolicy:
        """Create batch formation policy based on configuration."""
        policy_map = {
            SchedulingPolicy.FIFO: FIFOPolicy,
            SchedulingPolicy.SIZE_OPTIMAL: SizeOptimalPolicy,
            SchedulingPolicy.DEADLINE_DRIVEN: DeadlineDrivenPolicy,
            SchedulingPolicy.PRIORITY_WEIGHTED: PriorityWeightedPolicy,
            SchedulingPolicy.ADAPTIVE: AdaptivePolicy,
        }
        return policy_map[self.config.scheduling_policy]()
    
    def _get_queue_state(self) -> QueueState:
        """Compute current queue state snapshot."""
        with self._pending_lock:
            if not self._pending:
                return QueueState()
            
            # Compute length distribution buckets
            length_dist: Dict[str, int] = {}
            for req in self._pending:
                bucket = f"{(req.sequence_length // 32) * 32}-{(req.sequence_length // 32 + 1) * 32}"
                length_dist[bucket] = length_dist.get(bucket, 0) + 1
            
            total_tokens = sum(r.sequence_length + r.max_tokens for r in self._pending)
            
            # Rough memory estimate (4 bytes per token, 2x for KV cache)
            estimated_memory = total_tokens * 4 * 2 / (1024 * 1024)
            
            return QueueState(
                pending_count=len(self._pending),
                oldest_arrival=min(r.arrival_time for r in self._pending),
                highest_priority=min(r.priority for r in self._pending),
                estimated_memory_mb=estimated_memory,
                total_tokens=total_tokens,
                sequence_length_distribution=length_dist
            )
    
    def submit(
        self,
        request_id: str,
        sequence_length: int,
        max_tokens: int = 256,
        priority: int = 5,
        deadline: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> asyncio.Future:
        """Submit a request for scheduling.
        
        Args:
            request_id: Unique request identifier
            sequence_length: Input sequence length
            max_tokens: Maximum output tokens
            priority: Request priority (0 = highest, 9 = lowest)
            deadline: Optional deadline timestamp
            metadata: Optional request metadata
            
        Returns:
            Future that resolves when request is processed
        """
        request = ScheduledRequest(
            priority=priority,
            arrival_time=time.time(),
            deadline=deadline,
            request_id=request_id,
            sequence_length=sequence_length,
            max_tokens=max_tokens,
            metadata=metadata or {}
        )
        
        future = asyncio.get_event_loop().create_future()
        self._request_futures[request_id] = future
        
        with self._pending_lock:
            self._pending.append(request)
            self.stats.total_requests_scheduled += 1
        
        logger.debug(
            f"Request {request_id} submitted: "
            f"seq_len={sequence_length}, priority={priority}"
        )
        
        return future
    
    def _check_triggers(self) -> Optional[TriggerType]:
        """Evaluate all triggers and return the first one that fires."""
        queue_state = self._get_queue_state()
        
        if queue_state.pending_count == 0:
            return None
        
        for trigger in self._triggers:
            should_fire, trigger_type = trigger.check(queue_state)
            if should_fire:
                self.stats.trigger_counts[trigger_type] = \
                    self.stats.trigger_counts.get(trigger_type, 0) + 1
                return trigger_type
        
        return None
    
    def _form_batch(self) -> List[ScheduledRequest]:
        """Form a batch using the current policy."""
        with self._pending_lock:
            batch = self._policy.form_batch(
                self._pending.copy(),
                self.config.max_batch_size,
                self.config
            )
            
            # Remove selected requests from pending
            batch_ids = {r.request_id for r in batch}
            self._pending = [r for r in self._pending if r.request_id not in batch_ids]
            
            return batch
    
    async def _execute_batch(self, batch: List[ScheduledRequest], trigger: TriggerType):
        """Execute a formed batch."""
        if not batch:
            return
        
        batch_size = len(batch)
        now = time.time()
        
        # Calculate wait time statistics
        wait_times = [now - r.arrival_time for r in batch]
        avg_wait = sum(wait_times) / len(wait_times)
        
        # Check for deadline misses
        for req in batch:
            if req.deadline and now > req.deadline:
                self.stats.deadline_misses += 1
        
        # Update statistics
        self.stats.total_batches_executed += 1
        self.stats.total_wait_time += sum(wait_times)
        self.stats.avg_batch_size = (
            (self.stats.avg_batch_size * (self.stats.total_batches_executed - 1) + batch_size)
            / self.stats.total_batches_executed
        )
        self.stats.avg_wait_time = (
            self.stats.total_wait_time / self.stats.total_requests_scheduled
            if self.stats.total_requests_scheduled > 0 else 0
        )
        
        logger.info(
            f"Executing batch: size={batch_size}, trigger={trigger.name}, "
            f"avg_wait={avg_wait*1000:.2f}ms"
        )
        
        # Execute if executor is provided
        if self.executor:
            try:
                results = await self.executor(batch)
                
                # Resolve futures
                if isinstance(results, dict):
                    for req in batch:
                        if req.request_id in self._request_futures:
                            future = self._request_futures.pop(req.request_id)
                            if not future.done():
                                future.set_result(results.get(req.request_id))
                else:
                    # Single result for entire batch
                    for req in batch:
                        if req.request_id in self._request_futures:
                            future = self._request_futures.pop(req.request_id)
                            if not future.done():
                                future.set_result(results)
                                
            except Exception as e:
                logger.error(f"Batch execution failed: {e}")
                # Fail all futures in batch
                for req in batch:
                    if req.request_id in self._request_futures:
                        future = self._request_futures.pop(req.request_id)
                        if not future.done():
                            future.set_exception(e)
    
    async def _scheduling_loop(self):
        """Main scheduling loop."""
        logger.info("Scheduling loop started")
        
        while self._running:
            try:
                # Check triggers
                trigger = self._check_triggers()
                
                if trigger:
                    # Form and execute batch
                    batch = self._form_batch()
                    if batch:
                        await self._execute_batch(batch, trigger)
                
                # Small sleep to prevent busy-waiting
                await asyncio.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"Scheduling loop error: {e}")
                await asyncio.sleep(self.config.check_interval)
        
        logger.info("Scheduling loop stopped")
    
    async def start(self):
        """Start the scheduler."""
        if self._running:
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduling_loop())
        logger.info("Scheduler started")
    
    async def stop(self):
        """Stop the scheduler and flush remaining requests."""
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining requests
        while True:
            with self._pending_lock:
                if not self._pending:
                    break
            batch = self._form_batch()
            if batch:
                await self._execute_batch(batch, TriggerType.MANUAL)
        
        logger.info("Scheduler stopped")
    
    async def flush(self):
        """Manually flush all pending requests."""
        while True:
            with self._pending_lock:
                if not self._pending:
                    break
            batch = self._form_batch()
            if batch:
                await self._execute_batch(batch, TriggerType.MANUAL)
    
    def get_stats(self) -> SchedulerStats:
        """Get scheduler statistics."""
        return self.stats
    
    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        with self._pending_lock:
            return len(self._pending)
    
    def update_config(self, **kwargs):
        """Update configuration parameters dynamically."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Recreate triggers and policy if needed
        if 'scheduling_policy' in kwargs:
            self._policy = self._create_policy()
        
        self._triggers = self._create_triggers()
        logger.info(f"Scheduler config updated: {kwargs}")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_scheduler(
    policy: SchedulingPolicy = SchedulingPolicy.ADAPTIVE,
    max_batch_size: int = 64,
    max_wait_time: float = 0.1,
    executor: Optional[Callable] = None
) -> BatchScheduler:
    """Create a configured batch scheduler.
    
    Args:
        policy: Scheduling policy to use
        max_batch_size: Maximum batch size
        max_wait_time: Maximum wait time before forced flush
        executor: Async function to execute batches
        
    Returns:
        Configured BatchScheduler instance
    """
    config = SchedulerConfig(
        scheduling_policy=policy,
        max_batch_size=max_batch_size,
        max_wait_time=max_wait_time
    )
    return BatchScheduler(config=config, executor=executor)


def create_latency_optimized_scheduler(executor: Optional[Callable] = None) -> BatchScheduler:
    """Create scheduler optimized for low latency."""
    config = SchedulerConfig(
        scheduling_policy=SchedulingPolicy.DEADLINE_DRIVEN,
        max_batch_size=16,
        batch_size_target=8,
        max_wait_time=0.05,
        check_interval=0.005,
        urgent_priority_threshold=3
    )
    return BatchScheduler(config=config, executor=executor)


def create_throughput_optimized_scheduler(executor: Optional[Callable] = None) -> BatchScheduler:
    """Create scheduler optimized for high throughput."""
    config = SchedulerConfig(
        scheduling_policy=SchedulingPolicy.SIZE_OPTIMAL,
        max_batch_size=128,
        batch_size_target=64,
        max_wait_time=0.2,
        check_interval=0.02
    )
    return BatchScheduler(config=config, executor=executor)


# =============================================================================
# ASYNC CONTEXT MANAGER
# =============================================================================

class SchedulerContext:
    """Context manager for scheduler lifecycle."""
    
    def __init__(self, scheduler: BatchScheduler):
        self.scheduler = scheduler
    
    async def __aenter__(self) -> BatchScheduler:
        await self.scheduler.start()
        return self.scheduler
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.scheduler.stop()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import uuid
    
    async def dummy_executor(batch: List[ScheduledRequest]) -> Dict[str, str]:
        """Dummy executor for testing."""
        print(f"Executing batch of {len(batch)} requests")
        await asyncio.sleep(0.01)  # Simulate processing
        return {req.request_id: f"Result for {req.request_id}" for req in batch}
    
    async def main():
        # Create scheduler
        scheduler = create_scheduler(
            policy=SchedulingPolicy.ADAPTIVE,
            max_batch_size=8,
            max_wait_time=0.1,
            executor=dummy_executor
        )
        
        async with SchedulerContext(scheduler):
            # Submit requests
            futures = []
            for i in range(20):
                future = scheduler.submit(
                    request_id=str(uuid.uuid4()),
                    sequence_length=50 + i * 10,
                    max_tokens=100,
                    priority=i % 5
                )
                futures.append(future)
            
            # Wait for all results
            results = await asyncio.gather(*futures)
            print(f"Processed {len(results)} requests")
            
            # Print stats
            stats = scheduler.get_stats()
            print(f"Total batches: {stats.total_batches_executed}")
            print(f"Avg batch size: {stats.avg_batch_size:.2f}")
            print(f"Avg wait time: {stats.avg_wait_time*1000:.2f}ms")
            print(f"Trigger counts: {stats.trigger_counts}")
    
    asyncio.run(main())
