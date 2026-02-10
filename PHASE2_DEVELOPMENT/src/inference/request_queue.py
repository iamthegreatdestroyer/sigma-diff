"""
Advanced Request Queue for Ryzanstein LLM Inference System
===========================================================

Implements sophisticated queue management with:
- Admission control to prevent overload
- Fair scheduling across tenants/priorities
- Backpressure signaling
- Request coalescing for efficiency
- SLA-aware queue prioritization

Cross-Domain Synthesis:
- Network Engineering: Token bucket, RED/ECN congestion control
- Database Systems: Query admission control (Oracle, SQL Server)
- Operating Systems: Multilevel feedback queue (MLFQ)
- Queueing Theory: M/M/c models, Little's Law optimization

Sprint 4.1 - Batch Processing Engine
Created: 2026-01-06
"""

from __future__ import annotations

import asyncio
import heapq
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import logging
import math

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class QueuePriority(Enum):
    """Request priority levels.
    
    Inspired by DSCP (Differentiated Services Code Point) from networking.
    """
    CRITICAL = 0      # System-critical, bypass most limits
    REALTIME = 1      # Real-time interactive
    HIGH = 2          # High priority batch
    NORMAL = 3        # Standard requests
    LOW = 4           # Background processing
    BULK = 5          # Best-effort bulk operations
    
    @property
    def weight(self) -> float:
        """Weight for weighted fair queueing."""
        weights = {0: 8.0, 1: 4.0, 2: 2.0, 3: 1.0, 4: 0.5, 5: 0.25}
        return weights.get(self.value, 1.0)


class AdmissionDecision(Enum):
    """Result of admission control check."""
    ADMIT = auto()           # Request admitted
    REJECT_OVERLOAD = auto() # Rejected due to overload
    REJECT_QUOTA = auto()    # Rejected due to quota
    DEFER = auto()           # Deferred for later
    THROTTLE = auto()        # Admitted but throttled


class BackpressureLevel(Enum):
    """System backpressure levels."""
    NONE = 0           # No backpressure
    LOW = 1            # Light backpressure, delay low priority
    MEDIUM = 2         # Moderate, reject bulk requests
    HIGH = 3           # Heavy, reject low priority
    CRITICAL = 4       # Critical, only realtime/critical allowed


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QueuedRequest:
    """A request in the queue."""
    request_id: str
    tenant_id: str = "default"
    priority: QueuePriority = QueuePriority.NORMAL
    sequence_length: int = 0
    max_tokens: int = 256
    arrival_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    estimated_cost: float = 1.0  # Computational cost units
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking
    dequeue_time: Optional[float] = None
    retries: int = 0
    
    def __lt__(self, other: 'QueuedRequest') -> bool:
        """Priority ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.arrival_time < other.arrival_time
    
    @property
    def wait_time(self) -> float:
        """Time spent waiting in queue."""
        end = self.dequeue_time or time.time()
        return end - self.arrival_time
    
    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.sequence_length + self.max_tokens


@dataclass
class TenantQuota:
    """Quota configuration for a tenant."""
    tenant_id: str
    max_concurrent_requests: int = 100
    max_requests_per_second: float = 50.0
    max_tokens_per_second: float = 10000.0
    max_queue_depth: int = 500
    priority_boost: int = 0  # Adjust priority (-2 to +2)
    
    # Tracking
    current_concurrent: int = 0
    tokens_this_second: float = 0.0
    requests_this_second: float = 0.0
    last_reset: float = field(default_factory=time.time)


@dataclass
class QueueStats:
    """Queue statistics."""
    total_enqueued: int = 0
    total_dequeued: int = 0
    total_rejected: int = 0
    total_timeout: int = 0
    
    # Current state
    current_depth: int = 0
    depth_by_priority: Dict[QueuePriority, int] = field(default_factory=dict)
    
    # Performance
    avg_wait_time_ms: float = 0.0
    p99_wait_time_ms: float = 0.0
    current_backpressure: BackpressureLevel = BackpressureLevel.NONE
    
    # Rates
    enqueue_rate_per_sec: float = 0.0
    dequeue_rate_per_sec: float = 0.0


@dataclass
class QueueConfig:
    """Queue configuration."""
    # Size limits
    max_queue_size: int = 10000
    max_queue_size_per_priority: Dict[QueuePriority, int] = field(
        default_factory=lambda: {
            QueuePriority.CRITICAL: 1000,
            QueuePriority.REALTIME: 2000,
            QueuePriority.HIGH: 2000,
            QueuePriority.NORMAL: 3000,
            QueuePriority.LOW: 1500,
            QueuePriority.BULK: 500,
        }
    )
    
    # Timing
    default_timeout: float = 30.0  # Request timeout in seconds
    starvation_prevention_time: float = 5.0  # Boost priority after this
    
    # Admission control
    enable_admission_control: bool = True
    admission_threshold: float = 0.8  # Start rejecting at 80% capacity
    
    # Backpressure
    enable_backpressure: bool = True
    backpressure_thresholds: Dict[BackpressureLevel, float] = field(
        default_factory=lambda: {
            BackpressureLevel.LOW: 0.5,
            BackpressureLevel.MEDIUM: 0.7,
            BackpressureLevel.HIGH: 0.85,
            BackpressureLevel.CRITICAL: 0.95,
        }
    )
    
    # Rate limiting (Token Bucket parameters)
    rate_limit_enabled: bool = True
    default_rate_limit: float = 100.0  # requests/sec
    default_burst_size: int = 200


# =============================================================================
# ADMISSION CONTROL
# =============================================================================

class AdmissionController(ABC):
    """Abstract base class for admission control."""
    
    @abstractmethod
    def check_admission(
        self,
        request: QueuedRequest,
        queue_state: 'QueueState'
    ) -> Tuple[AdmissionDecision, str]:
        """Check if request should be admitted.
        
        Returns:
            Tuple of (decision, reason)
        """
        pass


class TokenBucketController(AdmissionController):
    """Token bucket rate limiter.
    
    Classic network traffic shaping algorithm adapted for
    request admission control.
    """
    
    def __init__(
        self,
        rate: float = 100.0,       # tokens per second
        burst_size: int = 200      # maximum burst
    ):
        self.rate = rate
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
        self.last_update = now
    
    def check_admission(
        self,
        request: QueuedRequest,
        queue_state: 'QueueState'
    ) -> Tuple[AdmissionDecision, str]:
        with self._lock:
            self._refill()
            
            cost = request.estimated_cost
            if self.tokens >= cost:
                self.tokens -= cost
                return AdmissionDecision.ADMIT, "token_available"
            
            # Check if we should defer vs reject
            if self.tokens > 0:
                return AdmissionDecision.THROTTLE, "partial_tokens"
            
            return AdmissionDecision.REJECT_OVERLOAD, "no_tokens"


class LoadBasedController(AdmissionController):
    """Load-based admission control.
    
    Inspired by database query admission control (e.g., Oracle Resource Manager).
    Adjusts admission based on current system load.
    """
    
    def __init__(self, config: QueueConfig):
        self.config = config
    
    def check_admission(
        self,
        request: QueuedRequest,
        queue_state: 'QueueState'
    ) -> Tuple[AdmissionDecision, str]:
        # Calculate load factor
        load = queue_state.current_depth / max(1, self.config.max_queue_size)
        
        # Priority-based admission thresholds
        priority_thresholds = {
            QueuePriority.CRITICAL: 1.0,    # Always admit
            QueuePriority.REALTIME: 0.95,
            QueuePriority.HIGH: 0.85,
            QueuePriority.NORMAL: 0.75,
            QueuePriority.LOW: 0.6,
            QueuePriority.BULK: 0.4,
        }
        
        threshold = priority_thresholds.get(request.priority, 0.5)
        
        if load < threshold:
            return AdmissionDecision.ADMIT, f"load={load:.2f}<{threshold:.2f}"
        
        if request.priority in (QueuePriority.CRITICAL, QueuePriority.REALTIME):
            return AdmissionDecision.ADMIT, "priority_override"
        
        return AdmissionDecision.REJECT_OVERLOAD, f"load={load:.2f}>={threshold:.2f}"


class CompositeAdmissionController(AdmissionController):
    """Combines multiple admission controllers."""
    
    def __init__(self, controllers: List[AdmissionController]):
        self.controllers = controllers
    
    def check_admission(
        self,
        request: QueuedRequest,
        queue_state: 'QueueState'
    ) -> Tuple[AdmissionDecision, str]:
        for controller in self.controllers:
            decision, reason = controller.check_admission(request, queue_state)
            if decision != AdmissionDecision.ADMIT:
                return decision, reason
        
        return AdmissionDecision.ADMIT, "all_checks_passed"


# =============================================================================
# FAIR SCHEDULER
# =============================================================================

class FairScheduler:
    """Weighted Fair Queue scheduler.
    
    Ensures fair distribution of resources across tenants and priorities
    while respecting weights. Inspired by:
    - WFQ (Weighted Fair Queueing) from networking
    - CFS (Completely Fair Scheduler) from Linux kernel
    """
    
    def __init__(self):
        # Virtual time tracking per tenant
        self.tenant_vtime: Dict[str, float] = defaultdict(float)
        self.global_vtime: float = 0.0
        
        # Per-tenant request counts
        self.tenant_requests: Dict[str, int] = defaultdict(int)
        self.tenant_tokens: Dict[str, int] = defaultdict(int)
    
    def compute_virtual_time(
        self,
        request: QueuedRequest,
        current_time: float
    ) -> float:
        """Compute virtual finish time for fair scheduling.
        
        Uses WFQ formula: virtual_finish = virtual_start + (length / weight)
        """
        weight = request.priority.weight
        length = request.total_tokens
        
        # Virtual start time is max of global time and tenant's last finish
        tenant_last = self.tenant_vtime.get(request.tenant_id, 0.0)
        virtual_start = max(self.global_vtime, tenant_last)
        
        # Virtual finish time
        virtual_finish = virtual_start + (length / weight)
        
        return virtual_finish
    
    def update_on_dequeue(self, request: QueuedRequest) -> None:
        """Update virtual times when request is dequeued."""
        vtime = self.compute_virtual_time(request, time.time())
        self.tenant_vtime[request.tenant_id] = vtime
        self.global_vtime = max(self.global_vtime, vtime)
        
        # Update counters
        self.tenant_requests[request.tenant_id] -= 1
        self.tenant_tokens[request.tenant_id] -= request.total_tokens
    
    def update_on_enqueue(self, request: QueuedRequest) -> None:
        """Update tracking when request is enqueued."""
        self.tenant_requests[request.tenant_id] += 1
        self.tenant_tokens[request.tenant_id] += request.total_tokens
    
    def get_tenant_fairness_score(self, tenant_id: str) -> float:
        """Get fairness score (lower = more deserving of resources)."""
        vtime = self.tenant_vtime.get(tenant_id, 0.0)
        return vtime - self.global_vtime


# =============================================================================
# QUEUE STATE
# =============================================================================

@dataclass
class QueueState:
    """Snapshot of queue state."""
    current_depth: int = 0
    depth_by_priority: Dict[QueuePriority, int] = field(default_factory=dict)
    backpressure_level: BackpressureLevel = BackpressureLevel.NONE
    oldest_request_age: float = 0.0
    enqueue_rate: float = 0.0
    dequeue_rate: float = 0.0


# =============================================================================
# MAIN REQUEST QUEUE
# =============================================================================

class RequestQueue:
    """Advanced request queue with admission control and fairness.
    
    Core data structure:
    - Priority queues per priority level
    - Heap-based ordering within each level
    - Fair scheduling across tenants
    
    Features:
    - Multi-level priority queues (MLFQ-inspired)
    - Token bucket rate limiting
    - Load-based admission control
    - Backpressure signaling
    - Starvation prevention via priority aging
    - Request timeout and cancellation
    """
    
    def __init__(self, config: Optional[QueueConfig] = None):
        self.config = config or QueueConfig()
        
        # Priority queues (min-heap by virtual finish time)
        self._queues: Dict[QueuePriority, List[Tuple[float, QueuedRequest]]] = {
            p: [] for p in QueuePriority
        }
        
        # Request lookup
        self._requests: Dict[str, QueuedRequest] = {}
        self._cancelled: Set[str] = set()
        
        # Tenant management
        self._tenant_quotas: Dict[str, TenantQuota] = {}
        
        # Fair scheduling
        self._fair_scheduler = FairScheduler()
        
        # Admission control
        self._admission_controller = self._create_admission_controller()
        
        # Statistics
        self._stats = QueueStats()
        self._wait_times: deque = deque(maxlen=1000)
        self._enqueue_times: deque = deque(maxlen=100)
        self._dequeue_times: deque = deque(maxlen=100)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Backpressure
        self._current_backpressure = BackpressureLevel.NONE
        
        logger.info(f"RequestQueue initialized: max_size={self.config.max_queue_size}")
    
    def _create_admission_controller(self) -> AdmissionController:
        """Create composite admission controller."""
        controllers = []
        
        if self.config.rate_limit_enabled:
            controllers.append(TokenBucketController(
                rate=self.config.default_rate_limit,
                burst_size=self.config.default_burst_size
            ))
        
        if self.config.enable_admission_control:
            controllers.append(LoadBasedController(self.config))
        
        return CompositeAdmissionController(controllers)
    
    def _get_queue_state(self) -> QueueState:
        """Get current queue state snapshot."""
        depth_by_priority = {
            p: len(self._queues[p]) for p in QueuePriority
        }
        
        oldest_age = 0.0
        for priority_queue in self._queues.values():
            if priority_queue:
                oldest = min(r.arrival_time for _, r in priority_queue)
                age = time.time() - oldest
                oldest_age = max(oldest_age, age)
        
        # Calculate rates
        now = time.time()
        recent_enqueues = [t for t in self._enqueue_times if now - t < 1.0]
        recent_dequeues = [t for t in self._dequeue_times if now - t < 1.0]
        
        return QueueState(
            current_depth=len(self._requests),
            depth_by_priority=depth_by_priority,
            backpressure_level=self._current_backpressure,
            oldest_request_age=oldest_age,
            enqueue_rate=len(recent_enqueues),
            dequeue_rate=len(recent_dequeues)
        )
    
    def _update_backpressure(self) -> None:
        """Update backpressure level based on queue state."""
        if not self.config.enable_backpressure:
            return
        
        load = len(self._requests) / max(1, self.config.max_queue_size)
        
        # Determine backpressure level
        new_level = BackpressureLevel.NONE
        for level, threshold in sorted(
            self.config.backpressure_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if load >= threshold:
                new_level = level
                break
        
        if new_level != self._current_backpressure:
            logger.info(
                f"Backpressure changed: {self._current_backpressure.name} -> {new_level.name}"
            )
            self._current_backpressure = new_level
    
    def _check_starvation(self) -> None:
        """Boost priority of starved requests.
        
        Implements MLFQ-style priority aging to prevent starvation.
        """
        now = time.time()
        threshold = self.config.starvation_prevention_time
        
        for priority in [QueuePriority.LOW, QueuePriority.BULK, QueuePriority.NORMAL]:
            queue = self._queues[priority]
            promoted = []
            remaining = []
            
            for vtime, request in queue:
                wait_time = now - request.arrival_time
                if wait_time > threshold and priority.value > 0:
                    # Promote to higher priority
                    new_priority = QueuePriority(priority.value - 1)
                    request.priority = new_priority
                    promoted.append((vtime, request))
                    self._stats.total_timeout += 1  # Track as starvation prevention
                else:
                    remaining.append((vtime, request))
            
            self._queues[priority] = remaining
            
            for vtime, request in promoted:
                heapq.heappush(
                    self._queues[request.priority],
                    (vtime - 1000, request)  # Boost virtual time
                )
    
    def enqueue(
        self,
        request_id: str,
        sequence_length: int,
        max_tokens: int = 256,
        priority: QueuePriority = QueuePriority.NORMAL,
        tenant_id: str = "default",
        deadline: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """Enqueue a request.
        
        Args:
            request_id: Unique request ID
            sequence_length: Input sequence length
            max_tokens: Maximum output tokens
            priority: Request priority
            tenant_id: Tenant identifier
            deadline: Optional deadline timestamp
            metadata: Optional metadata
            
        Returns:
            Tuple of (success, reason)
        """
        request = QueuedRequest(
            request_id=request_id,
            tenant_id=tenant_id,
            priority=priority,
            sequence_length=sequence_length,
            max_tokens=max_tokens,
            deadline=deadline,
            metadata=metadata or {}
        )
        
        with self._lock:
            # Check admission
            queue_state = self._get_queue_state()
            decision, reason = self._admission_controller.check_admission(
                request, queue_state
            )
            
            if decision == AdmissionDecision.REJECT_OVERLOAD:
                self._stats.total_rejected += 1
                return False, f"rejected: {reason}"
            
            if decision == AdmissionDecision.REJECT_QUOTA:
                self._stats.total_rejected += 1
                return False, f"quota_exceeded: {reason}"
            
            # Check per-priority limits
            priority_limit = self.config.max_queue_size_per_priority.get(
                priority, self.config.max_queue_size
            )
            if len(self._queues[priority]) >= priority_limit:
                self._stats.total_rejected += 1
                return False, f"priority_queue_full: {priority.name}"
            
            # Compute virtual time for fair scheduling
            vtime = self._fair_scheduler.compute_virtual_time(request, time.time())
            
            # Add to queue
            heapq.heappush(self._queues[priority], (vtime, request))
            self._requests[request_id] = request
            self._fair_scheduler.update_on_enqueue(request)
            
            # Update statistics
            self._stats.total_enqueued += 1
            self._stats.current_depth = len(self._requests)
            self._enqueue_times.append(time.time())
            
            # Update backpressure
            self._update_backpressure()
            
            logger.debug(
                f"Enqueued request {request_id}: priority={priority.name}, "
                f"queue_depth={len(self._requests)}"
            )
            
            return True, "enqueued"
    
    def dequeue(
        self,
        max_requests: int = 1,
        max_tokens: Optional[int] = None,
        allowed_priorities: Optional[Set[QueuePriority]] = None
    ) -> List[QueuedRequest]:
        """Dequeue requests for processing.
        
        Args:
            max_requests: Maximum requests to dequeue
            max_tokens: Maximum total tokens to dequeue
            allowed_priorities: Only dequeue from these priorities
            
        Returns:
            List of dequeued requests
        """
        with self._lock:
            # Check for starvation periodically
            self._check_starvation()
            
            dequeued = []
            total_tokens = 0
            
            # Process priorities in order
            priorities = allowed_priorities or set(QueuePriority)
            
            for priority in sorted(priorities, key=lambda p: p.value):
                queue = self._queues[priority]
                
                while queue and len(dequeued) < max_requests:
                    # Check token limit
                    if max_tokens is not None:
                        _, peek_request = queue[0]
                        if total_tokens + peek_request.total_tokens > max_tokens:
                            break
                    
                    # Pop request
                    vtime, request = heapq.heappop(queue)
                    
                    # Skip cancelled requests
                    if request.request_id in self._cancelled:
                        self._cancelled.discard(request.request_id)
                        del self._requests[request.request_id]
                        continue
                    
                    # Check timeout
                    if request.deadline and time.time() > request.deadline:
                        self._stats.total_timeout += 1
                        del self._requests[request.request_id]
                        continue
                    
                    # Dequeue
                    request.dequeue_time = time.time()
                    dequeued.append(request)
                    total_tokens += request.total_tokens
                    
                    # Update tracking
                    del self._requests[request.request_id]
                    self._fair_scheduler.update_on_dequeue(request)
                    self._wait_times.append(request.wait_time * 1000)
                    self._dequeue_times.append(time.time())
            
            # Update statistics
            self._stats.total_dequeued += len(dequeued)
            self._stats.current_depth = len(self._requests)
            
            if self._wait_times:
                self._stats.avg_wait_time_ms = sum(self._wait_times) / len(self._wait_times)
                sorted_times = sorted(self._wait_times)
                p99_idx = int(len(sorted_times) * 0.99)
                self._stats.p99_wait_time_ms = sorted_times[min(p99_idx, len(sorted_times) - 1)]
            
            # Update backpressure
            self._update_backpressure()
            
            return dequeued
    
    def cancel(self, request_id: str) -> bool:
        """Cancel a pending request.
        
        Args:
            request_id: Request to cancel
            
        Returns:
            True if request was found and cancelled
        """
        with self._lock:
            if request_id in self._requests:
                self._cancelled.add(request_id)
                return True
            return False
    
    def peek(self, n: int = 1) -> List[QueuedRequest]:
        """Peek at top N requests without dequeuing."""
        with self._lock:
            result = []
            for priority in QueuePriority:
                for _, request in self._queues[priority][:n - len(result)]:
                    if request.request_id not in self._cancelled:
                        result.append(request)
                    if len(result) >= n:
                        break
            return result
    
    def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        with self._lock:
            self._stats.depth_by_priority = {
                p: len(self._queues[p]) for p in QueuePriority
            }
            self._stats.current_backpressure = self._current_backpressure
            return self._stats
    
    def get_backpressure_level(self) -> BackpressureLevel:
        """Get current backpressure level."""
        return self._current_backpressure
    
    def set_tenant_quota(self, quota: TenantQuota) -> None:
        """Set quota for a tenant."""
        with self._lock:
            self._tenant_quotas[quota.tenant_id] = quota
    
    def __len__(self) -> int:
        """Get current queue depth."""
        return len(self._requests)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._requests) == 0


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_request_queue(
    max_size: int = 10000,
    enable_admission_control: bool = True,
    enable_backpressure: bool = True,
    rate_limit: float = 100.0
) -> RequestQueue:
    """Create a configured request queue.
    
    Args:
        max_size: Maximum queue size
        enable_admission_control: Enable admission control
        enable_backpressure: Enable backpressure signaling
        rate_limit: Rate limit (requests/sec)
        
    Returns:
        Configured RequestQueue
    """
    config = QueueConfig(
        max_queue_size=max_size,
        enable_admission_control=enable_admission_control,
        enable_backpressure=enable_backpressure,
        rate_limit_enabled=rate_limit > 0,
        default_rate_limit=rate_limit
    )
    return RequestQueue(config)


def create_high_throughput_queue() -> RequestQueue:
    """Create queue optimized for high throughput."""
    config = QueueConfig(
        max_queue_size=50000,
        enable_admission_control=True,
        admission_threshold=0.9,
        enable_backpressure=True,
        rate_limit_enabled=False  # No rate limiting for throughput
    )
    return RequestQueue(config)


def create_low_latency_queue() -> RequestQueue:
    """Create queue optimized for low latency."""
    config = QueueConfig(
        max_queue_size=1000,
        default_timeout=5.0,
        starvation_prevention_time=1.0,
        enable_admission_control=True,
        admission_threshold=0.5,  # Aggressive admission control
        enable_backpressure=True
    )
    return RequestQueue(config)


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

class QueueMonitor:
    """Monitors queue health and emits metrics."""
    
    def __init__(self, queue: RequestQueue, interval: float = 1.0):
        self.queue = queue
        self.interval = interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start monitoring."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
    
    async def _monitor_loop(self):
        """Monitoring loop."""
        while self._running:
            try:
                stats = self.queue.get_stats()
                
                logger.info(
                    f"Queue stats: depth={stats.current_depth}, "
                    f"backpressure={stats.current_backpressure.name}, "
                    f"avg_wait={stats.avg_wait_time_ms:.2f}ms, "
                    f"p99_wait={stats.p99_wait_time_ms:.2f}ms"
                )
                
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(self.interval)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import uuid
    
    # Create queue
    queue = create_request_queue(
        max_size=1000,
        enable_admission_control=True,
        rate_limit=50.0
    )
    
    # Enqueue requests with different priorities
    for i in range(100):
        priority = QueuePriority(i % 6)
        success, reason = queue.enqueue(
            request_id=str(uuid.uuid4()),
            sequence_length=50 + i * 5,
            max_tokens=100,
            priority=priority,
            tenant_id=f"tenant_{i % 3}"
        )
        print(f"Enqueue {i}: {success} - {reason}")
    
    print(f"\nQueue depth: {len(queue)}")
    print(f"Stats: {queue.get_stats()}")
    
    # Dequeue in batches
    batch = queue.dequeue(max_requests=10)
    print(f"\nDequeued {len(batch)} requests")
    for req in batch:
        print(f"  - {req.request_id[:8]}... priority={req.priority.name} wait={req.wait_time*1000:.2f}ms")
    
    print(f"\nFinal queue depth: {len(queue)}")
    print(f"Backpressure: {queue.get_backpressure_level().name}")
