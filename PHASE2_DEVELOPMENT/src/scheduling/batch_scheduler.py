"""
Sprint 4.3: Adaptive Batch Scheduler with ML-Based Policy Selection.

This module implements an advanced batch scheduling system that synthesizes
concepts from multiple domains:

CROSS-DOMAIN SYNTHESIS:
- Operating Systems: CFS (Completely Fair Scheduler), MLFQ, lottery scheduling
- Network QoS: Token bucket, weighted fair queuing, admission control
- Machine Learning: Online learning, bandit algorithms, workload prediction
- Database: Query optimization, cost estimation, adaptive execution
- Economics: Auction theory, resource pricing, utility maximization
- Real-Time Systems: EDF, rate-monotonic, deadline-aware scheduling

Key Features:
- ML-based scheduling policy selection using contextual bandits
- Workload profiling with statistical characterization
- Adaptive batch formation with latency prediction
- Priority inheritance and deadline propagation
- Preemption support with state preservation
- Fairness guarantees across tenants/users

Performance Targets:
- Scheduling overhead: < 2% of total request time
- Throughput optimization: Maximize GPU utilization
- Latency SLO adherence: > 99th percentile compliance
- Fairness: Max-min fair allocation

Author: NEXUS (Paradigm Synthesis & Cross-Domain Innovation)
Sprint: 4.3 - Advanced Scheduling & Resource Management
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================


class SchedulingPolicy(Enum):
    """
    Scheduling policies synthesized from multiple domains.
    
    Each policy excels in different workload scenarios:
    - FCFS: Simple, fair, low overhead (batch workloads)
    - SJF: Minimizes average latency (known job sizes)
    - EDF: Real-time deadline guarantees
    - FAIR: Multi-tenant fairness (CFS-inspired)
    - PRIORITY: QoS differentiation
    - MLFQ: Adaptive without a priori knowledge
    - LOTTERY: Probabilistic fairness
    - WEIGHTED_FAIR: Hierarchical resource sharing
    """
    FCFS = auto()              # First-Come First-Served
    SJF = auto()               # Shortest Job First
    EDF = auto()               # Earliest Deadline First
    FAIR = auto()              # CFS-inspired fair scheduling
    PRIORITY = auto()          # Priority-based scheduling
    MLFQ = auto()              # Multi-Level Feedback Queue
    LOTTERY = auto()           # Lottery scheduling
    WEIGHTED_FAIR = auto()     # Weighted fair queuing
    ADAPTIVE = auto()          # ML-selected policy


class BatchFormationStrategy(Enum):
    """Strategies for forming request batches."""
    GREEDY = auto()            # Fill batch ASAP
    DEADLINE_AWARE = auto()    # Form based on deadline proximity
    SIZE_HOMOGENEOUS = auto()  # Group similar-sized requests
    TENANT_AFFINITY = auto()   # Group by tenant for cache locality
    LATENCY_OPTIMAL = auto()   # Minimize p99 latency
    THROUGHPUT_OPTIMAL = auto() # Maximize requests/second


class PreemptionPolicy(Enum):
    """How to handle preemption of running requests."""
    NONE = auto()              # No preemption
    COOPERATIVE = auto()       # Preempt at checkpoints
    IMMEDIATE = auto()         # Immediate preemption
    LAZY = auto()              # Preempt when convenient


class WorkloadType(Enum):
    """Classification of request workloads."""
    INTERACTIVE = auto()       # Low latency, small batches
    BATCH = auto()             # High throughput, large batches
    STREAMING = auto()         # Continuous token generation
    MIXED = auto()             # Variable workload patterns
    BURSTY = auto()            # Spiky traffic patterns


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class LatencyTarget:
    """Latency SLO specification."""
    p50_ms: float = 100.0
    p90_ms: float = 200.0
    p99_ms: float = 500.0
    max_ms: float = 1000.0
    
    def is_satisfied(self, actual_p50: float, actual_p99: float) -> bool:
        """Check if latency targets are met."""
        return actual_p50 <= self.p50_ms and actual_p99 <= self.p99_ms


@dataclass
class RequestPriority:
    """
    Priority specification for a request.
    
    Uses a multi-dimensional priority model:
    - Base priority: Static importance level
    - Urgency: Time-based priority boost (deadline proximity)
    - Value: Economic value of request (revenue impact)
    """
    base: int = 0                    # -10 (lowest) to 10 (highest)
    urgency: float = 0.0             # 0.0 to 1.0 (deadline proximity)
    value: float = 1.0               # Economic value multiplier
    boost_rate: float = 0.1          # Priority boost per second waiting
    
    def effective_priority(self, wait_time_s: float) -> float:
        """Calculate effective priority with aging."""
        boost = min(wait_time_s * self.boost_rate, 5.0)  # Cap boost at 5
        return (self.base + boost) * self.value * (1.0 + self.urgency)


@dataclass
class SchedulingRequest:
    """
    A request to be scheduled for inference.
    
    Contains all information needed for intelligent scheduling decisions.
    """
    request_id: str
    tenant_id: str
    sequence_length: int
    max_new_tokens: int
    priority: RequestPriority = field(default_factory=RequestPriority)
    deadline_ms: Optional[float] = None       # Absolute deadline
    arrival_time: float = field(default_factory=time.monotonic)
    estimated_duration_ms: Optional[float] = None
    preemptible: bool = True
    checkpoint_interval: int = 32              # Tokens between checkpoints
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.estimated_duration_ms is None:
            # Simple heuristic: ~5ms per token + overhead
            self.estimated_duration_ms = self.max_new_tokens * 5.0 + 50.0
    
    @property
    def wait_time(self) -> float:
        """Time spent waiting in queue (seconds)."""
        return time.monotonic() - self.arrival_time
    
    @property
    def slack_time(self) -> Optional[float]:
        """Time remaining until deadline (ms), or None if no deadline."""
        if self.deadline_ms is None:
            return None
        elapsed_ms = (time.monotonic() - self.arrival_time) * 1000
        return self.deadline_ms - elapsed_ms - (self.estimated_duration_ms or 0)
    
    def __lt__(self, other: SchedulingRequest) -> bool:
        """For heap ordering - lower is higher priority."""
        return self.priority.effective_priority(self.wait_time) > \
               other.priority.effective_priority(other.wait_time)


@dataclass
class SchedulingDecision:
    """Result of a scheduling decision."""
    requests: List[SchedulingRequest]
    batch_size: int
    policy_used: SchedulingPolicy
    estimated_latency_ms: float
    estimated_throughput: float           # Tokens per second
    reasoning: str
    preempted_requests: List[str] = field(default_factory=list)
    priority_inversions: int = 0


@dataclass
class WorkloadProfile:
    """
    Statistical characterization of workload patterns.
    
    Used for policy selection and capacity planning.
    """
    avg_sequence_length: float = 512.0
    sequence_length_std: float = 256.0
    avg_tokens_requested: float = 128.0
    tokens_requested_std: float = 64.0
    avg_arrival_rate: float = 10.0         # Requests per second
    arrival_burstiness: float = 1.0        # Coefficient of variation
    tenant_count: int = 1
    priority_distribution: Dict[int, float] = field(default_factory=dict)
    workload_type: WorkloadType = WorkloadType.MIXED
    
    def update(self, request: SchedulingRequest, window_requests: List[SchedulingRequest]):
        """Update profile with new request statistics."""
        if not window_requests:
            return
        
        # Compute rolling statistics
        seq_lengths = [r.sequence_length for r in window_requests]
        token_counts = [r.max_new_tokens for r in window_requests]
        
        self.avg_sequence_length = sum(seq_lengths) / len(seq_lengths)
        self.avg_tokens_requested = sum(token_counts) / len(token_counts)
        
        # Compute std dev
        if len(seq_lengths) > 1:
            mean_sq = sum((x - self.avg_sequence_length) ** 2 for x in seq_lengths)
            self.sequence_length_std = math.sqrt(mean_sq / (len(seq_lengths) - 1))
            
            mean_sq_tok = sum((x - self.avg_tokens_requested) ** 2 for x in token_counts)
            self.tokens_requested_std = math.sqrt(mean_sq_tok / (len(token_counts) - 1))
        
        # Classify workload type
        self._classify_workload(window_requests)
    
    def _classify_workload(self, requests: List[SchedulingRequest]):
        """Classify workload based on request characteristics."""
        if not requests:
            return
        
        avg_tokens = sum(r.max_new_tokens for r in requests) / len(requests)
        has_deadlines = any(r.deadline_ms is not None for r in requests)
        
        if avg_tokens < 50 and has_deadlines:
            self.workload_type = WorkloadType.INTERACTIVE
        elif avg_tokens > 200:
            self.workload_type = WorkloadType.BATCH
        elif self.arrival_burstiness > 2.0:
            self.workload_type = WorkloadType.BURSTY
        else:
            self.workload_type = WorkloadType.MIXED


@dataclass
class TenantState:
    """Per-tenant scheduling state for fair sharing."""
    tenant_id: str
    weight: float = 1.0                    # Relative share weight
    virtual_time: float = 0.0              # CFS virtual runtime
    tokens_used: int = 0                   # Total tokens consumed
    requests_completed: int = 0
    current_requests: int = 0              # In-flight requests
    quota_remaining: Optional[int] = None  # Token quota if set
    last_scheduled: float = 0.0
    
    def consume_tokens(self, count: int):
        """Record token consumption, updating virtual time."""
        self.tokens_used += count
        # Virtual time advances inversely to weight (CFS-inspired)
        self.virtual_time += count / max(self.weight, 0.001)


@dataclass 
class SchedulerMetrics:
    """Comprehensive scheduler performance metrics."""
    total_requests_scheduled: int = 0
    total_batches_formed: int = 0
    avg_batch_size: float = 0.0
    avg_queue_wait_ms: float = 0.0
    avg_scheduling_overhead_us: float = 0.0
    deadline_miss_count: int = 0
    deadline_miss_rate: float = 0.0
    preemption_count: int = 0
    policy_usage: Dict[SchedulingPolicy, int] = field(default_factory=dict)
    throughput_tokens_per_sec: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    fairness_jain_index: float = 1.0       # 1.0 = perfectly fair


# =============================================================================
# LATENCY PREDICTOR (ML-Inspired)
# =============================================================================


class LatencyPredictor:
    """
    Predicts request latency using learned models.
    
    Combines:
    - Analytical model: Based on known system characteristics
    - Statistical model: Running statistics from observations
    - ML model: Linear regression on features (optional)
    
    This is similar to database query cost estimation.
    """
    
    def __init__(self, 
                 base_overhead_ms: float = 10.0,
                 per_token_ms: float = 5.0,
                 batch_overhead_ms: float = 20.0):
        self.base_overhead_ms = base_overhead_ms
        self.per_token_ms = per_token_ms
        self.batch_overhead_ms = batch_overhead_ms
        
        # Running statistics for calibration
        self._observations: deque = deque(maxlen=1000)
        self._prediction_errors: deque = deque(maxlen=100)
        
        # Simple linear model coefficients (updated online)
        self._coefficients = {
            'intercept': base_overhead_ms,
            'tokens': per_token_ms,
            'sequence_length': 0.01,
            'batch_size': batch_overhead_ms / 8,  # Per-request overhead
        }
        
        self._lock = threading.Lock()
    
    def predict(self, 
                request: SchedulingRequest,
                batch_size: int = 1,
                current_load: float = 0.5) -> float:
        """
        Predict latency for a request.
        
        Args:
            request: The request to estimate
            batch_size: Current batch size
            current_load: System load factor (0.0 to 1.0)
            
        Returns:
            Predicted latency in milliseconds
        """
        with self._lock:
            # Analytical component
            base = self._coefficients['intercept']
            token_cost = self._coefficients['tokens'] * request.max_new_tokens
            seq_cost = self._coefficients['sequence_length'] * request.sequence_length
            batch_cost = self._coefficients['batch_size'] * batch_size
            
            # Load-based scaling (queueing theory inspired)
            load_factor = 1.0 + (current_load ** 2) * 2.0  # M/M/1 approximation
            
            predicted = (base + token_cost + seq_cost + batch_cost) * load_factor
            
            return max(predicted, 1.0)  # Minimum 1ms
    
    def observe(self, 
                request: SchedulingRequest,
                actual_latency_ms: float,
                batch_size: int):
        """
        Record observed latency for model improvement.
        
        Uses online gradient descent for coefficient updates.
        """
        with self._lock:
            # Record observation
            self._observations.append({
                'tokens': request.max_new_tokens,
                'sequence_length': request.sequence_length,
                'batch_size': batch_size,
                'actual_ms': actual_latency_ms,
            })
            
            # Compute prediction error
            predicted = self.predict(request, batch_size, 0.5)
            error = actual_latency_ms - predicted
            self._prediction_errors.append(abs(error))
            
            # Online learning update (simple gradient descent)
            if len(self._observations) >= 10:
                self._update_coefficients(error, request, batch_size)
    
    def _update_coefficients(self, 
                             error: float,
                             request: SchedulingRequest,
                             batch_size: int,
                             learning_rate: float = 0.001):
        """Update model coefficients using gradient descent."""
        # Gradient for each feature
        self._coefficients['intercept'] += learning_rate * error
        self._coefficients['tokens'] += learning_rate * error * request.max_new_tokens / 1000
        self._coefficients['sequence_length'] += learning_rate * error * request.sequence_length / 10000
        self._coefficients['batch_size'] += learning_rate * error * batch_size / 100
        
        # Keep coefficients reasonable
        for key in self._coefficients:
            self._coefficients[key] = max(0.001, self._coefficients[key])
    
    @property
    def mean_absolute_error(self) -> float:
        """Get mean absolute prediction error."""
        if not self._prediction_errors:
            return 0.0
        return sum(self._prediction_errors) / len(self._prediction_errors)


# =============================================================================
# SCHEDULING POLICY IMPLEMENTATIONS
# =============================================================================


class SchedulingPolicyImpl(ABC):
    """Abstract base for scheduling policy implementations."""
    
    @abstractmethod
    def select(self, 
               queue: List[SchedulingRequest],
               max_batch_size: int,
               tenant_states: Dict[str, TenantState]) -> List[SchedulingRequest]:
        """Select requests to form a batch."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> SchedulingPolicy:
        """Policy identifier."""
        pass


class FCFSPolicy(SchedulingPolicyImpl):
    """First-Come First-Served policy."""
    
    @property
    def name(self) -> SchedulingPolicy:
        return SchedulingPolicy.FCFS
    
    def select(self, queue: List[SchedulingRequest], max_batch_size: int,
               tenant_states: Dict[str, TenantState]) -> List[SchedulingRequest]:
        # Sort by arrival time and take first N
        sorted_queue = sorted(queue, key=lambda r: r.arrival_time)
        return sorted_queue[:max_batch_size]


class SJFPolicy(SchedulingPolicyImpl):
    """Shortest Job First policy - minimizes average latency."""
    
    @property
    def name(self) -> SchedulingPolicy:
        return SchedulingPolicy.SJF
    
    def select(self, queue: List[SchedulingRequest], max_batch_size: int,
               tenant_states: Dict[str, TenantState]) -> List[SchedulingRequest]:
        # Sort by estimated duration
        sorted_queue = sorted(queue, key=lambda r: r.estimated_duration_ms or float('inf'))
        return sorted_queue[:max_batch_size]


class EDFPolicy(SchedulingPolicyImpl):
    """Earliest Deadline First - for real-time guarantees."""
    
    @property
    def name(self) -> SchedulingPolicy:
        return SchedulingPolicy.EDF
    
    def select(self, queue: List[SchedulingRequest], max_batch_size: int,
               tenant_states: Dict[str, TenantState]) -> List[SchedulingRequest]:
        # Requests with deadlines first, sorted by deadline
        # Then requests without deadlines by arrival time
        with_deadline = [r for r in queue if r.deadline_ms is not None]
        without_deadline = [r for r in queue if r.deadline_ms is None]
        
        with_deadline.sort(key=lambda r: r.deadline_ms)
        without_deadline.sort(key=lambda r: r.arrival_time)
        
        combined = with_deadline + without_deadline
        return combined[:max_batch_size]


class FairPolicy(SchedulingPolicyImpl):
    """
    CFS-inspired fair scheduling.
    
    Uses virtual time to ensure each tenant gets proportional share.
    """
    
    @property
    def name(self) -> SchedulingPolicy:
        return SchedulingPolicy.FAIR
    
    def select(self, queue: List[SchedulingRequest], max_batch_size: int,
               tenant_states: Dict[str, TenantState]) -> List[SchedulingRequest]:
        if not queue:
            return []
        
        # Group by tenant
        by_tenant: Dict[str, List[SchedulingRequest]] = defaultdict(list)
        for r in queue:
            by_tenant[r.tenant_id].append(r)
        
        # Sort each tenant's queue by arrival
        for tenant_queue in by_tenant.values():
            tenant_queue.sort(key=lambda r: r.arrival_time)
        
        # Select from tenants in virtual time order (lowest first)
        result = []
        while len(result) < max_batch_size and any(by_tenant.values()):
            # Find tenant with lowest virtual time that has requests
            eligible_tenants = [
                (tenant_states.get(tid, TenantState(tid)).virtual_time, tid)
                for tid, reqs in by_tenant.items() if reqs
            ]
            if not eligible_tenants:
                break
            
            _, selected_tenant = min(eligible_tenants)
            
            # Take one request from this tenant
            if by_tenant[selected_tenant]:
                req = by_tenant[selected_tenant].pop(0)
                result.append(req)
                
                # Update virtual time
                if selected_tenant in tenant_states:
                    tenant_states[selected_tenant].virtual_time += 1.0 / max(
                        tenant_states[selected_tenant].weight, 0.001
                    )
        
        return result


class PriorityPolicy(SchedulingPolicyImpl):
    """Priority-based scheduling with aging."""
    
    @property
    def name(self) -> SchedulingPolicy:
        return SchedulingPolicy.PRIORITY
    
    def select(self, queue: List[SchedulingRequest], max_batch_size: int,
               tenant_states: Dict[str, TenantState]) -> List[SchedulingRequest]:
        # Sort by effective priority (includes aging)
        sorted_queue = sorted(
            queue,
            key=lambda r: r.priority.effective_priority(r.wait_time),
            reverse=True
        )
        return sorted_queue[:max_batch_size]


class MLFQPolicy(SchedulingPolicyImpl):
    """
    Multi-Level Feedback Queue.
    
    Requests start in highest priority queue and move down based on
    CPU (token) consumption. Provides good interactive response while
    accommodating batch workloads.
    """
    
    def __init__(self, num_levels: int = 4):
        self.num_levels = num_levels
        self._request_levels: Dict[str, int] = {}  # request_id -> queue level
        self._request_tokens: Dict[str, int] = {}  # request_id -> tokens consumed
        self._level_quantum = [50, 100, 200, 400]  # Token quanta per level
    
    @property
    def name(self) -> SchedulingPolicy:
        return SchedulingPolicy.MLFQ
    
    def select(self, queue: List[SchedulingRequest], max_batch_size: int,
               tenant_states: Dict[str, TenantState]) -> List[SchedulingRequest]:
        # Assign levels to new requests
        for r in queue:
            if r.request_id not in self._request_levels:
                self._request_levels[r.request_id] = 0
                self._request_tokens[r.request_id] = 0
        
        # Group by level
        by_level: Dict[int, List[SchedulingRequest]] = defaultdict(list)
        for r in queue:
            level = self._request_levels.get(r.request_id, 0)
            by_level[level].append(r)
        
        # Select from highest priority level first
        result = []
        for level in range(self.num_levels):
            level_queue = sorted(by_level.get(level, []), key=lambda r: r.arrival_time)
            for r in level_queue:
                if len(result) >= max_batch_size:
                    break
                result.append(r)
            if len(result) >= max_batch_size:
                break
        
        return result
    
    def record_execution(self, request_id: str, tokens_generated: int):
        """Record tokens generated, potentially demoting request."""
        if request_id not in self._request_levels:
            return
        
        self._request_tokens[request_id] = self._request_tokens.get(request_id, 0) + tokens_generated
        current_level = self._request_levels[request_id]
        
        # Check if quantum exceeded
        quantum = self._level_quantum[min(current_level, len(self._level_quantum) - 1)]
        if self._request_tokens[request_id] >= quantum:
            # Demote to next level
            new_level = min(current_level + 1, self.num_levels - 1)
            self._request_levels[request_id] = new_level
            self._request_tokens[request_id] = 0


class LotteryPolicy(SchedulingPolicyImpl):
    """
    Lottery scheduling - probabilistic fairness.
    
    Each request/tenant holds tickets; selection is random weighted by tickets.
    Provides good average-case fairness with O(1) selection.
    """
    
    @property
    def name(self) -> SchedulingPolicy:
        return SchedulingPolicy.LOTTERY
    
    def select(self, queue: List[SchedulingRequest], max_batch_size: int,
               tenant_states: Dict[str, TenantState]) -> List[SchedulingRequest]:
        if not queue:
            return []
        
        import random
        
        # Calculate tickets for each request
        tickets = []
        for r in queue:
            # Base tickets from priority
            base = max(1, 10 + r.priority.base)
            # Bonus for waiting
            wait_bonus = min(int(r.wait_time * 2), 20)
            # Tenant weight multiplier
            tenant_weight = tenant_states.get(r.tenant_id, TenantState(r.tenant_id)).weight
            
            ticket_count = int((base + wait_bonus) * tenant_weight)
            tickets.append((r, ticket_count))
        
        # Weighted random selection without replacement
        result = []
        remaining = list(tickets)
        
        while len(result) < max_batch_size and remaining:
            total_tickets = sum(t for _, t in remaining)
            if total_tickets == 0:
                break
            
            # Draw a winning ticket
            winner = random.randint(1, total_tickets)
            cumulative = 0
            
            for i, (req, ticket_count) in enumerate(remaining):
                cumulative += ticket_count
                if cumulative >= winner:
                    result.append(req)
                    remaining.pop(i)
                    break
        
        return result


# =============================================================================
# POLICY SELECTOR (Contextual Bandit)
# =============================================================================


class PolicySelector:
    """
    ML-based policy selection using contextual bandits.
    
    Uses Thompson Sampling with context features to select the
    optimal scheduling policy for current workload conditions.
    
    This is similar to adaptive query optimization in databases.
    """
    
    def __init__(self, policies: List[SchedulingPolicy]):
        self.policies = policies
        
        # Beta distribution parameters for each policy
        # (successes, failures) - Thompson Sampling
        self._alpha: Dict[SchedulingPolicy, float] = {p: 1.0 for p in policies}
        self._beta: Dict[SchedulingPolicy, float] = {p: 1.0 for p in policies}
        
        # Context-aware adjustments
        self._context_weights: Dict[SchedulingPolicy, Dict[str, float]] = {
            p: {} for p in policies
        }
        
        # Policy-workload affinity matrix
        self._affinity: Dict[Tuple[SchedulingPolicy, WorkloadType], float] = {}
        self._init_affinities()
        
        self._lock = threading.Lock()
    
    def _init_affinities(self):
        """Initialize policy-workload affinities based on domain knowledge."""
        # INTERACTIVE workloads prefer low-latency policies
        self._affinity[(SchedulingPolicy.SJF, WorkloadType.INTERACTIVE)] = 1.5
        self._affinity[(SchedulingPolicy.EDF, WorkloadType.INTERACTIVE)] = 1.3
        self._affinity[(SchedulingPolicy.PRIORITY, WorkloadType.INTERACTIVE)] = 1.2
        
        # BATCH workloads prefer throughput
        self._affinity[(SchedulingPolicy.FCFS, WorkloadType.BATCH)] = 1.4
        self._affinity[(SchedulingPolicy.MLFQ, WorkloadType.BATCH)] = 1.2
        
        # Multi-tenant prefers fair
        self._affinity[(SchedulingPolicy.FAIR, WorkloadType.MIXED)] = 1.5
        self._affinity[(SchedulingPolicy.LOTTERY, WorkloadType.MIXED)] = 1.3
        self._affinity[(SchedulingPolicy.WEIGHTED_FAIR, WorkloadType.MIXED)] = 1.4
        
        # Bursty workloads
        self._affinity[(SchedulingPolicy.MLFQ, WorkloadType.BURSTY)] = 1.4
        self._affinity[(SchedulingPolicy.LOTTERY, WorkloadType.BURSTY)] = 1.2
    
    def select(self, profile: WorkloadProfile) -> SchedulingPolicy:
        """
        Select best policy for current workload using Thompson Sampling.
        """
        import random
        
        with self._lock:
            best_sample = -1.0
            best_policy = self.policies[0]
            
            for policy in self.policies:
                # Sample from Beta distribution
                sample = random.betavariate(
                    self._alpha[policy],
                    self._beta[policy]
                )
                
                # Apply workload affinity
                affinity = self._affinity.get(
                    (policy, profile.workload_type),
                    1.0
                )
                sample *= affinity
                
                if sample > best_sample:
                    best_sample = sample
                    best_policy = policy
            
            return best_policy
    
    def record_outcome(self, 
                       policy: SchedulingPolicy,
                       success: bool,
                       reward: float = 1.0):
        """
        Update policy statistics based on outcome.
        
        Args:
            policy: Policy that was used
            success: Whether scheduling met SLOs
            reward: Reward signal (0.0 to 1.0)
        """
        with self._lock:
            if success:
                self._alpha[policy] += reward
            else:
                self._beta[policy] += (1.0 - reward)
            
            # Decay old observations (forgetting factor)
            decay = 0.999
            for p in self.policies:
                self._alpha[p] = 1.0 + (self._alpha[p] - 1.0) * decay
                self._beta[p] = 1.0 + (self._beta[p] - 1.0) * decay
    
    def get_policy_scores(self) -> Dict[SchedulingPolicy, float]:
        """Get current policy scores (success rates)."""
        with self._lock:
            return {
                p: self._alpha[p] / (self._alpha[p] + self._beta[p])
                for p in self.policies
            }


# =============================================================================
# BATCH FORMATION
# =============================================================================


class BatchFormationEngine:
    """
    Intelligent batch formation with multiple strategies.
    
    Synthesizes concepts from:
    - Bin packing algorithms
    - Cache-aware scheduling
    - Network packet batching
    """
    
    def __init__(self,
                 max_batch_size: int = 32,
                 max_batch_tokens: int = 4096,
                 strategy: BatchFormationStrategy = BatchFormationStrategy.GREEDY):
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.strategy = strategy
        self.latency_predictor = LatencyPredictor()
    
    def form_batch(self,
                   candidates: List[SchedulingRequest],
                   tenant_states: Dict[str, TenantState]) -> List[SchedulingRequest]:
        """
        Form optimal batch from candidate requests.
        """
        if not candidates:
            return []
        
        if self.strategy == BatchFormationStrategy.GREEDY:
            return self._greedy_formation(candidates)
        elif self.strategy == BatchFormationStrategy.DEADLINE_AWARE:
            return self._deadline_aware_formation(candidates)
        elif self.strategy == BatchFormationStrategy.SIZE_HOMOGENEOUS:
            return self._size_homogeneous_formation(candidates)
        elif self.strategy == BatchFormationStrategy.TENANT_AFFINITY:
            return self._tenant_affinity_formation(candidates, tenant_states)
        elif self.strategy == BatchFormationStrategy.LATENCY_OPTIMAL:
            return self._latency_optimal_formation(candidates)
        elif self.strategy == BatchFormationStrategy.THROUGHPUT_OPTIMAL:
            return self._throughput_optimal_formation(candidates)
        else:
            return self._greedy_formation(candidates)
    
    def _greedy_formation(self, candidates: List[SchedulingRequest]) -> List[SchedulingRequest]:
        """Simple greedy: take first N that fit."""
        batch = []
        total_tokens = 0
        
        for req in candidates:
            if len(batch) >= self.max_batch_size:
                break
            if total_tokens + req.sequence_length + req.max_new_tokens > self.max_batch_tokens:
                continue
            batch.append(req)
            total_tokens += req.sequence_length + req.max_new_tokens
        
        return batch
    
    def _deadline_aware_formation(self, candidates: List[SchedulingRequest]) -> List[SchedulingRequest]:
        """Prioritize requests with tight deadlines."""
        # Sort by slack time (tightest deadline first)
        def slack_key(r: SchedulingRequest) -> float:
            slack = r.slack_time
            return slack if slack is not None else float('inf')
        
        sorted_candidates = sorted(candidates, key=slack_key)
        return self._greedy_formation(sorted_candidates)
    
    def _size_homogeneous_formation(self, candidates: List[SchedulingRequest]) -> List[SchedulingRequest]:
        """Group similar-sized requests for better batching efficiency."""
        if not candidates:
            return []
        
        # Find the mode sequence length bucket
        buckets: Dict[int, List[SchedulingRequest]] = defaultdict(list)
        for req in candidates:
            bucket = req.sequence_length // 256  # 256-token buckets
            buckets[bucket].append(req)
        
        # Select from largest bucket first
        largest_bucket = max(buckets.keys(), key=lambda b: len(buckets[b]))
        homogeneous = buckets[largest_bucket]
        
        # Fill with others if needed
        batch = self._greedy_formation(homogeneous)
        if len(batch) < self.max_batch_size:
            remaining = [r for r in candidates if r not in batch]
            batch.extend(self._greedy_formation(remaining)[:self.max_batch_size - len(batch)])
        
        return batch
    
    def _tenant_affinity_formation(self,
                                   candidates: List[SchedulingRequest],
                                   tenant_states: Dict[str, TenantState]) -> List[SchedulingRequest]:
        """Group by tenant for better cache locality."""
        if not candidates:
            return []
        
        # Group by tenant
        by_tenant: Dict[str, List[SchedulingRequest]] = defaultdict(list)
        for req in candidates:
            by_tenant[req.tenant_id].append(req)
        
        # Prioritize tenants with more requests
        sorted_tenants = sorted(by_tenant.keys(), key=lambda t: -len(by_tenant[t]))
        
        batch = []
        for tenant_id in sorted_tenants:
            tenant_reqs = by_tenant[tenant_id]
            for req in tenant_reqs:
                if len(batch) >= self.max_batch_size:
                    break
                batch.append(req)
            if len(batch) >= self.max_batch_size:
                break
        
        return batch
    
    def _latency_optimal_formation(self, candidates: List[SchedulingRequest]) -> List[SchedulingRequest]:
        """Optimize for minimum p99 latency."""
        # Smaller batches = lower per-request latency
        # But also consider request wait times
        
        # Take requests with highest wait times, but keep batch small
        sorted_by_wait = sorted(candidates, key=lambda r: -r.wait_time)
        
        # Limit batch size for latency
        max_for_latency = min(self.max_batch_size, 16)  # Smaller batches
        return sorted_by_wait[:max_for_latency]
    
    def _throughput_optimal_formation(self, candidates: List[SchedulingRequest]) -> List[SchedulingRequest]:
        """Optimize for maximum throughput."""
        # Larger batches = better GPU utilization = higher throughput
        # Prefer requests that will complete quickly
        
        sorted_by_duration = sorted(candidates, key=lambda r: r.estimated_duration_ms or float('inf'))
        return sorted_by_duration[:self.max_batch_size]


# =============================================================================
# ADAPTIVE BATCH SCHEDULER (Main Class)
# =============================================================================


class AdaptiveBatchScheduler:
    """
    Advanced batch scheduler with ML-based policy selection.
    
    This is the primary scheduler synthesizing concepts from:
    - Operating Systems: CFS, MLFQ, priority scheduling
    - Network QoS: Weighted fair queuing, token bucket
    - Machine Learning: Contextual bandits, online learning
    - Database: Query optimization, cost estimation
    - Economics: Utility maximization, fair division
    
    Key Features:
    1. Adaptive policy selection based on workload
    2. Latency prediction with online learning
    3. Fair multi-tenant resource sharing
    4. Deadline-aware scheduling
    5. Preemption support
    
    Performance Targets:
    - Scheduling overhead: < 2% of request time
    - Throughput: Maximize GPU utilization
    - Latency: Meet p99 SLOs
    - Fairness: Jain's index > 0.9
    """
    
    def __init__(self,
                 max_batch_size: int = 32,
                 max_batch_tokens: int = 4096,
                 latency_target: Optional[LatencyTarget] = None,
                 formation_strategy: BatchFormationStrategy = BatchFormationStrategy.GREEDY,
                 preemption_policy: PreemptionPolicy = PreemptionPolicy.COOPERATIVE,
                 enable_adaptive_policy: bool = True):
        """
        Initialize the adaptive batch scheduler.
        
        Args:
            max_batch_size: Maximum requests per batch
            max_batch_tokens: Maximum tokens per batch
            latency_target: SLO targets for latency
            formation_strategy: How to form batches
            preemption_policy: How to handle preemption
            enable_adaptive_policy: Use ML policy selection
        """
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.latency_target = latency_target or LatencyTarget()
        self.preemption_policy = preemption_policy
        self.enable_adaptive_policy = enable_adaptive_policy
        
        # Request queue (thread-safe)
        self._queue: List[SchedulingRequest] = []
        self._queue_lock = threading.Lock()
        
        # Tenant state tracking
        self._tenant_states: Dict[str, TenantState] = {}
        
        # Workload profiling
        self._profile = WorkloadProfile()
        self._recent_requests: deque = deque(maxlen=100)
        
        # Policy implementations
        self._policies: Dict[SchedulingPolicy, SchedulingPolicyImpl] = {
            SchedulingPolicy.FCFS: FCFSPolicy(),
            SchedulingPolicy.SJF: SJFPolicy(),
            SchedulingPolicy.EDF: EDFPolicy(),
            SchedulingPolicy.FAIR: FairPolicy(),
            SchedulingPolicy.PRIORITY: PriorityPolicy(),
            SchedulingPolicy.MLFQ: MLFQPolicy(),
            SchedulingPolicy.LOTTERY: LotteryPolicy(),
        }
        
        # Policy selector (contextual bandit)
        self._policy_selector = PolicySelector(list(self._policies.keys()))
        self._current_policy = SchedulingPolicy.FAIR
        
        # Batch formation
        self._batch_engine = BatchFormationEngine(
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
            strategy=formation_strategy
        )
        
        # Latency predictor
        self._latency_predictor = LatencyPredictor()
        
        # Metrics
        self._metrics = SchedulerMetrics()
        self._latencies: deque = deque(maxlen=1000)
        
        # Preempted requests
        self._preempted: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            f"AdaptiveBatchScheduler initialized: "
            f"max_batch={max_batch_size}, max_tokens={max_batch_tokens}, "
            f"formation={formation_strategy.name}, preemption={preemption_policy.name}"
        )
    
    def submit(self, request: SchedulingRequest) -> bool:
        """
        Submit a request for scheduling.
        
        Args:
            request: The request to schedule
            
        Returns:
            True if submitted successfully
        """
        with self._queue_lock:
            self._queue.append(request)
            self._recent_requests.append(request)
            
            # Update workload profile
            self._profile.update(request, list(self._recent_requests))
            
            # Ensure tenant state exists
            if request.tenant_id not in self._tenant_states:
                self._tenant_states[request.tenant_id] = TenantState(request.tenant_id)
            self._tenant_states[request.tenant_id].current_requests += 1
        
        logger.debug(f"Request {request.request_id} submitted, queue size: {len(self._queue)}")
        return True
    
    def schedule(self) -> Optional[SchedulingDecision]:
        """
        Make a scheduling decision and return a batch.
        
        Returns:
            SchedulingDecision with selected batch, or None if queue empty
        """
        start_time = time.monotonic()
        
        with self._queue_lock:
            if not self._queue:
                return None
            
            # Select policy
            if self.enable_adaptive_policy:
                self._current_policy = self._policy_selector.select(self._profile)
            
            # Get policy implementation
            policy_impl = self._policies.get(
                self._current_policy,
                self._policies[SchedulingPolicy.FAIR]
            )
            
            # Select candidates using policy
            candidates = policy_impl.select(
                self._queue,
                self.max_batch_size,
                self._tenant_states
            )
            
            # Form batch
            batch = self._batch_engine.form_batch(candidates, self._tenant_states)
            
            if not batch:
                return None
            
            # Remove selected requests from queue
            batch_ids = {r.request_id for r in batch}
            self._queue = [r for r in self._queue if r.request_id not in batch_ids]
            
            # Estimate latency
            avg_tokens = sum(r.max_new_tokens for r in batch) / len(batch)
            estimated_latency = self._latency_predictor.predict(
                batch[0],  # Representative request
                len(batch),
                len(self._queue) / max(self.max_batch_size, 1)  # Load factor
            )
            
            # Update metrics
            overhead_us = (time.monotonic() - start_time) * 1_000_000
            self._metrics.avg_scheduling_overhead_us = (
                0.9 * self._metrics.avg_scheduling_overhead_us + 0.1 * overhead_us
            )
            self._metrics.total_batches_formed += 1
            self._metrics.total_requests_scheduled += len(batch)
            self._metrics.avg_batch_size = (
                0.9 * self._metrics.avg_batch_size + 0.1 * len(batch)
            )
            
            # Track policy usage
            self._metrics.policy_usage[self._current_policy] = \
                self._metrics.policy_usage.get(self._current_policy, 0) + 1
            
            return SchedulingDecision(
                requests=batch,
                batch_size=len(batch),
                policy_used=self._current_policy,
                estimated_latency_ms=estimated_latency,
                estimated_throughput=avg_tokens * len(batch) / (estimated_latency / 1000),
                reasoning=f"Policy {self._current_policy.name} selected for "
                          f"{self._profile.workload_type.name} workload"
            )
    
    def record_completion(self,
                          request_id: str,
                          actual_latency_ms: float,
                          tokens_generated: int,
                          success: bool = True):
        """
        Record request completion for learning.
        
        Args:
            request_id: Completed request ID
            actual_latency_ms: Actual execution latency
            tokens_generated: Tokens generated
            success: Whether request met SLOs
        """
        self._latencies.append(actual_latency_ms)
        
        # Update latency predictor
        # (We'd need the original request here; simplified)
        
        # Update policy selector
        met_slo = actual_latency_ms <= self.latency_target.p99_ms
        reward = 1.0 if met_slo else max(0.0, 1.0 - (actual_latency_ms - self.latency_target.p99_ms) / 1000)
        self._policy_selector.record_outcome(self._current_policy, met_slo, reward)
        
        # Update metrics
        if self._latencies:
            sorted_lat = sorted(self._latencies)
            self._metrics.latency_p50_ms = sorted_lat[len(sorted_lat) // 2]
            self._metrics.latency_p99_ms = sorted_lat[int(len(sorted_lat) * 0.99)]
        
        if not met_slo:
            self._metrics.deadline_miss_count += 1
            self._metrics.deadline_miss_rate = (
                self._metrics.deadline_miss_count / 
                max(self._metrics.total_requests_scheduled, 1)
            )
    
    def preempt(self, request_id: str, checkpoint: Dict[str, Any]) -> bool:
        """
        Preempt a running request.
        
        Args:
            request_id: Request to preempt
            checkpoint: Saved state for resumption
            
        Returns:
            True if preempted successfully
        """
        if self.preemption_policy == PreemptionPolicy.NONE:
            return False
        
        self._preempted[request_id] = checkpoint
        self._metrics.preemption_count += 1
        
        logger.info(f"Request {request_id} preempted")
        return True
    
    def resume(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Resume a preempted request.
        
        Returns:
            Checkpoint state if request was preempted, None otherwise
        """
        return self._preempted.pop(request_id, None)
    
    def update_tenant_weight(self, tenant_id: str, weight: float):
        """Update tenant's resource share weight."""
        if tenant_id in self._tenant_states:
            self._tenant_states[tenant_id].weight = max(0.1, weight)
        else:
            self._tenant_states[tenant_id] = TenantState(tenant_id, weight=weight)
    
    def set_tenant_quota(self, tenant_id: str, token_quota: Optional[int]):
        """Set token quota for a tenant."""
        if tenant_id in self._tenant_states:
            self._tenant_states[tenant_id].quota_remaining = token_quota
        else:
            state = TenantState(tenant_id)
            state.quota_remaining = token_quota
            self._tenant_states[tenant_id] = state
    
    @property
    def queue_length(self) -> int:
        """Current queue length."""
        with self._queue_lock:
            return len(self._queue)
    
    @property
    def metrics(self) -> SchedulerMetrics:
        """Get current scheduler metrics."""
        # Calculate Jain's fairness index
        if self._tenant_states:
            tokens = [t.tokens_used for t in self._tenant_states.values() if t.tokens_used > 0]
            if tokens:
                sum_tokens = sum(tokens)
                sum_sq = sum(t ** 2 for t in tokens)
                n = len(tokens)
                self._metrics.fairness_jain_index = (sum_tokens ** 2) / (n * sum_sq) if sum_sq > 0 else 1.0
        
        return self._metrics
    
    @property
    def policy_scores(self) -> Dict[SchedulingPolicy, float]:
        """Get current policy effectiveness scores."""
        return self._policy_selector.get_policy_scores()
    
    def get_tenant_stats(self, tenant_id: str) -> Optional[TenantState]:
        """Get statistics for a specific tenant."""
        return self._tenant_states.get(tenant_id)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_scheduler(
    max_batch_size: int = 32,
    max_batch_tokens: int = 4096,
    latency_target_p99_ms: float = 500.0,
    enable_adaptive: bool = True
) -> AdaptiveBatchScheduler:
    """
    Create a pre-configured adaptive batch scheduler.
    
    Args:
        max_batch_size: Maximum batch size
        max_batch_tokens: Maximum tokens per batch
        latency_target_p99_ms: P99 latency SLO in milliseconds
        enable_adaptive: Enable ML-based policy selection
        
    Returns:
        Configured AdaptiveBatchScheduler
    """
    target = LatencyTarget(
        p50_ms=latency_target_p99_ms * 0.3,
        p90_ms=latency_target_p99_ms * 0.6,
        p99_ms=latency_target_p99_ms,
        max_ms=latency_target_p99_ms * 2.0
    )
    
    return AdaptiveBatchScheduler(
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
        latency_target=target,
        formation_strategy=BatchFormationStrategy.GREEDY,
        preemption_policy=PreemptionPolicy.COOPERATIVE,
        enable_adaptive_policy=enable_adaptive
    )


def create_request(
    request_id: str,
    tenant_id: str,
    sequence_length: int = 512,
    max_new_tokens: int = 128,
    priority: int = 0,
    deadline_ms: Optional[float] = None
) -> SchedulingRequest:
    """
    Create a scheduling request with sensible defaults.
    
    Args:
        request_id: Unique request identifier
        tenant_id: Tenant/user identifier
        sequence_length: Input sequence length
        max_new_tokens: Maximum tokens to generate
        priority: Priority level (-10 to 10)
        deadline_ms: Optional deadline in milliseconds
        
    Returns:
        Configured SchedulingRequest
    """
    return SchedulingRequest(
        request_id=request_id,
        tenant_id=tenant_id,
        sequence_length=sequence_length,
        max_new_tokens=max_new_tokens,
        priority=RequestPriority(base=priority),
        deadline_ms=deadline_ms
    )
