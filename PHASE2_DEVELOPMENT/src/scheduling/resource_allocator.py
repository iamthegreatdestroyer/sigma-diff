"""
Resource Allocator Module
Sprint 4.3: Advanced Scheduling & Resource Management

Cross-Domain Synthesis:
- Cloud Computing: Multi-tenant isolation, resource quotas, SLA enforcement
- Economics: Dominant Resource Fairness (DRF), utility maximization, auction mechanisms
- Operating Systems: Hierarchical scheduling, cgroups-inspired isolation
- Real-Time Systems: Resource reservation, admission control
- Network QoS: Bandwidth allocation, traffic shaping, policing
- Game Theory: Fair division, envy-freeness, proportional allocation

Features:
- Multi-tenant resource isolation with strong guarantees
- Hierarchical resource management (cluster -> node -> GPU -> partition)
- Dynamic resource rebalancing with minimal disruption
- Fair share allocation using Dominant Resource Fairness (DRF)
- Resource quotas with soft/hard limits and burst capacity
- Admission control with resource reservation
- SLA-aware allocation with priority classes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from collections import defaultdict
import threading
import time
import math
import heapq
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums for Resource Management
# =============================================================================

class ResourceType(Enum):
    """Types of resources that can be allocated."""
    GPU_COMPUTE = auto()      # GPU compute cycles (SMs)
    GPU_MEMORY = auto()       # GPU memory (bytes)
    CPU_COMPUTE = auto()      # CPU cycles
    SYSTEM_MEMORY = auto()    # System RAM
    NETWORK_BANDWIDTH = auto() # Network I/O
    STORAGE_IOPS = auto()     # Storage operations
    KV_CACHE = auto()         # KV cache slots
    BATCH_SLOTS = auto()      # Concurrent request slots


class AllocationMode(Enum):
    """Resource allocation modes."""
    EXCLUSIVE = auto()     # Tenant gets exclusive access
    SHARED = auto()        # Resources shared with isolation
    BEST_EFFORT = auto()   # No guarantees, opportunistic
    RESERVED = auto()      # Pre-reserved, guaranteed availability
    BURST = auto()         # Can exceed quota temporarily


class IsolationLevel(Enum):
    """Tenant isolation levels (inspired by database isolation)."""
    NONE = auto()          # No isolation (single-tenant mode)
    SOFT = auto()          # Logical isolation, shared resources
    MEDIUM = auto()        # Resource limits enforced
    STRICT = auto()        # Strong isolation, dedicated resources
    DEDICATED = auto()     # Completely isolated hardware


class PriorityClass(Enum):
    """Priority classes for resource allocation."""
    SYSTEM = 0       # System-level, highest priority
    CRITICAL = 1     # Business-critical workloads
    HIGH = 2         # High-priority production
    NORMAL = 3       # Standard workloads
    LOW = 4          # Background/batch jobs
    SCAVENGER = 5    # Uses only spare capacity


class QuotaEnforcement(Enum):
    """How quotas are enforced."""
    NONE = auto()          # No enforcement
    SOFT = auto()          # Warnings only
    HARD = auto()          # Strict enforcement
    THROTTLE = auto()      # Slow down on exceed
    PREEMPT = auto()       # Preempt on exceed


class RebalanceStrategy(Enum):
    """Strategies for resource rebalancing."""
    NONE = auto()           # No rebalancing
    GRADUAL = auto()        # Slow, minimal disruption
    AGGRESSIVE = auto()     # Fast, may cause disruption
    OPPORTUNISTIC = auto()  # Rebalance when idle
    SCHEDULED = auto()      # Rebalance at specific times


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResourceCapacity:
    """Represents the capacity of a resource."""
    resource_type: ResourceType
    total: float
    available: float = 0.0
    reserved: float = 0.0
    
    def __post_init__(self):
        if self.available == 0.0:
            self.available = self.total
    
    @property
    def used(self) -> float:
        return self.total - self.available
    
    @property
    def utilization(self) -> float:
        return self.used / self.total if self.total > 0 else 0.0
    
    def can_allocate(self, amount: float) -> bool:
        return self.available >= amount


@dataclass
class ResourceQuota:
    """Resource quota for a tenant or workload."""
    tenant_id: str
    resource_type: ResourceType
    guaranteed: float = 0.0      # Minimum guaranteed amount
    limit: float = float('inf')  # Maximum allowed
    burst_limit: float = 0.0     # Extra burst capacity
    burst_duration_ms: float = 1000.0  # How long burst allowed
    enforcement: QuotaEnforcement = QuotaEnforcement.HARD
    
    @property
    def effective_limit(self) -> float:
        return self.limit + self.burst_limit


@dataclass
class ResourceRequest:
    """Request for resource allocation."""
    request_id: str
    tenant_id: str
    resources: Dict[ResourceType, float]
    priority: PriorityClass = PriorityClass.NORMAL
    mode: AllocationMode = AllocationMode.SHARED
    isolation: IsolationLevel = IsolationLevel.MEDIUM
    deadline_ms: Optional[float] = None
    preemptible: bool = True
    affinity_hints: List[str] = field(default_factory=list)
    anti_affinity: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class AllocationGrant:
    """Result of a resource allocation."""
    request_id: str
    tenant_id: str
    granted: Dict[ResourceType, float]
    node_id: str
    partition_id: Optional[str] = None
    granted_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    preemptible: bool = True
    quality_score: float = 1.0  # 0-1, how well request was satisfied


@dataclass
class TenantState:
    """State tracking for a tenant."""
    tenant_id: str
    quotas: Dict[ResourceType, ResourceQuota] = field(default_factory=dict)
    allocations: Dict[str, AllocationGrant] = field(default_factory=dict)
    priority_class: PriorityClass = PriorityClass.NORMAL
    isolation_level: IsolationLevel = IsolationLevel.MEDIUM
    created_at: float = field(default_factory=time.time)
    burst_usage: Dict[ResourceType, Tuple[float, float]] = field(default_factory=dict)
    
    def get_total_allocated(self, resource_type: ResourceType) -> float:
        """Get total resources allocated to this tenant."""
        return sum(
            grant.granted.get(resource_type, 0.0)
            for grant in self.allocations.values()
        )
    
    def is_over_quota(self, resource_type: ResourceType) -> bool:
        """Check if tenant is over quota for a resource."""
        if resource_type not in self.quotas:
            return False
        quota = self.quotas[resource_type]
        allocated = self.get_total_allocated(resource_type)
        return allocated > quota.limit


@dataclass
class NodeResources:
    """Resources available on a node."""
    node_id: str
    capacities: Dict[ResourceType, ResourceCapacity] = field(default_factory=dict)
    allocations: Dict[str, AllocationGrant] = field(default_factory=dict)
    health_score: float = 1.0
    load_factor: float = 0.0
    
    def get_available(self, resource_type: ResourceType) -> float:
        """Get available capacity for a resource type."""
        if resource_type in self.capacities:
            return self.capacities[resource_type].available
        return 0.0
    
    def get_utilization(self, resource_type: ResourceType) -> float:
        """Get utilization for a resource type."""
        if resource_type in self.capacities:
            return self.capacities[resource_type].utilization
        return 0.0


@dataclass
class AllocationMetrics:
    """Metrics for the resource allocator."""
    total_requests: int = 0
    successful_allocations: int = 0
    failed_allocations: int = 0
    preemptions: int = 0
    rebalances: int = 0
    quota_violations: int = 0
    total_allocation_time_ms: float = 0.0
    fairness_index: float = 1.0
    utilization_by_resource: Dict[ResourceType, float] = field(default_factory=dict)


# =============================================================================
# Fair Share Algorithms
# =============================================================================

class FairShareAlgorithm(ABC):
    """Base class for fair share allocation algorithms."""
    
    @abstractmethod
    def compute_shares(
        self,
        demands: Dict[str, Dict[ResourceType, float]],
        capacities: Dict[ResourceType, float]
    ) -> Dict[str, Dict[ResourceType, float]]:
        """Compute fair share allocations for all tenants."""
        pass


class MaxMinFairShare(FairShareAlgorithm):
    """
    Max-Min Fair Share algorithm.
    
    Progressively saturates demands from smallest to largest,
    ensuring no tenant is starved.
    """
    
    def compute_shares(
        self,
        demands: Dict[str, Dict[ResourceType, float]],
        capacities: Dict[ResourceType, float]
    ) -> Dict[str, Dict[ResourceType, float]]:
        shares: Dict[str, Dict[ResourceType, float]] = {
            tenant_id: {} for tenant_id in demands
        }
        
        for resource_type, capacity in capacities.items():
            # Get demands for this resource
            resource_demands = [
                (tenant_id, d.get(resource_type, 0.0))
                for tenant_id, d in demands.items()
            ]
            
            # Sort by demand (ascending)
            resource_demands.sort(key=lambda x: x[1])
            
            remaining_capacity = capacity
            remaining_tenants = len(resource_demands)
            
            for tenant_id, demand in resource_demands:
                if remaining_tenants == 0:
                    break
                
                # Fair share of remaining capacity
                fair_share = remaining_capacity / remaining_tenants
                
                # Allocate min of demand and fair share
                allocation = min(demand, fair_share)
                shares[tenant_id][resource_type] = allocation
                
                remaining_capacity -= allocation
                remaining_tenants -= 1
        
        return shares


class DominantResourceFairness(FairShareAlgorithm):
    """
    Dominant Resource Fairness (DRF) algorithm.
    
    From Ghodsi et al., "Dominant Resource Fairness: Fair Allocation
    of Multiple Resource Types" (NSDI 2011).
    
    Key insight: For multi-resource fairness, equalize the dominant
    resource share (the resource where a tenant's share is largest
    relative to demand).
    """
    
    def __init__(self, max_iterations: int = 100):
        self.max_iterations = max_iterations
    
    def compute_shares(
        self,
        demands: Dict[str, Dict[ResourceType, float]],
        capacities: Dict[ResourceType, float]
    ) -> Dict[str, Dict[ResourceType, float]]:
        if not demands:
            return {}
        
        # Initialize allocations to zero
        allocations: Dict[str, Dict[ResourceType, float]] = {
            tenant_id: {rt: 0.0 for rt in capacities}
            for tenant_id in demands
        }
        
        # Track remaining capacity
        remaining = {rt: cap for rt, cap in capacities.items()}
        
        # Track which tenants are saturated (got all they demanded)
        saturated: Set[str] = set()
        
        for _ in range(self.max_iterations):
            # Find tenant with minimum dominant share
            min_dominant_share = float('inf')
            min_tenant = None
            min_resource = None
            
            for tenant_id, demand in demands.items():
                if tenant_id in saturated:
                    continue
                
                # Compute dominant resource share for this tenant
                dominant_share = 0.0
                dominant_resource = None
                
                for resource_type, demand_val in demand.items():
                    if demand_val <= 0 or resource_type not in capacities:
                        continue
                    
                    current_alloc = allocations[tenant_id].get(resource_type, 0.0)
                    share = current_alloc / capacities[resource_type]
                    
                    if share > dominant_share:
                        dominant_share = share
                        dominant_resource = resource_type
                
                if dominant_resource is not None:
                    if dominant_share < min_dominant_share:
                        min_dominant_share = dominant_share
                        min_tenant = tenant_id
                        min_resource = dominant_resource
            
            if min_tenant is None:
                break  # All tenants saturated
            
            # Give the minimum tenant one unit of their demanded resources
            tenant_demand = demands[min_tenant]
            can_allocate = True
            
            for resource_type, demand_val in tenant_demand.items():
                current = allocations[min_tenant].get(resource_type, 0.0)
                needed = demand_val - current
                
                if needed > 0:
                    # Allocate proportionally
                    increment = min(needed, remaining.get(resource_type, 0.0))
                    if increment > 0:
                        allocations[min_tenant][resource_type] = current + increment
                        remaining[resource_type] -= increment
                    
                    if current + increment < demand_val and remaining.get(resource_type, 0.0) <= 0:
                        can_allocate = False
            
            # Check if tenant is now saturated
            is_saturated = all(
                allocations[min_tenant].get(rt, 0.0) >= d
                for rt, d in tenant_demand.items()
            )
            
            if is_saturated or not can_allocate:
                saturated.add(min_tenant)
            
            # Check if all resources exhausted
            if all(r <= 0 for r in remaining.values()):
                break
        
        return allocations


class WeightedFairShare(FairShareAlgorithm):
    """
    Weighted Fair Share algorithm.
    
    Allocates resources proportionally to tenant weights,
    respecting minimum guarantees.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {}
    
    def compute_shares(
        self,
        demands: Dict[str, Dict[ResourceType, float]],
        capacities: Dict[ResourceType, float]
    ) -> Dict[str, Dict[ResourceType, float]]:
        if not demands:
            return {}
        
        # Get weights (default to 1.0)
        weights = {
            tenant_id: self.weights.get(tenant_id, 1.0)
            for tenant_id in demands
        }
        total_weight = sum(weights.values())
        
        shares: Dict[str, Dict[ResourceType, float]] = {
            tenant_id: {} for tenant_id in demands
        }
        
        for resource_type, capacity in capacities.items():
            # Compute weighted fair shares
            for tenant_id, demand in demands.items():
                weight_fraction = weights[tenant_id] / total_weight
                fair_share = capacity * weight_fraction
                actual_demand = demand.get(resource_type, 0.0)
                shares[tenant_id][resource_type] = min(fair_share, actual_demand)
        
        return shares


# =============================================================================
# Admission Control
# =============================================================================

class AdmissionController:
    """
    Admission control for resource requests.
    
    Determines whether a request can be admitted based on
    available resources, quotas, and SLAs.
    """
    
    def __init__(
        self,
        overbooking_factor: float = 1.2,  # Allow 20% overbooking
        reservation_fraction: float = 0.1  # Reserve 10% for high-priority
    ):
        self.overbooking_factor = overbooking_factor
        self.reservation_fraction = reservation_fraction
    
    def can_admit(
        self,
        request: ResourceRequest,
        node: NodeResources,
        tenant: TenantState
    ) -> Tuple[bool, str]:
        """
        Check if a request can be admitted.
        
        Returns:
            Tuple of (can_admit, reason)
        """
        # Check each requested resource
        for resource_type, amount in request.resources.items():
            # Check node capacity
            available = node.get_available(resource_type)
            effective_available = available * self.overbooking_factor
            
            # Reserve some for high-priority requests
            if request.priority.value > PriorityClass.HIGH.value:
                capacity = node.capacities.get(resource_type)
                if capacity:
                    reserved = capacity.total * self.reservation_fraction
                    effective_available -= reserved
            
            if amount > effective_available:
                return False, f"Insufficient {resource_type.name}: need {amount}, have {effective_available}"
            
            # Check tenant quota
            if resource_type in tenant.quotas:
                quota = tenant.quotas[resource_type]
                current_usage = tenant.get_total_allocated(resource_type)
                
                if current_usage + amount > quota.effective_limit:
                    if quota.enforcement == QuotaEnforcement.HARD:
                        return False, f"Quota exceeded for {resource_type.name}"
                    elif quota.enforcement == QuotaEnforcement.PREEMPT:
                        return False, f"Would require preemption for {resource_type.name}"
        
        return True, "Admitted"
    
    def compute_admission_score(
        self,
        request: ResourceRequest,
        node: NodeResources,
        tenant: TenantState
    ) -> float:
        """
        Compute an admission score (0-1) for ranking requests.
        
        Higher score = more likely to admit.
        """
        score = 1.0
        
        # Priority factor
        priority_weight = 1.0 - (request.priority.value / 10.0)
        score *= (0.5 + 0.5 * priority_weight)
        
        # Resource fit factor (prefer requests that fit well)
        for resource_type, amount in request.resources.items():
            available = node.get_available(resource_type)
            if available > 0:
                fit = 1.0 - abs(amount / available - 0.5)  # Prefer 50% utilization
                score *= (0.7 + 0.3 * fit)
        
        # Quota headroom factor
        for resource_type, amount in request.resources.items():
            if resource_type in tenant.quotas:
                quota = tenant.quotas[resource_type]
                current = tenant.get_total_allocated(resource_type)
                headroom = (quota.limit - current) / quota.limit if quota.limit > 0 else 1.0
                score *= (0.6 + 0.4 * headroom)
        
        return max(0.0, min(1.0, score))


# =============================================================================
# Resource Preemption
# =============================================================================

class PreemptionManager:
    """
    Manages resource preemption for higher-priority requests.
    """
    
    def __init__(
        self,
        grace_period_ms: float = 5000.0,
        min_runtime_before_preempt_ms: float = 10000.0
    ):
        self.grace_period_ms = grace_period_ms
        self.min_runtime_before_preempt_ms = min_runtime_before_preempt_ms
    
    def find_preemption_victims(
        self,
        request: ResourceRequest,
        node: NodeResources,
        tenants: Dict[str, TenantState]
    ) -> List[str]:
        """
        Find allocations that can be preempted to satisfy request.
        
        Returns list of allocation IDs to preempt.
        """
        victims: List[Tuple[float, str]] = []  # (score, allocation_id)
        
        # Calculate resources needed
        needed = dict(request.resources)
        
        for alloc_id, grant in node.allocations.items():
            # Can't preempt our own allocations
            if grant.tenant_id == request.tenant_id:
                continue
            
            # Can't preempt non-preemptible allocations
            if not grant.preemptible:
                continue
            
            # Can't preempt higher or equal priority
            tenant = tenants.get(grant.tenant_id)
            if tenant and tenant.priority_class.value <= request.priority.value:
                continue
            
            # Check runtime
            runtime_ms = (time.time() - grant.granted_at) * 1000
            if runtime_ms < self.min_runtime_before_preempt_ms:
                continue
            
            # Score this victim (lower is better to preempt)
            score = self._compute_victim_score(grant, tenant, needed)
            victims.append((score, alloc_id))
        
        # Sort by score (preempt lowest scores first)
        victims.sort(key=lambda x: x[0])
        
        # Select victims until we have enough resources
        selected: List[str] = []
        freed: Dict[ResourceType, float] = defaultdict(float)
        
        for score, alloc_id in victims:
            grant = node.allocations[alloc_id]
            
            # Add freed resources
            for rt, amount in grant.granted.items():
                freed[rt] += amount
            
            selected.append(alloc_id)
            
            # Check if we have enough
            have_enough = all(
                freed.get(rt, 0.0) >= amount
                for rt, amount in needed.items()
            )
            
            if have_enough:
                break
        
        return selected
    
    def _compute_victim_score(
        self,
        grant: AllocationGrant,
        tenant: Optional[TenantState],
        needed: Dict[ResourceType, float]
    ) -> float:
        """
        Compute preemption score for a victim.
        
        Lower score = more likely to be preempted.
        """
        score = 1.0
        
        # Priority factor (lower priority = lower score)
        if tenant:
            score *= (1.0 + tenant.priority_class.value / 10.0)
        
        # Age factor (older allocations slightly protected)
        age_ms = (time.time() - grant.granted_at) * 1000
        age_factor = min(age_ms / 60000.0, 1.0)  # Max at 1 minute
        score *= (1.0 + 0.3 * age_factor)
        
        # Resource overlap (prefer victims that free what we need)
        overlap = 0.0
        for rt, amount in grant.granted.items():
            if rt in needed and needed[rt] > 0:
                overlap += min(amount, needed[rt]) / needed[rt]
        
        if overlap > 0:
            score *= (0.5 + 0.5 / overlap)
        
        return score


# =============================================================================
# Resource Rebalancer
# =============================================================================

class ResourceRebalancer:
    """
    Handles dynamic resource rebalancing across the cluster.
    """
    
    def __init__(
        self,
        strategy: RebalanceStrategy = RebalanceStrategy.GRADUAL,
        imbalance_threshold: float = 0.2,  # 20% imbalance triggers rebalance
        rebalance_interval_ms: float = 60000.0
    ):
        self.strategy = strategy
        self.imbalance_threshold = imbalance_threshold
        self.rebalance_interval_ms = rebalance_interval_ms
        self.last_rebalance = 0.0
    
    def should_rebalance(
        self,
        nodes: Dict[str, NodeResources]
    ) -> bool:
        """Check if rebalancing is needed."""
        if self.strategy == RebalanceStrategy.NONE:
            return False
        
        # Check time since last rebalance
        now = time.time() * 1000
        if now - self.last_rebalance < self.rebalance_interval_ms:
            return False
        
        # Check imbalance across nodes
        for resource_type in ResourceType:
            utilizations = [
                node.get_utilization(resource_type)
                for node in nodes.values()
                if resource_type in node.capacities
            ]
            
            if len(utilizations) < 2:
                continue
            
            max_util = max(utilizations)
            min_util = min(utilizations)
            
            if max_util - min_util > self.imbalance_threshold:
                return True
        
        return False
    
    def compute_rebalance_plan(
        self,
        nodes: Dict[str, NodeResources],
        tenants: Dict[str, TenantState]
    ) -> List[Tuple[str, str, str]]:
        """
        Compute a rebalancing plan.
        
        Returns:
            List of (allocation_id, source_node, target_node) migrations
        """
        migrations: List[Tuple[str, str, str]] = []
        
        if self.strategy == RebalanceStrategy.NONE:
            return migrations
        
        # Find overloaded and underloaded nodes
        overloaded: List[Tuple[float, str]] = []
        underloaded: List[Tuple[float, str]] = []
        
        for node_id, node in nodes.items():
            # Compute average utilization
            utils = [
                cap.utilization
                for cap in node.capacities.values()
            ]
            
            if utils:
                avg_util = sum(utils) / len(utils)
                if avg_util > 0.8:
                    overloaded.append((avg_util, node_id))
                elif avg_util < 0.3:
                    underloaded.append((avg_util, node_id))
        
        # Sort: most overloaded first, least loaded first
        overloaded.sort(reverse=True)
        underloaded.sort()
        
        # Find migrations
        for _, source_id in overloaded:
            source = nodes[source_id]
            
            # Find preemptible allocations
            moveable = [
                (alloc_id, grant)
                for alloc_id, grant in source.allocations.items()
                if grant.preemptible
            ]
            
            if not moveable:
                continue
            
            # Sort by size (move smaller ones first for gradual)
            moveable.sort(key=lambda x: sum(x[1].granted.values()))
            
            for alloc_id, grant in moveable:
                # Find a target node
                for _, target_id in underloaded:
                    target = nodes[target_id]
                    
                    # Check if target can accommodate
                    can_fit = all(
                        target.get_available(rt) >= amount
                        for rt, amount in grant.granted.items()
                    )
                    
                    if can_fit:
                        migrations.append((alloc_id, source_id, target_id))
                        break
                
                # Limit migrations per rebalance cycle
                if len(migrations) >= 5:
                    break
            
            if len(migrations) >= 5:
                break
        
        return migrations


# =============================================================================
# Main Resource Allocator
# =============================================================================

class ResourceAllocator:
    """
    Advanced resource allocator with multi-tenant isolation and fair sharing.
    
    Features:
    - Dominant Resource Fairness (DRF) allocation
    - Multi-level isolation (soft, medium, strict)
    - Dynamic quota management with burst capacity
    - Admission control with overbooking
    - Preemption for high-priority requests
    - Automatic rebalancing across nodes
    
    Performance Targets:
    - GPU memory utilization > 85%
    - Scheduling overhead < 2%
    - Fair allocation (Jain's index > 0.9)
    """
    
    def __init__(
        self,
        fair_share_algo: Optional[FairShareAlgorithm] = None,
        admission_controller: Optional[AdmissionController] = None,
        preemption_manager: Optional[PreemptionManager] = None,
        rebalancer: Optional[ResourceRebalancer] = None
    ):
        # Core components
        self.fair_share = fair_share_algo or DominantResourceFairness()
        self.admission = admission_controller or AdmissionController()
        self.preemption = preemption_manager or PreemptionManager()
        self.rebalancer = rebalancer or ResourceRebalancer()
        
        # State
        self.nodes: Dict[str, NodeResources] = {}
        self.tenants: Dict[str, TenantState] = {}
        self.pending_requests: List[ResourceRequest] = []
        self.metrics = AllocationMetrics()
        
        # Synchronization
        self._lock = threading.RLock()
        
        # Callbacks
        self._allocation_callbacks: List[Callable[[AllocationGrant], None]] = []
        self._release_callbacks: List[Callable[[str], None]] = []
    
    # -------------------------------------------------------------------------
    # Node Management
    # -------------------------------------------------------------------------
    
    def register_node(
        self,
        node_id: str,
        capacities: Dict[ResourceType, float]
    ) -> None:
        """Register a new node with the allocator."""
        with self._lock:
            resource_caps = {
                rt: ResourceCapacity(resource_type=rt, total=cap)
                for rt, cap in capacities.items()
            }
            
            self.nodes[node_id] = NodeResources(
                node_id=node_id,
                capacities=resource_caps
            )
            
            logger.info(f"Registered node {node_id} with capacities: {capacities}")
    
    def unregister_node(self, node_id: str) -> List[str]:
        """
        Unregister a node, returning affected allocation IDs.
        """
        with self._lock:
            if node_id not in self.nodes:
                return []
            
            node = self.nodes[node_id]
            affected = list(node.allocations.keys())
            
            # Release all allocations on this node
            for alloc_id in affected:
                self._release_allocation_internal(alloc_id, node_id)
            
            del self.nodes[node_id]
            logger.info(f"Unregistered node {node_id}, affected {len(affected)} allocations")
            
            return affected
    
    def update_node_health(self, node_id: str, health_score: float) -> None:
        """Update a node's health score."""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].health_score = health_score
    
    # -------------------------------------------------------------------------
    # Tenant Management
    # -------------------------------------------------------------------------
    
    def register_tenant(
        self,
        tenant_id: str,
        priority: PriorityClass = PriorityClass.NORMAL,
        isolation: IsolationLevel = IsolationLevel.MEDIUM,
        quotas: Optional[Dict[ResourceType, Tuple[float, float]]] = None
    ) -> None:
        """
        Register a new tenant.
        
        Args:
            tenant_id: Unique tenant identifier
            priority: Default priority class
            isolation: Isolation level
            quotas: Dict of resource_type -> (guaranteed, limit)
        """
        with self._lock:
            tenant = TenantState(
                tenant_id=tenant_id,
                priority_class=priority,
                isolation_level=isolation
            )
            
            if quotas:
                for rt, (guaranteed, limit) in quotas.items():
                    tenant.quotas[rt] = ResourceQuota(
                        tenant_id=tenant_id,
                        resource_type=rt,
                        guaranteed=guaranteed,
                        limit=limit
                    )
            
            self.tenants[tenant_id] = tenant
            logger.info(f"Registered tenant {tenant_id} with priority {priority.name}")
    
    def update_tenant_quota(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        guaranteed: float,
        limit: float,
        burst_limit: float = 0.0
    ) -> None:
        """Update a tenant's quota for a resource type."""
        with self._lock:
            if tenant_id not in self.tenants:
                raise ValueError(f"Unknown tenant: {tenant_id}")
            
            tenant = self.tenants[tenant_id]
            tenant.quotas[resource_type] = ResourceQuota(
                tenant_id=tenant_id,
                resource_type=resource_type,
                guaranteed=guaranteed,
                limit=limit,
                burst_limit=burst_limit
            )
    
    # -------------------------------------------------------------------------
    # Resource Allocation
    # -------------------------------------------------------------------------
    
    def allocate(
        self,
        request: ResourceRequest
    ) -> Optional[AllocationGrant]:
        """
        Allocate resources for a request.
        
        Returns:
            AllocationGrant if successful, None otherwise
        """
        start_time = time.time()
        self.metrics.total_requests += 1
        
        with self._lock:
            # Ensure tenant exists
            if request.tenant_id not in self.tenants:
                self.register_tenant(request.tenant_id)
            
            tenant = self.tenants[request.tenant_id]
            
            # Find best node
            best_node = self._find_best_node(request, tenant)
            
            if best_node is None:
                # Try preemption if allowed
                if request.priority.value <= PriorityClass.HIGH.value:
                    best_node = self._try_preemption(request, tenant)
            
            if best_node is None:
                self.metrics.failed_allocations += 1
                logger.warning(f"Failed to allocate resources for {request.request_id}")
                return None
            
            # Perform allocation
            grant = self._perform_allocation(request, best_node, tenant)
            
            if grant:
                self.metrics.successful_allocations += 1
                
                # Notify callbacks
                for callback in self._allocation_callbacks:
                    try:
                        callback(grant)
                    except Exception as e:
                        logger.error(f"Allocation callback error: {e}")
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.total_allocation_time_ms += elapsed_ms
            
            return grant
    
    def _find_best_node(
        self,
        request: ResourceRequest,
        tenant: TenantState
    ) -> Optional[NodeResources]:
        """Find the best node for a request."""
        candidates: List[Tuple[float, NodeResources]] = []
        
        for node in self.nodes.values():
            # Check admission
            can_admit, reason = self.admission.can_admit(request, node, tenant)
            
            if not can_admit:
                continue
            
            # Score this node
            score = self._score_node(request, node, tenant)
            candidates.append((score, node))
        
        if not candidates:
            return None
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    def _score_node(
        self,
        request: ResourceRequest,
        node: NodeResources,
        tenant: TenantState
    ) -> float:
        """Score a node for a request."""
        score = 1.0
        
        # Health factor
        score *= node.health_score
        
        # Resource fit factor
        for rt, amount in request.resources.items():
            available = node.get_available(rt)
            if available > 0:
                # Prefer nodes where request uses 50-70% of available
                utilization = amount / available
                if 0.5 <= utilization <= 0.7:
                    score *= 1.2
                elif utilization > 0.9:
                    score *= 0.8
        
        # Affinity hints
        for hint in request.affinity_hints:
            if hint in node.node_id:
                score *= 1.3
        
        # Anti-affinity
        for anti in request.anti_affinity:
            if anti in node.node_id:
                score *= 0.5
        
        # Existing allocations from same tenant (bin packing)
        tenant_allocs = sum(
            1 for g in node.allocations.values()
            if g.tenant_id == request.tenant_id
        )
        score *= (1.0 + 0.1 * min(tenant_allocs, 3))
        
        return score
    
    def _try_preemption(
        self,
        request: ResourceRequest,
        tenant: TenantState
    ) -> Optional[NodeResources]:
        """Try to find a node by preempting lower-priority allocations."""
        for node in self.nodes.values():
            victims = self.preemption.find_preemption_victims(
                request, node, self.tenants
            )
            
            if victims:
                # Preempt victims
                for alloc_id in victims:
                    self._preempt_allocation(alloc_id, node.node_id)
                
                self.metrics.preemptions += len(victims)
                return node
        
        return None
    
    def _preempt_allocation(self, alloc_id: str, node_id: str) -> None:
        """Preempt an allocation."""
        node = self.nodes.get(node_id)
        if not node or alloc_id not in node.allocations:
            return
        
        grant = node.allocations[alloc_id]
        tenant = self.tenants.get(grant.tenant_id)
        
        # Release resources
        for rt, amount in grant.granted.items():
            if rt in node.capacities:
                node.capacities[rt].available += amount
        
        # Remove from node and tenant
        del node.allocations[alloc_id]
        if tenant and alloc_id in tenant.allocations:
            del tenant.allocations[alloc_id]
        
        logger.info(f"Preempted allocation {alloc_id} on node {node_id}")
    
    def _perform_allocation(
        self,
        request: ResourceRequest,
        node: NodeResources,
        tenant: TenantState
    ) -> AllocationGrant:
        """Perform the actual resource allocation."""
        # Deduct resources
        granted = {}
        for rt, amount in request.resources.items():
            if rt in node.capacities:
                node.capacities[rt].available -= amount
                granted[rt] = amount
        
        # Create grant
        grant = AllocationGrant(
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            granted=granted,
            node_id=node.node_id,
            preemptible=request.preemptible,
            quality_score=self.admission.compute_admission_score(request, node, tenant)
        )
        
        # Record allocation
        node.allocations[request.request_id] = grant
        tenant.allocations[request.request_id] = grant
        
        # Update load factor
        utils = [cap.utilization for cap in node.capacities.values()]
        node.load_factor = sum(utils) / len(utils) if utils else 0.0
        
        return grant
    
    # -------------------------------------------------------------------------
    # Resource Release
    # -------------------------------------------------------------------------
    
    def release(self, allocation_id: str) -> bool:
        """Release an allocation."""
        with self._lock:
            # Find the allocation
            for node_id, node in self.nodes.items():
                if allocation_id in node.allocations:
                    return self._release_allocation_internal(allocation_id, node_id)
            
            return False
    
    def _release_allocation_internal(
        self,
        allocation_id: str,
        node_id: str
    ) -> bool:
        """Internal release implementation."""
        node = self.nodes.get(node_id)
        if not node or allocation_id not in node.allocations:
            return False
        
        grant = node.allocations[allocation_id]
        
        # Return resources
        for rt, amount in grant.granted.items():
            if rt in node.capacities:
                node.capacities[rt].available += amount
        
        # Remove from node
        del node.allocations[allocation_id]
        
        # Remove from tenant
        tenant = self.tenants.get(grant.tenant_id)
        if tenant and allocation_id in tenant.allocations:
            del tenant.allocations[allocation_id]
        
        # Update load factor
        utils = [cap.utilization for cap in node.capacities.values()]
        node.load_factor = sum(utils) / len(utils) if utils else 0.0
        
        # Notify callbacks
        for callback in self._release_callbacks:
            try:
                callback(allocation_id)
            except Exception as e:
                logger.error(f"Release callback error: {e}")
        
        logger.debug(f"Released allocation {allocation_id}")
        return True
    
    # -------------------------------------------------------------------------
    # Fair Share Computation
    # -------------------------------------------------------------------------
    
    def compute_fair_shares(self) -> Dict[str, Dict[ResourceType, float]]:
        """Compute fair share allocations for all tenants."""
        with self._lock:
            # Gather demands
            demands = {}
            for tenant_id, tenant in self.tenants.items():
                demands[tenant_id] = {}
                for rt, quota in tenant.quotas.items():
                    demands[tenant_id][rt] = quota.limit
            
            # Gather total capacities
            capacities: Dict[ResourceType, float] = defaultdict(float)
            for node in self.nodes.values():
                for rt, cap in node.capacities.items():
                    capacities[rt] += cap.total
            
            return self.fair_share.compute_shares(demands, dict(capacities))
    
    # -------------------------------------------------------------------------
    # Rebalancing
    # -------------------------------------------------------------------------
    
    def check_and_rebalance(self) -> int:
        """Check if rebalancing is needed and perform it."""
        with self._lock:
            if not self.rebalancer.should_rebalance(self.nodes):
                return 0
            
            plan = self.rebalancer.compute_rebalance_plan(self.nodes, self.tenants)
            
            if not plan:
                return 0
            
            # Execute migrations
            successful = 0
            for alloc_id, source_id, target_id in plan:
                if self._migrate_allocation(alloc_id, source_id, target_id):
                    successful += 1
            
            self.rebalancer.last_rebalance = time.time() * 1000
            self.metrics.rebalances += successful
            
            logger.info(f"Rebalanced {successful}/{len(plan)} allocations")
            return successful
    
    def _migrate_allocation(
        self,
        allocation_id: str,
        source_id: str,
        target_id: str
    ) -> bool:
        """Migrate an allocation from source to target node."""
        source = self.nodes.get(source_id)
        target = self.nodes.get(target_id)
        
        if not source or not target:
            return False
        
        if allocation_id not in source.allocations:
            return False
        
        grant = source.allocations[allocation_id]
        
        # Check if target can accommodate
        for rt, amount in grant.granted.items():
            if rt in target.capacities:
                if target.capacities[rt].available < amount:
                    return False
        
        # Perform migration
        # Release from source
        for rt, amount in grant.granted.items():
            if rt in source.capacities:
                source.capacities[rt].available += amount
        
        del source.allocations[allocation_id]
        
        # Allocate on target
        for rt, amount in grant.granted.items():
            if rt in target.capacities:
                target.capacities[rt].available -= amount
        
        # Update grant
        new_grant = AllocationGrant(
            request_id=grant.request_id,
            tenant_id=grant.tenant_id,
            granted=grant.granted,
            node_id=target_id,
            preemptible=grant.preemptible,
            quality_score=grant.quality_score
        )
        
        target.allocations[allocation_id] = new_grant
        
        # Update tenant
        tenant = self.tenants.get(grant.tenant_id)
        if tenant:
            tenant.allocations[allocation_id] = new_grant
        
        return True
    
    # -------------------------------------------------------------------------
    # Metrics and Monitoring
    # -------------------------------------------------------------------------
    
    def get_metrics(self) -> AllocationMetrics:
        """Get current allocation metrics."""
        with self._lock:
            # Update utilization metrics
            for rt in ResourceType:
                total_cap = 0.0
                total_used = 0.0
                
                for node in self.nodes.values():
                    if rt in node.capacities:
                        cap = node.capacities[rt]
                        total_cap += cap.total
                        total_used += cap.used
                
                if total_cap > 0:
                    self.metrics.utilization_by_resource[rt] = total_used / total_cap
            
            # Compute fairness index
            self.metrics.fairness_index = self._compute_fairness_index()
            
            return self.metrics
    
    def _compute_fairness_index(self) -> float:
        """
        Compute Jain's Fairness Index.
        
        Index = (sum(x_i))^2 / (n * sum(x_i^2))
        
        Where x_i is the fraction of fair share received by tenant i.
        Values range from 1/n (unfair) to 1 (perfectly fair).
        """
        if not self.tenants:
            return 1.0
        
        # Compute fair shares
        fair_shares = self.compute_fair_shares()
        
        shares = []
        for tenant_id, tenant in self.tenants.items():
            if tenant_id not in fair_shares:
                continue
            
            fair = fair_shares[tenant_id]
            
            # Compute fraction of fair share received
            for rt in fair:
                fair_amount = fair.get(rt, 0.0)
                actual = tenant.get_total_allocated(rt)
                
                if fair_amount > 0:
                    fraction = min(actual / fair_amount, 1.0)
                    shares.append(fraction)
        
        if not shares:
            return 1.0
        
        n = len(shares)
        sum_x = sum(shares)
        sum_x2 = sum(x * x for x in shares)
        
        if sum_x2 == 0:
            return 1.0
        
        return (sum_x * sum_x) / (n * sum_x2)
    
    def get_cluster_utilization(self) -> Dict[ResourceType, float]:
        """Get cluster-wide utilization by resource type."""
        with self._lock:
            result: Dict[ResourceType, float] = {}
            
            for rt in ResourceType:
                total_cap = 0.0
                total_used = 0.0
                
                for node in self.nodes.values():
                    if rt in node.capacities:
                        cap = node.capacities[rt]
                        total_cap += cap.total
                        total_used += cap.used
                
                if total_cap > 0:
                    result[rt] = total_used / total_cap
            
            return result
    
    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    
    def on_allocation(
        self,
        callback: Callable[[AllocationGrant], None]
    ) -> None:
        """Register allocation callback."""
        self._allocation_callbacks.append(callback)
    
    def on_release(
        self,
        callback: Callable[[str], None]
    ) -> None:
        """Register release callback."""
        self._release_callbacks.append(callback)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_allocator(
    fair_share: str = "drf",
    overbooking: float = 1.2,
    rebalance_strategy: str = "gradual"
) -> ResourceAllocator:
    """
    Create a resource allocator with default configuration.
    
    Args:
        fair_share: Fair share algorithm ("drf", "max_min", "weighted")
        overbooking: Overbooking factor for admission
        rebalance_strategy: Rebalancing strategy
    
    Returns:
        Configured ResourceAllocator
    """
    # Select fair share algorithm
    if fair_share == "drf":
        algo = DominantResourceFairness()
    elif fair_share == "max_min":
        algo = MaxMinFairShare()
    elif fair_share == "weighted":
        algo = WeightedFairShare()
    else:
        algo = DominantResourceFairness()
    
    # Select rebalance strategy
    strategy_map = {
        "none": RebalanceStrategy.NONE,
        "gradual": RebalanceStrategy.GRADUAL,
        "aggressive": RebalanceStrategy.AGGRESSIVE,
        "opportunistic": RebalanceStrategy.OPPORTUNISTIC
    }
    strategy = strategy_map.get(rebalance_strategy, RebalanceStrategy.GRADUAL)
    
    return ResourceAllocator(
        fair_share_algo=algo,
        admission_controller=AdmissionController(overbooking_factor=overbooking),
        rebalancer=ResourceRebalancer(strategy=strategy)
    )


def create_request(
    request_id: str,
    tenant_id: str,
    gpu_memory: float = 0.0,
    gpu_compute: float = 0.0,
    priority: str = "normal",
    preemptible: bool = True
) -> ResourceRequest:
    """
    Create a resource request.
    
    Args:
        request_id: Unique request identifier
        tenant_id: Tenant making the request
        gpu_memory: GPU memory in bytes
        gpu_compute: GPU compute fraction (0-1)
        priority: Priority class name
        preemptible: Whether request can be preempted
    
    Returns:
        ResourceRequest
    """
    resources = {}
    
    if gpu_memory > 0:
        resources[ResourceType.GPU_MEMORY] = gpu_memory
    if gpu_compute > 0:
        resources[ResourceType.GPU_COMPUTE] = gpu_compute
    
    priority_map = {
        "system": PriorityClass.SYSTEM,
        "critical": PriorityClass.CRITICAL,
        "high": PriorityClass.HIGH,
        "normal": PriorityClass.NORMAL,
        "low": PriorityClass.LOW,
        "scavenger": PriorityClass.SCAVENGER
    }
    
    return ResourceRequest(
        request_id=request_id,
        tenant_id=tenant_id,
        resources=resources,
        priority=priority_map.get(priority, PriorityClass.NORMAL),
        preemptible=preemptible
    )
