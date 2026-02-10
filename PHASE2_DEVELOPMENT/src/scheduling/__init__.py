"""
Advanced Scheduling & Resource Management Module.

Sprint 4.3 Deliverable: Sophisticated scheduling infrastructure for high-performance
LLM serving with multi-tenant isolation, GPU memory optimization, and adaptive
workload management.

Architecture Overview:
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCHEDULING & RESOURCE MANAGEMENT                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  GPU Memory     │    │   Adaptive      │    │   Resource      │         │
│  │   Manager       │◄──►│   Batch         │◄──►│   Allocator     │         │
│  │                 │    │   Scheduler     │    │                 │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Unified Metrics & Telemetry                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Cross-Domain Synthesis:                                                     │
│  ├─ Operating Systems: CFS, EDF, Rate Monotonic                             │
│  ├─ Databases: Query scheduling, resource pools                             │
│  ├─ Network QoS: Traffic shaping, fair queuing                              │
│  ├─ Cloud: Kubernetes scheduling, Borg concepts                             │
│  └─ Economics: Auction theory, market-based allocation                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Key Features:
- GPU Memory Management: Pooling, defragmentation, pressure detection
- Adaptive Scheduling: ML-based policy selection, workload prediction
- Multi-Tenant Isolation: Fair sharing, quota enforcement, SLA guarantees
- Dynamic Rebalancing: Real-time resource redistribution

Success Criteria:
- GPU memory utilization > 85%
- Scheduling overhead < 2%
- Fair allocation across tenants
- Dynamic resource rebalancing

Copyright (c) 2025. All Rights Reserved.
"""

from .gpu_memory_manager import (
    GPUMemoryManager,
    MemoryPool,
    AllocationPolicy,
    MemoryBlock,
    MemoryStats,
    MemoryPressureLevel,
    DefragmentationStrategy,
)
from .batch_scheduler import (
    AdaptiveBatchScheduler,
    SchedulingPolicy,
    WorkloadProfile,
    SchedulingDecision,
    BatchFormationStrategy,
    LatencyPredictor,
)
from .resource_allocator import (
    ResourceAllocator,
    TenantQuota,
    AllocationRequest,
    AllocationResult,
    FairnessPolicy,
    RebalancingStrategy,
    IsolationLevel,
)

__all__ = [
    # GPU Memory Management
    "GPUMemoryManager",
    "MemoryPool",
    "AllocationPolicy",
    "MemoryBlock",
    "MemoryStats",
    "MemoryPressureLevel",
    "DefragmentationStrategy",
    # Adaptive Batch Scheduling
    "AdaptiveBatchScheduler",
    "SchedulingPolicy",
    "WorkloadProfile",
    "SchedulingDecision",
    "BatchFormationStrategy",
    "LatencyPredictor",
    # Resource Allocation
    "ResourceAllocator",
    "TenantQuota",
    "AllocationRequest",
    "AllocationResult",
    "FairnessPolicy",
    "RebalancingStrategy",
    "IsolationLevel",
]

__version__ = "4.3.0"
__sprint__ = "Sprint 4.3: Advanced Scheduling & Resource Management"
