"""
Distributed inference engine components.

This module provides comprehensive multi-GPU optimization capabilities:
- Core distributed inference engine with parallelism strategies
- Tensor parallelism (Megatron-style column/row parallel)
- Pipeline parallelism (GPipe/1F1B scheduling)
- Multi-GPU distributed KV cache with coherency
- GPU coordination and workload scheduling
"""

from .engine import (
    DistributedInferenceEngine,
    DistributedConfig,
    ParallelismStrategy,
    TensorShardManager,
    CollectiveCommunicator,
    GPUMemoryManager,
    DistributedStats,
    create_distributed_engine,
)

from .tensor_parallelism import (
    TensorParallelMode,
    TensorParallelConfig,
    ProcessGroupManager,
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelAttention,
    ParallelMLP,
    ParallelTransformerLayer,
    TensorParallelModel,
    create_tensor_parallel_model,
)

from .pipeline_parallelism import (
    PipelineSchedule,
    ActivationPolicy,
    PipelineConfig,
    PipelineCommunicator,
    ActivationCheckpointer,
    GPipeScheduler,
    OneFOneBScheduler,
    InterleavedScheduler,
    PipelineParallelEngine,
    create_pipeline_engine,
)

from .multi_gpu_cache import (
    CacheCoherencyProtocol,
    MESIState,
    MESIProtocol,
    DirectoryProtocol,
    GPUPageAllocator,
    PageTransferManager,
    DistributedKVCache,
    SequenceAwareCacheSharding,
    create_distributed_cache,
)

from .gpu_coordinator import (
    GPUState,
    WorkloadPriority,
    SchedulingPolicy,
    TopologyType,
    FailureType,
    RecoveryAction,
    HealthMonitorConfig,
    SchedulerConfig,
    CoordinatorConfig,
    GPUInfo,
    TopologyInfo,
    WorkloadTask,
    CoordinatorStatistics,
    TopologyDetector,
    HealthMonitor,
    WorkloadScheduler,
    GPUCoordinator,
    create_coordinator,
    get_gpu_cluster_info,
)

__all__ = [
    # Core distributed engine
    "DistributedInferenceEngine",
    "DistributedConfig",
    "ParallelismStrategy",
    "TensorShardManager",
    "CollectiveCommunicator",
    "GPUMemoryManager",
    "DistributedStats",
    "create_distributed_engine",
    # Tensor parallelism
    "TensorParallelMode",
    "TensorParallelConfig",
    "ProcessGroupManager",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "ParallelAttention",
    "ParallelMLP",
    "ParallelTransformerLayer",
    "TensorParallelModel",
    "create_tensor_parallel_model",
    # Pipeline parallelism
    "PipelineSchedule",
    "ActivationPolicy",
    "PipelineConfig",
    "PipelineCommunicator",
    "ActivationCheckpointer",
    "GPipeScheduler",
    "OneFOneBScheduler",
    "InterleavedScheduler",
    "PipelineParallelEngine",
    "create_pipeline_engine",
    # Multi-GPU cache
    "CacheCoherencyProtocol",
    "MESIState",
    "MESIProtocol",
    "DirectoryProtocol",
    "GPUPageAllocator",
    "PageTransferManager",
    "DistributedKVCache",
    "SequenceAwareCacheSharding",
    "create_distributed_cache",
    # GPU coordinator
    "GPUState",
    "WorkloadPriority",
    "SchedulingPolicy",
    "TopologyType",
    "FailureType",
    "RecoveryAction",
    "HealthMonitorConfig",
    "SchedulerConfig",
    "CoordinatorConfig",
    "GPUInfo",
    "TopologyInfo",
    "WorkloadTask",
    "CoordinatorStatistics",
    "TopologyDetector",
    "HealthMonitor",
    "WorkloadScheduler",
    "GPUCoordinator",
    "create_coordinator",
    "get_gpu_cluster_info",
]
