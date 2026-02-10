"""
Inference module for RYZEN-LLM Phase 2.

This module provides advanced inference capabilities including:
- Speculative decoding for accelerated text generation
- Multimodal processing
- Inference pipelines
- Batch processing engine (Sprint 4.1)
  - Batch optimization with adaptive sizing
  - Request scheduling with multiple policies
  - Request queuing with admission control
"""

from .speculative_decoder import (
    SpeculativeConfig,
    SpeculativeStats,
    DraftModel,
    SimpleDraftModel,
    Verifier,
    TargetModelVerifier,
    SpeculativeDecoder,
    create_speculative_decoder,
    benchmark_speculative_decoding,
    performance_monitor,
)

# Sprint 4.1: Batch Processing Engine
from .batch_optimizer import (
    OptimizationStrategy,
    OptimizerConfig,
    BatchMetrics,
    BatchSizePredictor,
    MemoryEstimator,
    BatchOptimizer,
    AdaptiveBatchOptimizer,
    create_batch_optimizer,
)

from .batch_scheduler import (
    SchedulingPolicy,
    TriggerType,
    SchedulerConfig,
    ScheduledRequest,
    SizeThresholdTrigger,
    TimeDeadlineTrigger,
    PriorityUrgentTrigger,
    MemoryPressureTrigger,
    LoadSheddingTrigger,
    BatchFormationPolicy,
    FIFOPolicy,
    SizeOptimalPolicy,
    DeadlineDrivenPolicy,
    PriorityWeightedPolicy,
    AdaptivePolicy,
    BatchScheduler,
    SchedulerContext,
    create_scheduler,
    create_latency_optimized_scheduler,
    create_throughput_optimized_scheduler,
)

from .request_queue import (
    QueuePriority,
    AdmissionDecision,
    BackpressureLevel,
    QueueConfig,
    QueuedRequest,
    TokenBucketController,
    LoadBasedController,
    CompositeAdmissionController,
    FairScheduler,
    RequestQueue,
    QueueMonitor,
    create_request_queue,
    create_high_throughput_queue,
    create_low_latency_queue,
)

__all__ = [
    # Speculative Decoding
    "SpeculativeConfig",
    "SpeculativeStats",
    "DraftModel",
    "SimpleDraftModel",
    "Verifier",
    "TargetModelVerifier",
    "SpeculativeDecoder",
    "create_speculative_decoder",
    "benchmark_speculative_decoding",
    "performance_monitor",
    
    # Batch Optimizer (Sprint 4.1)
    "OptimizationStrategy",
    "OptimizerConfig",
    "BatchMetrics",
    "BatchSizePredictor",
    "MemoryEstimator",
    "BatchOptimizer",
    "AdaptiveBatchOptimizer",
    "create_batch_optimizer",
    
    # Batch Scheduler (Sprint 4.1)
    "SchedulingPolicy",
    "TriggerType",
    "SchedulerConfig",
    "ScheduledRequest",
    "SizeThresholdTrigger",
    "TimeDeadlineTrigger",
    "PriorityUrgentTrigger",
    "MemoryPressureTrigger",
    "LoadSheddingTrigger",
    "BatchFormationPolicy",
    "FIFOPolicy",
    "SizeOptimalPolicy",
    "DeadlineDrivenPolicy",
    "PriorityWeightedPolicy",
    "AdaptivePolicy",
    "BatchScheduler",
    "SchedulerContext",
    "create_scheduler",
    "create_latency_optimized_scheduler",
    "create_throughput_optimized_scheduler",
    
    # Request Queue (Sprint 4.1)
    "QueuePriority",
    "AdmissionDecision",
    "BackpressureLevel",
    "QueueConfig",
    "QueuedRequest",
    "TokenBucketController",
    "LoadBasedController",
    "CompositeAdmissionController",
    "FairScheduler",
    "RequestQueue",
    "QueueMonitor",
    "create_request_queue",
    "create_high_throughput_queue",
    "create_low_latency_queue",
]
