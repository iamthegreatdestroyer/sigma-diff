"""
KV cache and memory management components.

Sprint 2.2 Days 5-9 Implementation:
- KV Cache Compression (INT8/INT4 quantization)
- Adaptive Cache Sizing (dynamic memory management)
- Distributed Cache Optimization (cross-node coordination)
- Production Hardening (circuit breaker, health checks, metrics)
"""

from .manager import (
    PagedAttentionKVCache,
    PrefixCache,
    GPUMemoryPool,
    PageConfig,
    CacheMetadata,
    EvictionPolicy,
    create_kv_cache_manager,
)

# Sprint 2.2 Day 5: KV Cache Compression
from .kv_cache_compression import (
    QuantizationType,
    ScalingMode,
    QuantizationConfig,
    Int8Quantizer,
    Int4Quantizer,
    MixedPrecisionQuantizer,
    QuantizedKVCacheManager,
    create_quantized_kv_cache,
)

# Sprint 2.2 Day 6: Adaptive Cache Sizing
from .adaptive_cache_manager import (
    MemoryPressureLevel,
    CacheTier,
    MemoryStats,
    WorkloadStats,
    SizingDecision,
    AdaptiveSizingConfig,
    MemoryMonitor,
    WorkloadAnalyzer,
    AdaptiveCacheSizer,
    create_adaptive_sizer,
)

# Sprint 2.2 Day 7: Distributed Cache Optimization
from .distributed_cache_optimizer import (
    NodeState,
    MigrationState,
    NodeInfo,
    CacheEntry,
    DistributedCacheConfig,
    ConsistentHash,
    CacheCoordinator,
    create_distributed_cache_optimizer,
)

# Sprint 2.2 Days 8-9: Production Hardening
from .production_hardening import (
    CircuitState,
    HealthStatus,
    CircuitBreakerConfig,
    HealthCheckResult,
    CircuitBreaker,
    CircuitOpenError,
    GracefulDegradation,
    HealthChecker,
    MetricsCollector,
    RateLimiter,
    RateLimitExceeded,
    ProductionCacheWrapper,
    harden_cache,
)

__all__ = [
    # Core cache components
    "PagedAttentionKVCache",
    "PrefixCache",
    "GPUMemoryPool",
    "PageConfig",
    "CacheMetadata",
    "EvictionPolicy",
    "create_kv_cache_manager",
    # Day 5: Compression
    "QuantizationType",
    "ScalingMode",
    "QuantizationConfig",
    "Int8Quantizer",
    "Int4Quantizer",
    "MixedPrecisionQuantizer",
    "QuantizedKVCacheManager",
    "create_quantized_kv_cache",
    # Day 6: Adaptive Sizing
    "MemoryPressureLevel",
    "CacheTier",
    "MemoryStats",
    "WorkloadStats",
    "SizingDecision",
    "AdaptiveSizingConfig",
    "MemoryMonitor",
    "WorkloadAnalyzer",
    "AdaptiveCacheSizer",
    "create_adaptive_sizer",
    # Day 7: Distributed Optimization
    "NodeState",
    "MigrationState",
    "NodeInfo",
    "CacheEntry",
    "DistributedCacheConfig",
    "ConsistentHash",
    "CacheCoordinator",
    "create_distributed_cache_optimizer",
    # Days 8-9: Production Hardening
    "CircuitState",
    "HealthStatus",
    "CircuitBreakerConfig",
    "HealthCheckResult",
    "CircuitBreaker",
    "CircuitOpenError",
    "GracefulDegradation",
    "HealthChecker",
    "MetricsCollector",
    "RateLimiter",
    "RateLimitExceeded",
    "ProductionCacheWrapper",
    "harden_cache",
]
