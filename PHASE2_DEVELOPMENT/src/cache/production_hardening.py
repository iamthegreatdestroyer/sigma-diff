"""
Production Hardening Module
===========================

Error handling, graceful degradation, health checks, and resilience
for production cache deployments.

Key Features:
- Circuit breaker pattern for failing dependencies
- Graceful degradation with fallback strategies
- Health checks and readiness probes
- Structured error handling and logging
- Metrics collection for observability
- Rate limiting and backpressure

Sprint 2.2: Days 5-6 Delivery
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from collections import deque
import traceback

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5         # Failures before opening
    success_threshold: int = 2         # Successes to close from half-open
    timeout_seconds: float = 30.0      # Time before trying half-open
    half_open_max_calls: int = 3       # Calls allowed in half-open


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.
    
    Usage:
        breaker = CircuitBreaker("cache-service")
        
        @breaker.protect
        def call_cache():
            return cache.get(key)
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()
        
    @property
    def state(self) -> CircuitState:
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.config.timeout_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN")
            
            return self._state
    
    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        state = self.state
        
        if state == CircuitState.CLOSED:
            return True
        
        if state == CircuitState.OPEN:
            return False
        
        # HALF_OPEN: Allow limited requests
        with self._lock:
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
    
    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED")
            else:
                self._failure_count = max(0, self._failure_count - 1)
    
    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN")
            elif (self._state == CircuitState.CLOSED and 
                  self._failure_count >= self.config.failure_threshold):
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name}: CLOSED -> OPEN")
    
    def protect(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to protect a function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not self.allow_request():
                raise CircuitOpenError(f"Circuit {self.name} is open")
            
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise
        
        return wrapper


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class GracefulDegradation:
    """
    Manages graceful degradation strategies.
    
    Provides fallback behaviors when primary operations fail.
    """
    
    def __init__(self):
        self._fallbacks: Dict[str, Callable] = {}
        self._degradation_level = 0  # 0 = normal, higher = more degraded
        self._lock = threading.Lock()
        
    def register_fallback(self, name: str, fallback_fn: Callable) -> None:
        """Register a fallback function for an operation."""
        self._fallbacks[name] = fallback_fn
    
    def execute_with_fallback(
        self,
        name: str,
        primary_fn: Callable[..., T],
        *args,
        **kwargs
    ) -> Tuple[T, bool]:
        """Execute with fallback on failure.
        
        Returns:
            Tuple of (result, used_fallback)
        """
        try:
            result = primary_fn(*args, **kwargs)
            return result, False
        except Exception as e:
            logger.warning(f"Primary operation {name} failed: {e}")
            
            if name in self._fallbacks:
                try:
                    fallback_result = self._fallbacks[name](*args, **kwargs)
                    return fallback_result, True
                except Exception as fallback_error:
                    logger.error(f"Fallback for {name} also failed: {fallback_error}")
                    raise
            raise
    
    def set_degradation_level(self, level: int) -> None:
        """Set the current degradation level."""
        with self._lock:
            old_level = self._degradation_level
            self._degradation_level = level
            if level != old_level:
                logger.info(f"Degradation level changed: {old_level} -> {level}")
    
    def get_degradation_level(self) -> int:
        """Get the current degradation level."""
        with self._lock:
            return self._degradation_level


class HealthChecker:
    """
    Health checking system for cache components.
    
    Provides liveness, readiness, and custom health checks.
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._lock = threading.Lock()
        
    def register_check(
        self,
        name: str,
        check_fn: Callable[[], HealthCheckResult]
    ) -> None:
        """Register a health check."""
        self._checks[name] = check_fn
    
    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self._checks:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Unknown health check: {name}"
            )
        
        start_time = time.time()
        try:
            result = self._checks[name]()
            result.latency_ms = (time.time() - start_time) * 1000
        except Exception as e:
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )
        
        with self._lock:
            self._last_results[name] = result
        
        return result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        for name in self._checks:
            results[name] = self.run_check(name)
        return results
    
    def get_overall_health(self) -> HealthCheckResult:
        """Get overall health status."""
        results = self.run_all_checks()
        
        if not results:
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="No health checks registered"
            )
        
        unhealthy = [n for n, r in results.items() if r.status == HealthStatus.UNHEALTHY]
        degraded = [n for n, r in results.items() if r.status == HealthStatus.DEGRADED]
        
        if unhealthy:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Unhealthy checks: {', '.join(unhealthy)}",
                details={"unhealthy": unhealthy, "degraded": degraded}
            )
        
        if degraded:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                message=f"Degraded checks: {', '.join(degraded)}",
                details={"degraded": degraded}
            )
        
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="All checks passing"
        )
    
    def is_ready(self) -> bool:
        """Check if service is ready to accept traffic."""
        health = self.get_overall_health()
        return health.status != HealthStatus.UNHEALTHY
    
    def is_alive(self) -> bool:
        """Check if service is alive (for liveness probes)."""
        # Basic liveness - can we run health checks at all?
        try:
            self.get_overall_health()
            return True
        except Exception:
            return False


class MetricsCollector:
    """Simple metrics collection for observability."""
    
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, deque] = {}
        self._lock = threading.Lock()
    
    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        with self._lock:
            self._gauges[name] = value
    
    def observe(self, name: str, value: float) -> None:
        """Observe a histogram value."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = deque(maxlen=1000)
            self._histograms[name].append(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            histogram_stats = {}
            for name, values in self._histograms.items():
                if values:
                    sorted_vals = sorted(values)
                    histogram_stats[name] = {
                        "count": len(sorted_vals),
                        "min": sorted_vals[0],
                        "max": sorted_vals[-1],
                        "avg": sum(sorted_vals) / len(sorted_vals),
                        "p50": sorted_vals[len(sorted_vals) // 2],
                        "p99": sorted_vals[int(len(sorted_vals) * 0.99)]
                    }
            
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": histogram_stats
            }


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(
        self,
        rate: float,          # Tokens per second
        burst: int = 10       # Maximum burst size
    ):
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens. Returns True if successful."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._last_update = now
            
            # Add tokens based on elapsed time
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False
    
    def wait_for_token(self, timeout: float = 5.0) -> bool:
        """Wait for a token to become available."""
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            if self.acquire():
                return True
            time.sleep(0.01)
        
        return False


# Production-ready cache wrapper
class ProductionCacheWrapper:
    """
    Wraps a cache with production hardening features.
    """
    
    def __init__(self, cache: Any, name: str = "cache"):
        self.cache = cache
        self.name = name
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(name)
        
        # Health checker
        self.health_checker = HealthChecker()
        self.health_checker.register_check("cache_responsive", self._check_cache_responsive)
        
        # Metrics
        self.metrics = MetricsCollector()
        
        # Rate limiter
        self.rate_limiter = RateLimiter(rate=1000, burst=100)
        
        # Graceful degradation
        self.degradation = GracefulDegradation()
        
    def _check_cache_responsive(self) -> HealthCheckResult:
        """Check if cache is responsive."""
        try:
            # Try a simple operation
            start = time.time()
            if hasattr(self.cache, 'get_stats'):
                self.cache.get_stats()
            latency = (time.time() - start) * 1000
            
            if latency < 100:
                return HealthCheckResult(HealthStatus.HEALTHY, "Cache responsive", {"latency_ms": latency})
            elif latency < 500:
                return HealthCheckResult(HealthStatus.DEGRADED, "Cache slow", {"latency_ms": latency})
            else:
                return HealthCheckResult(HealthStatus.UNHEALTHY, "Cache too slow", {"latency_ms": latency})
        except Exception as e:
            return HealthCheckResult(HealthStatus.UNHEALTHY, f"Cache error: {e}")
    
    def get(self, key: str) -> Any:
        """Get from cache with production hardening."""
        if not self.rate_limiter.acquire():
            self.metrics.increment("rate_limited")
            raise RateLimitExceeded("Rate limit exceeded")
        
        start_time = time.time()
        
        try:
            result = self.circuit_breaker.protect(lambda: self.cache.get(key))()
            
            latency = (time.time() - start_time) * 1000
            self.metrics.observe("get_latency_ms", latency)
            
            if result is not None:
                self.metrics.increment("cache_hits")
            else:
                self.metrics.increment("cache_misses")
            
            return result
            
        except CircuitOpenError:
            self.metrics.increment("circuit_open_rejects")
            raise
        except Exception as e:
            self.metrics.increment("cache_errors")
            raise
    
    def set(self, key: str, value: Any) -> bool:
        """Set in cache with production hardening."""
        if not self.rate_limiter.acquire():
            self.metrics.increment("rate_limited")
            raise RateLimitExceeded("Rate limit exceeded")
        
        start_time = time.time()
        
        try:
            result = self.circuit_breaker.protect(lambda: self.cache.set(key, value))()
            
            latency = (time.time() - start_time) * 1000
            self.metrics.observe("set_latency_ms", latency)
            self.metrics.increment("cache_sets")
            
            return result
            
        except CircuitOpenError:
            self.metrics.increment("circuit_open_rejects")
            raise
        except Exception as e:
            self.metrics.increment("cache_errors")
            raise
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        health = self.health_checker.get_overall_health()
        return {
            "status": health.status.value,
            "message": health.message,
            "details": health.details,
            "circuit_state": self.circuit_breaker.state.value,
            "degradation_level": self.degradation.get_degradation_level()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics."""
        return self.metrics.get_metrics()


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


# Convenience factory
def harden_cache(cache: Any, name: str = "cache") -> ProductionCacheWrapper:
    """Wrap a cache with production hardening features."""
    return ProductionCacheWrapper(cache, name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Production Hardening...")
    
    # Test circuit breaker
    breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
    
    @breaker.protect
    def failing_operation():
        raise RuntimeError("Simulated failure")
    
    # Trigger circuit open
    for i in range(5):
        try:
            failing_operation()
        except (RuntimeError, CircuitOpenError) as e:
            print(f"Attempt {i+1}: {type(e).__name__}")
    
    print(f"Circuit state: {breaker.state.value}")
    
    # Test health checker
    checker = HealthChecker()
    checker.register_check("test", lambda: HealthCheckResult(HealthStatus.HEALTHY, "OK"))
    
    health = checker.get_overall_health()
    print(f"\nHealth: {health.status.value} - {health.message}")
    
    # Test metrics
    metrics = MetricsCollector()
    metrics.increment("requests", 100)
    metrics.set_gauge("memory_mb", 512.5)
    for i in range(50):
        metrics.observe("latency_ms", i * 2)
    
    print(f"\nMetrics: {metrics.get_metrics()}")
