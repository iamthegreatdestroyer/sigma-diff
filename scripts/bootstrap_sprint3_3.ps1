# Autonomous Phase 3 Completion - Sprint 3.3 Bootstrap
# Created: January 6, 2026
# Purpose: Enable autonomous execution of Sprint 3.3: Resilience & Fault Tolerance

$ErrorActionPreference = "Stop"
$Phase2Dev = "S:\Ryot\PHASE2_DEVELOPMENT"

Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       RYZANSTEIN LLM - AUTONOMOUS PHASE 3 COMPLETION         ║" -ForegroundColor Cyan
Write-Host "║              Sprint 3.3: Resilience & Fault Tolerance        ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

# =============================================================================
# CREATE RESILIENCE MODULE STRUCTURE
# =============================================================================

$ResilienceDir = "$Phase2Dev\src\resilience"
Write-Host "`n📁 Creating resilience module structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $ResilienceDir | Out-Null
Write-Host "  ✅ Created: $ResilienceDir" -ForegroundColor Green

# =============================================================================
# 1. __init__.py - Module Exports
# =============================================================================

$InitPy = @"
"""
Resilience & Fault Tolerance Module for Ryzanstein LLM.

This module provides patterns for building resilient distributed systems:
- CircuitBreaker: Prevent cascade failures
- RetryPolicy: Exponential backoff with jitter
- FallbackHandler: Degraded operation modes
- Bulkhead: Resource isolation
- HealthChecker: Liveness and readiness probes

Sprint 3.3 Implementation
Created: January 6, 2026
"""

from .circuit_breaker import CircuitBreaker, CircuitState
from .retry_policy import RetryPolicy, RetryConfig
from .fallback import FallbackHandler, FallbackStrategy
from .bulkhead import Bulkhead, BulkheadConfig
from .health_check import HealthChecker, HealthStatus

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    # Retry Policy
    "RetryPolicy",
    "RetryConfig",
    # Fallback
    "FallbackHandler",
    "FallbackStrategy",
    # Bulkhead
    "Bulkhead",
    "BulkheadConfig",
    # Health Check
    "HealthChecker",
    "HealthStatus",
]

__version__ = "1.0.0"
"@

$InitPy | Out-File -FilePath "$ResilienceDir\__init__.py" -Encoding UTF8
Write-Host "  ✅ Created: resilience/__init__.py" -ForegroundColor Green

# =============================================================================
# 2. Circuit Breaker Implementation
# =============================================================================

$CircuitBreaker = @"
"""
Circuit Breaker Pattern Implementation.

Prevents cascade failures by monitoring error rates and temporarily
stopping requests to failing services.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failures detected, requests rejected immediately
- HALF_OPEN: Testing if service recovered

Usage:
    breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=30.0,
        half_open_max_calls=3
    )
    
    async with breaker:
        result = await call_external_service()
"""

import asyncio
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    
    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes in half-open to close
    recovery_timeout: float = 30.0  # Seconds before attempting recovery
    half_open_max_calls: int = 3  # Max concurrent calls in half-open
    failure_rate_threshold: float = 0.5  # Failure rate to trigger open
    window_size: int = 10  # Sliding window for failure tracking


class CircuitBreaker:
    """
    Circuit Breaker implementation with sliding window.
    
    Attributes:
        name: Identifier for this circuit breaker
        state: Current state (CLOSED, OPEN, HALF_OPEN)
        config: Configuration parameters
        stats: Runtime statistics
    """
    
    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 3,
        failure_rate_threshold: float = 0.5,
        window_size: int = 10,
    ):
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
            failure_rate_threshold=failure_rate_threshold,
            window_size=window_size,
        )
        
        self._state = CircuitState.CLOSED
        self._lock = asyncio.Lock()
        self._last_failure_time: Optional[float] = None
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._results_window: deque = deque(maxlen=window_size)
        
        self.stats = CircuitBreakerStats()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN
    
    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            self.stats.state_changes += 1
            logger.info(
                f"Circuit breaker '{self.name}' transitioned: "
                f"{old_state.value} -> {new_state.value}"
            )
    
    async def _check_recovery(self) -> None:
        """Check if recovery timeout has elapsed."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.recovery_timeout:
                    await self._transition_to(CircuitState.HALF_OPEN)
                    self._half_open_calls = 0
                    self._success_count = 0
    
    async def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        async with self._lock:
            await self._check_recovery()
            
            if self._state == CircuitState.CLOSED:
                return True
            
            elif self._state == CircuitState.OPEN:
                self.stats.rejected_requests += 1
                return False
            
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        
        return False
    
    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self.stats.successful_requests += 1
            self.stats.last_success_time = time.time()
            self._results_window.append(True)
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
                    self._failure_count = 0
    
    async def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        async with self._lock:
            self.stats.failed_requests += 1
            self.stats.last_failure_time = time.time()
            self._last_failure_time = time.time()
            self._results_window.append(False)
            self._failure_count += 1
            
            if self._state == CircuitState.HALF_OPEN:
                await self._transition_to(CircuitState.OPEN)
            
            elif self._state == CircuitState.CLOSED:
                # Check failure threshold
                if self._failure_count >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)
                
                # Check failure rate
                elif len(self._results_window) >= self.config.window_size:
                    failures = sum(1 for r in self._results_window if not r)
                    rate = failures / len(self._results_window)
                    if rate >= self.config.failure_rate_threshold:
                        await self._transition_to(CircuitState.OPEN)
    
    async def execute(
        self,
        func: Callable[..., Any],
        *args,
        fallback: Optional[Callable[..., Any]] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            fallback: Optional fallback function if circuit is open
            **kwargs: Keyword arguments
        
        Returns:
            Result of func or fallback
        
        Raises:
            CircuitOpenError: If circuit is open and no fallback
        """
        self.stats.total_requests += 1
        
        if not await self._can_execute():
            if fallback is not None:
                return await fallback(*args, **kwargs)
            raise CircuitOpenError(
                f"Circuit breaker '{self.name}' is open. "
                f"Retry after {self.config.recovery_timeout}s"
            )
        
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        
        except Exception as e:
            await self._record_failure(e)
            raise
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not await self._can_execute():
            raise CircuitOpenError(f"Circuit breaker '{self.name}' is open")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is None:
            await self._record_success()
        else:
            await self._record_failure(exc_val)
        return False
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "rejected_requests": self.stats.rejected_requests,
            "failure_rate": self.stats.failure_rate,
            "state_changes": self.stats.state_changes,
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass
"@

$CircuitBreaker | Out-File -FilePath "$ResilienceDir\circuit_breaker.py" -Encoding UTF8
Write-Host "  ✅ Created: resilience/circuit_breaker.py" -ForegroundColor Green

# =============================================================================
# 3. Retry Policy Implementation
# =============================================================================

$RetryPolicy = @"
"""
Retry Policy with Exponential Backoff and Jitter.

Implements intelligent retry logic for transient failures with:
- Exponential backoff
- Jitter to prevent thundering herd
- Maximum retry limits
- Configurable retry conditions

Usage:
    retry = RetryPolicy(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        exponential_base=2,
        jitter=True
    )
    
    @retry
    async def call_service():
        return await http.get("/api/data")
"""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Optional, Callable, Any, Set, Type
from functools import wraps
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry policy."""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1  # 10% jitter
    retryable_exceptions: Set[Type[Exception]] = None
    
    def __post_init__(self):
        if self.retryable_exceptions is None:
            self.retryable_exceptions = {
                ConnectionError,
                TimeoutError,
                IOError,
            }


@dataclass
class RetryStats:
    """Statistics for retry monitoring."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    retries_performed: int = 0
    total_delay: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts


class RetryPolicy:
    """
    Retry policy with exponential backoff and jitter.
    
    Attributes:
        config: Retry configuration
        stats: Runtime statistics
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        jitter_factor: float = 0.1,
        retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    ):
        self.config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            jitter_factor=jitter_factor,
            retryable_exceptions=retryable_exceptions,
        )
        self.stats = RetryStats()
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt."""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            jitter = delay * self.config.jitter_factor * random.random()
            delay = delay + jitter
        
        return delay
    
    def _is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        return any(
            isinstance(exception, exc_type)
            for exc_type in self.config.retryable_exceptions
        )
    
    async def execute(
        self,
        func: Callable[..., Any],
        *args,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            on_retry: Optional callback on each retry
            **kwargs: Keyword arguments
        
        Returns:
            Result of func
        
        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            self.stats.total_attempts += 1
            
            try:
                result = await func(*args, **kwargs)
                self.stats.successful_attempts += 1
                return result
            
            except Exception as e:
                last_exception = e
                self.stats.failed_attempts += 1
                
                # Check if we should retry
                if attempt >= self.config.max_retries:
                    logger.error(f"Max retries ({self.config.max_retries}) exceeded")
                    raise
                
                if not self._is_retryable(e):
                    logger.error(f"Non-retryable exception: {type(e).__name__}")
                    raise
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                self.stats.retries_performed += 1
                self.stats.total_delay += delay
                
                logger.warning(
                    f"Retry {attempt + 1}/{self.config.max_retries} "
                    f"after {delay:.2f}s due to: {e}"
                )
                
                if on_retry:
                    on_retry(attempt + 1, e)
                
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)
        return wrapper
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "total_attempts": self.stats.total_attempts,
            "successful_attempts": self.stats.successful_attempts,
            "failed_attempts": self.stats.failed_attempts,
            "retries_performed": self.stats.retries_performed,
            "total_delay": self.stats.total_delay,
            "success_rate": self.stats.success_rate,
        }
"@

$RetryPolicy | Out-File -FilePath "$ResilienceDir\retry_policy.py" -Encoding UTF8
Write-Host "  ✅ Created: resilience/retry_policy.py" -ForegroundColor Green

# =============================================================================
# 4. Fallback Handler Implementation
# =============================================================================

$Fallback = @"
"""
Fallback Handler for Degraded Operation Modes.

Provides graceful degradation when primary services fail:
- Static fallback responses
- Cached fallback data
- Alternative service routing
- Custom fallback strategies

Usage:
    fallback = FallbackHandler()
    
    @fallback.with_fallback(default_response)
    async def get_recommendations():
        return await recommendation_service.get()
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict
from enum import Enum
from functools import wraps
import logging
import time

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Fallback strategies."""
    STATIC = "static"  # Return static value
    CACHE = "cache"  # Return cached value
    ALTERNATIVE = "alternative"  # Call alternative function
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Reduced functionality


@dataclass
class FallbackConfig:
    """Configuration for fallback handler."""
    strategy: FallbackStrategy = FallbackStrategy.STATIC
    cache_ttl: float = 300.0  # 5 minutes
    log_fallback: bool = True


@dataclass
class FallbackStats:
    """Statistics for fallback monitoring."""
    primary_successes: int = 0
    primary_failures: int = 0
    fallback_activations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class FallbackHandler:
    """
    Fallback handler for graceful degradation.
    
    Attributes:
        config: Fallback configuration
        stats: Runtime statistics
    """
    
    def __init__(
        self,
        strategy: FallbackStrategy = FallbackStrategy.STATIC,
        cache_ttl: float = 300.0,
        log_fallback: bool = True,
    ):
        self.config = FallbackConfig(
            strategy=strategy,
            cache_ttl=cache_ttl,
            log_fallback=log_fallback,
        )
        self.stats = FallbackStats()
        self._cache: Dict[str, tuple] = {}  # key -> (value, timestamp)
    
    def _get_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function call."""
        return f"{func.__name__}:{hash(args)}:{hash(frozenset(kwargs.items()))}"
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.config.cache_ttl:
                self.stats.cache_hits += 1
                return value
            else:
                del self._cache[key]
        self.stats.cache_misses += 1
        return None
    
    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached value."""
        self._cache[key] = (value, time.time())
    
    async def execute(
        self,
        func: Callable[..., Any],
        *args,
        fallback_value: Any = None,
        fallback_func: Optional[Callable[..., Any]] = None,
        **kwargs
    ) -> Any:
        """
        Execute with fallback support.
        
        Args:
            func: Primary async function
            *args: Positional arguments
            fallback_value: Static fallback value
            fallback_func: Alternative function
            **kwargs: Keyword arguments
        
        Returns:
            Primary result or fallback
        """
        cache_key = self._get_cache_key(func, args, kwargs)
        
        try:
            result = await func(*args, **kwargs)
            self.stats.primary_successes += 1
            
            # Cache successful result
            self._set_cached(cache_key, result)
            return result
        
        except Exception as e:
            self.stats.primary_failures += 1
            self.stats.fallback_activations += 1
            
            if self.config.log_fallback:
                logger.warning(f"Primary failed, activating fallback: {e}")
            
            # Try fallback strategies
            if self.config.strategy == FallbackStrategy.CACHE:
                cached = self._get_cached(cache_key)
                if cached is not None:
                    return cached
            
            if self.config.strategy == FallbackStrategy.ALTERNATIVE:
                if fallback_func is not None:
                    return await fallback_func(*args, **kwargs)
            
            if fallback_value is not None:
                return fallback_value
            
            raise
    
    def with_fallback(
        self,
        fallback_value: Any = None,
        fallback_func: Optional[Callable[..., Any]] = None,
    ) -> Callable:
        """Decorator for adding fallback to a function."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.execute(
                    func,
                    *args,
                    fallback_value=fallback_value,
                    fallback_func=fallback_func,
                    **kwargs
                )
            return wrapper
        return decorator
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "primary_successes": self.stats.primary_successes,
            "primary_failures": self.stats.primary_failures,
            "fallback_activations": self.stats.fallback_activations,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "fallback_rate": (
                self.stats.fallback_activations / 
                max(1, self.stats.primary_successes + self.stats.primary_failures)
            ),
        }
"@

$Fallback | Out-File -FilePath "$ResilienceDir\fallback.py" -Encoding UTF8
Write-Host "  ✅ Created: resilience/fallback.py" -ForegroundColor Green

# =============================================================================
# 5. Bulkhead Pattern Implementation
# =============================================================================

$Bulkhead = @"
"""
Bulkhead Pattern for Resource Isolation.

Isolates resources to prevent cascading failures:
- Limits concurrent calls
- Provides queue overflow protection
- Enables per-service isolation

Usage:
    bulkhead = Bulkhead(max_concurrent=10, max_queue=100)
    
    async with bulkhead:
        result = await call_service()
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead."""
    max_concurrent: int = 10
    max_queue: int = 100
    timeout: float = 30.0


@dataclass
class BulkheadStats:
    """Statistics for bulkhead monitoring."""
    accepted_calls: int = 0
    rejected_calls: int = 0
    active_calls: int = 0
    queued_calls: int = 0
    completed_calls: int = 0
    failed_calls: int = 0


class BulkheadFullError(Exception):
    """Raised when bulkhead is at capacity."""
    pass


class Bulkhead:
    """
    Bulkhead pattern implementation.
    
    Limits concurrent access to a resource to prevent
    cascade failures and resource exhaustion.
    """
    
    def __init__(
        self,
        name: str = "default",
        max_concurrent: int = 10,
        max_queue: int = 100,
        timeout: float = 30.0,
    ):
        self.name = name
        self.config = BulkheadConfig(
            max_concurrent=max_concurrent,
            max_queue=max_queue,
            timeout=timeout,
        )
        self.stats = BulkheadStats()
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_semaphore = asyncio.Semaphore(max_concurrent + max_queue)
        self._active = 0
        self._queued = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire a slot in the bulkhead."""
        # Check if we can queue
        if not self._queue_semaphore.locked():
            try:
                await asyncio.wait_for(
                    self._queue_semaphore.acquire(),
                    timeout=0.001  # Quick check
                )
            except asyncio.TimeoutError:
                self.stats.rejected_calls += 1
                raise BulkheadFullError(
                    f"Bulkhead '{self.name}' is full "
                    f"(max_concurrent={self.config.max_concurrent}, "
                    f"max_queue={self.config.max_queue})"
                )
        
        async with self._lock:
            self._queued += 1
            self.stats.queued_calls = self._queued
        
        # Wait for execution slot
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.config.timeout
            )
        except asyncio.TimeoutError:
            async with self._lock:
                self._queued -= 1
            self._queue_semaphore.release()
            self.stats.rejected_calls += 1
            raise BulkheadFullError(f"Bulkhead '{self.name}' timeout")
        
        async with self._lock:
            self._queued -= 1
            self._active += 1
            self.stats.active_calls = self._active
            self.stats.accepted_calls += 1
        
        return True
    
    def release(self, success: bool = True) -> None:
        """Release a slot in the bulkhead."""
        self._semaphore.release()
        self._queue_semaphore.release()
        
        # Update stats synchronously
        self._active = max(0, self._active - 1)
        self.stats.active_calls = self._active
        
        if success:
            self.stats.completed_calls += 1
        else:
            self.stats.failed_calls += 1
    
    async def execute(self, func, *args, **kwargs):
        """Execute a function within the bulkhead."""
        await self.acquire()
        try:
            result = await func(*args, **kwargs)
            self.release(success=True)
            return result
        except Exception as e:
            self.release(success=False)
            raise
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release(success=exc_type is None)
        return False
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "name": self.name,
            "max_concurrent": self.config.max_concurrent,
            "max_queue": self.config.max_queue,
            "active_calls": self.stats.active_calls,
            "queued_calls": self.stats.queued_calls,
            "accepted_calls": self.stats.accepted_calls,
            "rejected_calls": self.stats.rejected_calls,
            "completed_calls": self.stats.completed_calls,
            "failed_calls": self.stats.failed_calls,
        }
"@

$Bulkhead | Out-File -FilePath "$ResilienceDir\bulkhead.py" -Encoding UTF8
Write-Host "  ✅ Created: resilience/bulkhead.py" -ForegroundColor Green

# =============================================================================
# 6. Health Check Implementation
# =============================================================================

$HealthCheck = @"
"""
Health Check Implementation for Liveness and Readiness Probes.

Provides Kubernetes-compatible health endpoints:
- /health/live - Liveness probe
- /health/ready - Readiness probe
- Component health aggregation

Usage:
    health = HealthChecker()
    health.register("database", db.check_connection)
    health.register("cache", cache.ping)
    
    status = await health.check_readiness()
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Callable, Optional, Any, List
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    last_check: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Aggregated health report."""
    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY
    
    @property
    def is_ready(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
    
    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                    "details": c.details,
                }
                for c in self.components
            ]
        }


class HealthChecker:
    """
    Health checker with component registration.
    
    Manages health checks for multiple components and
    provides aggregated health status.
    """
    
    def __init__(self, check_timeout: float = 5.0):
        self.check_timeout = check_timeout
        self._checks: Dict[str, Callable] = {}
        self._critical: set = set()
        self._last_report: Optional[HealthReport] = None
    
    def register(
        self,
        name: str,
        check_func: Callable[[], Any],
        critical: bool = True,
    ) -> None:
        """
        Register a health check.
        
        Args:
            name: Component name
            check_func: Async function that returns True/dict if healthy
            critical: If False, failure only causes DEGRADED
        """
        self._checks[name] = check_func
        if critical:
            self._critical.add(name)
        logger.info(f"Registered health check: {name} (critical={critical})")
    
    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        self._checks.pop(name, None)
        self._critical.discard(name)
    
    async def _check_component(self, name: str) -> ComponentHealth:
        """Check health of a single component."""
        check_func = self._checks[name]
        start = asyncio.get_event_loop().time()
        
        try:
            result = await asyncio.wait_for(
                check_func() if asyncio.iscoroutinefunction(check_func)
                else asyncio.to_thread(check_func),
                timeout=self.check_timeout
            )
            
            latency = (asyncio.get_event_loop().time() - start) * 1000
            
            if isinstance(result, dict):
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    last_check=datetime.now(),
                    details=result,
                )
            elif result:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    last_check=datetime.now(),
                )
            else:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message="Check returned False",
                    latency_ms=latency,
                    last_check=datetime.now(),
                )
        
        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Timeout after {self.check_timeout}s",
                last_check=datetime.now(),
            )
        
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                last_check=datetime.now(),
            )
    
    async def check_all(self) -> HealthReport:
        """Run all health checks."""
        if not self._checks:
            return HealthReport(
                status=HealthStatus.HEALTHY,
                components=[],
            )
        
        # Run all checks concurrently
        tasks = [
            self._check_component(name)
            for name in self._checks
        ]
        components = await asyncio.gather(*tasks)
        
        # Determine overall status
        critical_unhealthy = any(
            c.status == HealthStatus.UNHEALTHY and c.name in self._critical
            for c in components
        )
        any_unhealthy = any(
            c.status == HealthStatus.UNHEALTHY
            for c in components
        )
        
        if critical_unhealthy:
            overall = HealthStatus.UNHEALTHY
        elif any_unhealthy:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        report = HealthReport(status=overall, components=list(components))
        self._last_report = report
        return report
    
    async def check_liveness(self) -> HealthReport:
        """
        Check if application is alive.
        
        Simple check that returns HEALTHY if the process is running.
        """
        return HealthReport(
            status=HealthStatus.HEALTHY,
            components=[
                ComponentHealth(
                    name="process",
                    status=HealthStatus.HEALTHY,
                    last_check=datetime.now(),
                )
            ]
        )
    
    async def check_readiness(self) -> HealthReport:
        """
        Check if application is ready to serve traffic.
        
        Runs all registered health checks.
        """
        return await self.check_all()
    
    def get_last_report(self) -> Optional[HealthReport]:
        """Get the last health report."""
        return self._last_report
"@

$HealthCheck | Out-File -FilePath "$ResilienceDir\health_check.py" -Encoding UTF8
Write-Host "  ✅ Created: resilience/health_check.py" -ForegroundColor Green

# =============================================================================
# 7. Create Tests
# =============================================================================

Write-Host "`n📝 Creating resilience tests..." -ForegroundColor Yellow

$ResilienceTests = @"
"""
Resilience Module Tests
Sprint 3.3: Resilience & Fault Tolerance
Created: January 6, 2026
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

# Add src to path
import sys
sys.path.insert(0, 'src')

from resilience import (
    CircuitBreaker, CircuitState,
    RetryPolicy, RetryConfig,
    FallbackHandler, FallbackStrategy,
    Bulkhead, BulkheadConfig,
    HealthChecker, HealthStatus,
)
from resilience.circuit_breaker import CircuitOpenError
from resilience.bulkhead import BulkheadFullError


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_initial_state_closed(self):
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_success_keeps_closed(self):
        breaker = CircuitBreaker()
        
        async def success():
            return "ok"
        
        result = await breaker.execute(success)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_failures_open_circuit(self):
        breaker = CircuitBreaker(failure_threshold=3)
        
        async def fail():
            raise ConnectionError("fail")
        
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await breaker.execute(fail)
        
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_open_circuit_rejects(self):
        breaker = CircuitBreaker(failure_threshold=1)
        
        async def fail():
            raise ConnectionError()
        
        with pytest.raises(ConnectionError):
            await breaker.execute(fail)
        
        async def success():
            return "ok"
        
        with pytest.raises(CircuitOpenError):
            await breaker.execute(success)


class TestRetryPolicy:
    """Test retry policy functionality."""
    
    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        retry = RetryPolicy(max_retries=3)
        
        async def success():
            return "ok"
        
        result = await retry.execute(success)
        assert result == "ok"
        assert retry.stats.retries_performed == 0
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        retry = RetryPolicy(max_retries=3, base_delay=0.01)
        attempts = 0
        
        async def fail_then_succeed():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError()
            return "ok"
        
        result = await retry.execute(fail_then_succeed)
        assert result == "ok"
        assert attempts == 3
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        retry = RetryPolicy(max_retries=2, base_delay=0.01)
        
        async def always_fail():
            raise ConnectionError()
        
        with pytest.raises(ConnectionError):
            await retry.execute(always_fail)
        
        assert retry.stats.total_attempts == 3  # 1 initial + 2 retries


class TestFallbackHandler:
    """Test fallback handler functionality."""
    
    @pytest.mark.asyncio
    async def test_primary_success(self):
        fallback = FallbackHandler()
        
        async def primary():
            return "primary"
        
        result = await fallback.execute(primary, fallback_value="fallback")
        assert result == "primary"
    
    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        fallback = FallbackHandler()
        
        async def fail():
            raise Exception("fail")
        
        result = await fallback.execute(fail, fallback_value="fallback")
        assert result == "fallback"
    
    @pytest.mark.asyncio
    async def test_fallback_function(self):
        fallback = FallbackHandler()
        
        async def fail():
            raise Exception()
        
        async def alt():
            return "alternative"
        
        result = await fallback.execute(fail, fallback_func=alt)
        assert result == "alternative"


class TestBulkhead:
    """Test bulkhead functionality."""
    
    @pytest.mark.asyncio
    async def test_allows_under_limit(self):
        bulkhead = Bulkhead(max_concurrent=5)
        
        async with bulkhead:
            assert bulkhead.stats.active_calls == 1
    
    @pytest.mark.asyncio
    async def test_rejects_over_limit(self):
        bulkhead = Bulkhead(max_concurrent=1, max_queue=0, timeout=0.1)
        
        async def slow():
            await asyncio.sleep(1)
        
        # Start first call
        task1 = asyncio.create_task(bulkhead.execute(slow))
        await asyncio.sleep(0.01)  # Let it acquire
        
        # Second call should be rejected
        with pytest.raises(BulkheadFullError):
            await bulkhead.execute(slow)
        
        task1.cancel()


class TestHealthChecker:
    """Test health checker functionality."""
    
    @pytest.mark.asyncio
    async def test_healthy_when_all_pass(self):
        checker = HealthChecker()
        
        async def healthy():
            return True
        
        checker.register("db", healthy)
        checker.register("cache", healthy)
        
        report = await checker.check_readiness()
        assert report.status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_unhealthy_when_critical_fails(self):
        checker = HealthChecker()
        
        async def healthy():
            return True
        
        async def unhealthy():
            raise Exception("down")
        
        checker.register("db", unhealthy, critical=True)
        checker.register("cache", healthy)
        
        report = await checker.check_readiness()
        assert report.status == HealthStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_liveness_always_healthy(self):
        checker = HealthChecker()
        report = await checker.check_liveness()
        assert report.status == HealthStatus.HEALTHY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"@

$ResilienceTests | Out-File -FilePath "$Phase2Dev\tests\test_resilience.py" -Encoding UTF8
Write-Host "  ✅ Created: tests/test_resilience.py" -ForegroundColor Green

# =============================================================================
# SUMMARY
# =============================================================================

Write-Host "`n╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║              SPRINT 3.3 BOOTSTRAP COMPLETE! ✅                ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green

Write-Host "`n📦 Files Created:" -ForegroundColor Cyan
Write-Host "  • src/resilience/__init__.py" -ForegroundColor White
Write-Host "  • src/resilience/circuit_breaker.py" -ForegroundColor White
Write-Host "  • src/resilience/retry_policy.py" -ForegroundColor White
Write-Host "  • src/resilience/fallback.py" -ForegroundColor White
Write-Host "  • src/resilience/bulkhead.py" -ForegroundColor White
Write-Host "  • src/resilience/health_check.py" -ForegroundColor White
Write-Host "  • tests/test_resilience.py" -ForegroundColor White

Write-Host "`n🚀 Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Run tests:" -ForegroundColor White
Write-Host "     cd $Phase2Dev" -ForegroundColor Gray
Write-Host "     pytest tests/test_resilience.py -v" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Integrate with serving layer" -ForegroundColor White
Write-Host "  3. Add health endpoints to API" -ForegroundColor White

Write-Host "`n✅ Sprint 3.3 infrastructure ready!" -ForegroundColor Green
