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
