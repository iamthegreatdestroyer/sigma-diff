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
