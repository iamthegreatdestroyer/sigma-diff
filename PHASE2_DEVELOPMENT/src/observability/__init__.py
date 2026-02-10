"""
Observability Package - Unified Observability for Distributed Inference.

This package provides a unified observability solution combining:
- Prometheus metrics export
- Jaeger distributed tracing  
- Structured logging
- Health checking

Sprint 3.5 - Observability Stack
Created: 2026-01-06
"""

from src.observability.unified_client import (
    ObservabilityClient,
    ObservabilityConfig,
    RequestContext,
    StructuredLogger,
    HealthChecker,
    create_observability_client,
    create_test_client
)

__all__ = [
    "ObservabilityClient",
    "ObservabilityConfig",
    "RequestContext",
    "StructuredLogger",
    "HealthChecker",
    "create_observability_client",
    "create_test_client"
]
