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
