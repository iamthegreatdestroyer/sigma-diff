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
