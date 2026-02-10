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
