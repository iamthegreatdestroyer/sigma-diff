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
