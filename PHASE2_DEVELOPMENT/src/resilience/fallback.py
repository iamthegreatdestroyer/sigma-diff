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
            # First: If explicit fallback_func provided, use it (takes precedence)
            if fallback_func is not None:
                return await fallback_func(*args, **kwargs)
            
            # Second: Try strategy-based fallbacks
            if self.config.strategy == FallbackStrategy.CACHE:
                cached = self._get_cached(cache_key)
                if cached is not None:
                    return cached
            
            # Third: Static fallback value
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
