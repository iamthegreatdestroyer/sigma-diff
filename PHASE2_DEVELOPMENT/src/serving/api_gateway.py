"""
API Gateway for Distributed Inference.

Provides:
- Rate limiting
- Authentication/Authorization
- Request transformation
- Response caching
- Logging and metrics

Sprint 3.4 - Serving Layer
Created: 2026-01-06
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable
from functools import wraps
import json

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_second: float = 100.0
    burst_size: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    key_func: Optional[Callable[[Any], str]] = None


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_at: float
    retry_after: Optional[float] = None


class RateLimiter(ABC):
    """Abstract rate limiter."""
    
    @abstractmethod
    def allow(self, key: str) -> RateLimitResult:
        """Check if request is allowed."""
        pass
    
    @abstractmethod
    def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        pass


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.buckets: Dict[str, Dict[str, float]] = {}
        self._lock = asyncio.Lock()
    
    def allow(self, key: str) -> RateLimitResult:
        now = time.time()
        
        if key not in self.buckets:
            self.buckets[key] = {
                "tokens": float(self.config.burst_size),
                "last_update": now
            }
        
        bucket = self.buckets[key]
        
        # Refill tokens
        elapsed = now - bucket["last_update"]
        new_tokens = elapsed * self.config.requests_per_second
        bucket["tokens"] = min(
            self.config.burst_size,
            bucket["tokens"] + new_tokens
        )
        bucket["last_update"] = now
        
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return RateLimitResult(
                allowed=True,
                remaining=int(bucket["tokens"]),
                reset_at=now + (self.config.burst_size - bucket["tokens"]) / self.config.requests_per_second
            )
        else:
            retry_after = (1.0 - bucket["tokens"]) / self.config.requests_per_second
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=now + retry_after,
                retry_after=retry_after
            )
    
    def reset(self, key: str) -> None:
        if key in self.buckets:
            del self.buckets[key]


class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiter."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.windows: Dict[str, List[float]] = {}
        self.window_size = 1.0  # 1 second window
    
    def allow(self, key: str) -> RateLimitResult:
        now = time.time()
        window_start = now - self.window_size
        
        if key not in self.windows:
            self.windows[key] = []
        
        # Remove old timestamps
        self.windows[key] = [t for t in self.windows[key] if t > window_start]
        
        current_count = len(self.windows[key])
        max_requests = int(self.config.requests_per_second * self.window_size)
        
        if current_count < max_requests:
            self.windows[key].append(now)
            return RateLimitResult(
                allowed=True,
                remaining=max_requests - current_count - 1,
                reset_at=now + self.window_size
            )
        else:
            oldest = min(self.windows[key]) if self.windows[key] else now
            retry_after = oldest + self.window_size - now
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=now + retry_after,
                retry_after=max(0, retry_after)
            )
    
    def reset(self, key: str) -> None:
        if key in self.windows:
            del self.windows[key]


class RateLimiterFactory:
    """Factory for creating rate limiters."""
    
    @staticmethod
    def create(config: RateLimitConfig) -> RateLimiter:
        if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return TokenBucketRateLimiter(config)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return SlidingWindowRateLimiter(config)
        else:
            return TokenBucketRateLimiter(config)


@dataclass
class AuthConfig:
    """Authentication configuration."""
    enabled: bool = True
    api_key_header: str = "X-API-Key"
    bearer_header: str = "Authorization"
    allow_anonymous: bool = False


class AuthProvider(ABC):
    """Abstract authentication provider."""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Authenticate and return user context or None."""
        pass


class APIKeyAuthProvider(AuthProvider):
    """API key authentication provider."""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
    
    def register_key(self, api_key: str, metadata: Dict[str, Any]) -> None:
        """Register an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        self.api_keys[key_hash] = metadata
        logger.info(f"Registered API key for {metadata.get('name', 'unknown')}")
    
    async def authenticate(self, credentials: Dict[str, str]) -> Optional[Dict[str, Any]]:
        api_key = credentials.get("api_key")
        if not api_key:
            return None
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return self.api_keys.get(key_hash)


class JWTAuthProvider(AuthProvider):
    """JWT authentication provider (simplified)."""
    
    def __init__(self, secret: str):
        self.secret = secret
    
    async def authenticate(self, credentials: Dict[str, str]) -> Optional[Dict[str, Any]]:
        token = credentials.get("bearer_token")
        if not token:
            return None
        
        # In production, properly decode and verify JWT
        # This is a simplified placeholder
        try:
            # Verify token signature (placeholder)
            parts = token.split(".")
            if len(parts) != 3:
                return None
            
            # Decode payload (placeholder - in production use proper JWT library)
            import base64
            payload_b64 = parts[1] + "=="  # Add padding
            try:
                payload_json = base64.urlsafe_b64decode(payload_b64)
                payload = json.loads(payload_json)
                return payload
            except Exception:
                return None
                
        except Exception as e:
            logger.warning(f"JWT verification failed: {e}")
            return None


@dataclass
class GatewayRequest:
    """Request passing through the gateway."""
    request_id: str
    path: str
    method: str
    headers: Dict[str, str]
    body: Optional[Any] = None
    query_params: Dict[str, str] = field(default_factory=dict)
    client_ip: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class GatewayResponse:
    """Response from the gateway."""
    status_code: int
    body: Any
    headers: Dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class GatewayConfig:
    """API Gateway configuration."""
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    enable_caching: bool = True
    cache_ttl: float = 60.0
    enable_logging: bool = True
    enable_metrics: bool = True
    timeout: float = 30.0
    max_body_size: int = 10 * 1024 * 1024  # 10MB


Middleware = Callable[[GatewayRequest, Callable], Awaitable[GatewayResponse]]


class APIGateway:
    """
    API Gateway for inference requests.
    
    Features:
    - Rate limiting
    - Authentication
    - Request/Response transformation
    - Caching
    - Middleware pipeline
    """
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        self.rate_limiter = RateLimiterFactory.create(self.config.rate_limit)
        self.auth_providers: List[AuthProvider] = []
        self.middlewares: List[Middleware] = []
        self.response_cache: Dict[str, tuple[GatewayResponse, float]] = {}
        self.stats = GatewayStats()
        
        # Set up default middleware
        self._setup_default_middleware()
    
    def _setup_default_middleware(self) -> None:
        """Set up default middleware pipeline."""
        # Middlewares are applied in order
        pass
    
    def add_auth_provider(self, provider: AuthProvider) -> None:
        """Add an authentication provider."""
        self.auth_providers.append(provider)
    
    def add_middleware(self, middleware: Middleware) -> None:
        """Add a middleware to the pipeline."""
        self.middlewares.append(middleware)
    
    async def handle(
        self,
        request: GatewayRequest,
        handler: Callable[[GatewayRequest], Awaitable[GatewayResponse]]
    ) -> GatewayResponse:
        """Handle a request through the gateway pipeline."""
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            # 1. Rate limiting
            rate_key = request.client_ip or "default"
            rate_result = self.rate_limiter.allow(rate_key)
            
            if not rate_result.allowed:
                self.stats.rate_limited_requests += 1
                return GatewayResponse(
                    status_code=429,
                    body={"error": "Rate limit exceeded"},
                    headers={
                        "Retry-After": str(int(rate_result.retry_after or 1)),
                        "X-RateLimit-Remaining": "0"
                    }
                )
            
            # 2. Authentication
            if self.config.auth.enabled:
                auth_result = await self._authenticate(request)
                if auth_result is None and not self.config.auth.allow_anonymous:
                    self.stats.auth_failures += 1
                    return GatewayResponse(
                        status_code=401,
                        body={"error": "Unauthorized"}
                    )
            
            # 3. Check cache
            if self.config.enable_caching and request.method == "GET":
                cached = self._get_cached(request)
                if cached:
                    self.stats.cache_hits += 1
                    return cached
            
            # 4. Apply middlewares and call handler
            response = await self._apply_middlewares(request, handler)
            
            # 5. Cache response
            if self.config.enable_caching and request.method == "GET" and response.status_code == 200:
                self._cache_response(request, response)
            
            # 6. Record metrics
            latency_ms = (time.time() - start_time) * 1000
            response.latency_ms = latency_ms
            self.stats.record_latency(latency_ms)
            
            return response
            
        except asyncio.TimeoutError:
            self.stats.timeout_requests += 1
            return GatewayResponse(
                status_code=504,
                body={"error": "Gateway timeout"}
            )
        except Exception as e:
            self.stats.error_requests += 1
            logger.error(f"Gateway error: {e}")
            return GatewayResponse(
                status_code=500,
                body={"error": "Internal server error"}
            )
    
    async def _authenticate(self, request: GatewayRequest) -> Optional[Dict[str, Any]]:
        """Authenticate the request."""
        credentials = {}
        
        # Extract API key
        api_key = request.headers.get(self.config.auth.api_key_header)
        if api_key:
            credentials["api_key"] = api_key
        
        # Extract Bearer token
        auth_header = request.headers.get(self.config.auth.bearer_header, "")
        if auth_header.startswith("Bearer "):
            credentials["bearer_token"] = auth_header[7:]
        
        # Try each provider
        for provider in self.auth_providers:
            result = await provider.authenticate(credentials)
            if result:
                return result
        
        return None
    
    async def _apply_middlewares(
        self,
        request: GatewayRequest,
        handler: Callable[[GatewayRequest], Awaitable[GatewayResponse]]
    ) -> GatewayResponse:
        """Apply middleware pipeline."""
        if not self.middlewares:
            return await asyncio.wait_for(
                handler(request),
                timeout=self.config.timeout
            )
        
        # Build middleware chain
        async def chain(idx: int, req: GatewayRequest) -> GatewayResponse:
            if idx >= len(self.middlewares):
                return await asyncio.wait_for(
                    handler(req),
                    timeout=self.config.timeout
                )
            
            middleware = self.middlewares[idx]
            return await middleware(req, lambda r: chain(idx + 1, r))
        
        return await chain(0, request)
    
    def _get_cache_key(self, request: GatewayRequest) -> str:
        """Generate cache key for a request."""
        return hashlib.sha256(
            f"{request.method}:{request.path}:{json.dumps(request.query_params, sort_keys=True)}".encode()
        ).hexdigest()
    
    def _get_cached(self, request: GatewayRequest) -> Optional[GatewayResponse]:
        """Get cached response if valid."""
        key = self._get_cache_key(request)
        if key in self.response_cache:
            response, cached_at = self.response_cache[key]
            if time.time() - cached_at < self.config.cache_ttl:
                return response
            else:
                del self.response_cache[key]
        return None
    
    def _cache_response(self, request: GatewayRequest, response: GatewayResponse) -> None:
        """Cache a response."""
        key = self._get_cache_key(request)
        self.response_cache[key] = (response, time.time())
    
    def clear_cache(self) -> None:
        """Clear response cache."""
        self.response_cache.clear()


@dataclass
class GatewayStats:
    """Gateway statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    error_requests: int = 0
    rate_limited_requests: int = 0
    auth_failures: int = 0
    cache_hits: int = 0
    timeout_requests: int = 0
    total_latency_ms: float = 0.0
    
    def record_latency(self, latency_ms: float) -> None:
        """Record a request latency."""
        self.total_latency_ms += latency_ms
        self.successful_requests += 1
    
    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100


def rate_limit(config: Optional[RateLimitConfig] = None):
    """Decorator for rate limiting functions."""
    limiter = RateLimiterFactory.create(config or RateLimitConfig())
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract key (first arg or 'default')
            key = str(args[0]) if args else "default"
            result = limiter.allow(key)
            
            if not result.allowed:
                raise RateLimitExceeded(result.retry_after or 1.0)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.2f}s")
