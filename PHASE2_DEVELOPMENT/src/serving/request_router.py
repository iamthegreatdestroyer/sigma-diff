"""
Request Router for Distributed Inference.

Implements routing strategies:
- Path-based routing
- Content-based routing
- Header-based routing
- Model-based routing

Sprint 3.4 - Serving Layer
Created: 2026-01-06
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Pattern
import time

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategies."""
    PATH_BASED = "path_based"
    CONTENT_BASED = "content_based"
    HEADER_BASED = "header_based"
    MODEL_BASED = "model_based"
    WEIGHTED = "weighted"


@dataclass
class RouteMatch:
    """Result of a route match."""
    matched: bool
    route_id: str
    backend: str
    params: Dict[str, str] = field(default_factory=dict)
    priority: int = 0


@dataclass
class Route:
    """Defines a routing rule."""
    id: str
    pattern: str
    backend: str
    priority: int = 0
    methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    headers: Dict[str, str] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def __post_init__(self):
        self._compiled_pattern: Optional[Pattern] = None
        self._compile_pattern()
    
    def _compile_pattern(self) -> None:
        """Compile the route pattern to regex."""
        # Convert path params like {id} to regex groups
        regex_pattern = re.sub(r'\{(\w+)\}', r'(?P<\1>[^/]+)', self.pattern)
        regex_pattern = f"^{regex_pattern}$"
        self._compiled_pattern = re.compile(regex_pattern)
    
    def match(self, path: str) -> Optional[Dict[str, str]]:
        """Match a path against this route."""
        if not self.enabled or not self._compiled_pattern:
            return None
        
        match = self._compiled_pattern.match(path)
        if match:
            return match.groupdict()
        return None


@dataclass
class RoutingRequest:
    """Incoming request for routing."""
    path: str
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Any] = None
    model_id: Optional[str] = None
    request_id: Optional[str] = None


class Router(ABC):
    """Abstract base class for routers."""
    
    @abstractmethod
    def route(self, request: RoutingRequest) -> Optional[RouteMatch]:
        """Route a request to a backend."""
        pass
    
    @abstractmethod
    def add_route(self, route: Route) -> None:
        """Add a route."""
        pass
    
    @abstractmethod
    def remove_route(self, route_id: str) -> None:
        """Remove a route."""
        pass


class PathBasedRouter(Router):
    """Path-based router."""
    
    def __init__(self):
        self.routes: Dict[str, Route] = {}
        self._sorted_routes: List[Route] = []
    
    def add_route(self, route: Route) -> None:
        self.routes[route.id] = route
        self._resort_routes()
        logger.info(f"Added route {route.id}: {route.pattern} -> {route.backend}")
    
    def remove_route(self, route_id: str) -> None:
        if route_id in self.routes:
            del self.routes[route_id]
            self._resort_routes()
            logger.info(f"Removed route {route_id}")
    
    def _resort_routes(self) -> None:
        """Sort routes by priority (higher first)."""
        self._sorted_routes = sorted(
            self.routes.values(),
            key=lambda r: r.priority,
            reverse=True
        )
    
    def route(self, request: RoutingRequest) -> Optional[RouteMatch]:
        for route in self._sorted_routes:
            if not route.enabled:
                continue
            
            # Check method
            if request.method.upper() not in [m.upper() for m in route.methods]:
                continue
            
            # Check path match
            params = route.match(request.path)
            if params is not None:
                return RouteMatch(
                    matched=True,
                    route_id=route.id,
                    backend=route.backend,
                    params=params,
                    priority=route.priority
                )
        
        return None


class ContentBasedRouter(Router):
    """Content-based router that routes based on request body."""
    
    def __init__(self):
        self.routes: Dict[str, Route] = {}
        self._content_matchers: Dict[str, Callable[[Any], bool]] = {}
    
    def add_route(self, route: Route) -> None:
        self.routes[route.id] = route
        logger.info(f"Added content route {route.id} -> {route.backend}")
    
    def remove_route(self, route_id: str) -> None:
        if route_id in self.routes:
            del self.routes[route_id]
            if route_id in self._content_matchers:
                del self._content_matchers[route_id]
    
    def add_content_matcher(
        self,
        route_id: str,
        matcher: Callable[[Any], bool]
    ) -> None:
        """Add a content matcher function for a route."""
        self._content_matchers[route_id] = matcher
    
    def route(self, request: RoutingRequest) -> Optional[RouteMatch]:
        for route_id, route in self.routes.items():
            if not route.enabled:
                continue
            
            # Check content conditions
            if route_id in self._content_matchers:
                if self._content_matchers[route_id](request.body):
                    return RouteMatch(
                        matched=True,
                        route_id=route_id,
                        backend=route.backend,
                        priority=route.priority
                    )
            
            # Check conditions in route
            if self._match_conditions(request.body, route.conditions):
                return RouteMatch(
                    matched=True,
                    route_id=route_id,
                    backend=route.backend,
                    priority=route.priority
                )
        
        return None
    
    def _match_conditions(self, body: Any, conditions: Dict[str, Any]) -> bool:
        """Match body against conditions."""
        if not conditions or body is None:
            return False
        
        for key, expected in conditions.items():
            if isinstance(body, dict):
                if key not in body or body[key] != expected:
                    return False
            else:
                return False
        
        return True


class HeaderBasedRouter(Router):
    """Header-based router."""
    
    def __init__(self):
        self.routes: Dict[str, Route] = {}
    
    def add_route(self, route: Route) -> None:
        self.routes[route.id] = route
        logger.info(f"Added header route {route.id} -> {route.backend}")
    
    def remove_route(self, route_id: str) -> None:
        if route_id in self.routes:
            del self.routes[route_id]
    
    def route(self, request: RoutingRequest) -> Optional[RouteMatch]:
        for route in self.routes.values():
            if not route.enabled:
                continue
            
            # Check if all required headers match
            if self._headers_match(request.headers, route.headers):
                return RouteMatch(
                    matched=True,
                    route_id=route.id,
                    backend=route.backend,
                    priority=route.priority
                )
        
        return None
    
    def _headers_match(
        self,
        request_headers: Dict[str, str],
        route_headers: Dict[str, str]
    ) -> bool:
        """Check if request headers match route requirements."""
        if not route_headers:
            return False
        
        for key, expected in route_headers.items():
            # Case-insensitive header matching
            request_val = None
            for req_key, req_val in request_headers.items():
                if req_key.lower() == key.lower():
                    request_val = req_val
                    break
            
            if request_val is None:
                return False
            
            # Support regex patterns in header values
            if expected.startswith("~"):
                pattern = expected[1:]
                if not re.match(pattern, request_val):
                    return False
            elif request_val != expected:
                return False
        
        return True


class ModelBasedRouter(Router):
    """Routes requests to specific model backends."""
    
    def __init__(self):
        self.routes: Dict[str, Route] = {}
        self.model_backends: Dict[str, str] = {}  # model_id -> backend
        self.default_backend: Optional[str] = None
    
    def add_route(self, route: Route) -> None:
        self.routes[route.id] = route
        # Extract model mapping from metadata
        if "model_id" in route.metadata:
            self.model_backends[route.metadata["model_id"]] = route.backend
        logger.info(f"Added model route {route.id} -> {route.backend}")
    
    def remove_route(self, route_id: str) -> None:
        if route_id in self.routes:
            route = self.routes[route_id]
            if "model_id" in route.metadata:
                model_id = route.metadata["model_id"]
                if model_id in self.model_backends:
                    del self.model_backends[model_id]
            del self.routes[route_id]
    
    def set_default_backend(self, backend: str) -> None:
        """Set the default backend for unmatched models."""
        self.default_backend = backend
    
    def register_model(self, model_id: str, backend: str) -> None:
        """Register a model to a backend."""
        self.model_backends[model_id] = backend
        logger.info(f"Registered model {model_id} -> {backend}")
    
    def route(self, request: RoutingRequest) -> Optional[RouteMatch]:
        model_id = request.model_id
        
        # Try to get from body if not in request
        if model_id is None and isinstance(request.body, dict):
            model_id = request.body.get("model") or request.body.get("model_id")
        
        if model_id and model_id in self.model_backends:
            return RouteMatch(
                matched=True,
                route_id=f"model_{model_id}",
                backend=self.model_backends[model_id],
                params={"model_id": model_id}
            )
        
        # Use default backend
        if self.default_backend:
            return RouteMatch(
                matched=True,
                route_id="default",
                backend=self.default_backend,
                params={"model_id": model_id or "default"}
            )
        
        return None


class CompositeRouter(Router):
    """Combines multiple routers with priority."""
    
    def __init__(self):
        self.routers: List[tuple[int, Router]] = []  # (priority, router)
    
    def add_router(self, router: Router, priority: int = 0) -> None:
        """Add a router with priority (higher = checked first)."""
        self.routers.append((priority, router))
        self.routers.sort(key=lambda x: x[0], reverse=True)
    
    def add_route(self, route: Route) -> None:
        # Add to first router
        if self.routers:
            self.routers[0][1].add_route(route)
    
    def remove_route(self, route_id: str) -> None:
        for _, router in self.routers:
            router.remove_route(route_id)
    
    def route(self, request: RoutingRequest) -> Optional[RouteMatch]:
        for _, router in self.routers:
            match = router.route(request)
            if match and match.matched:
                return match
        return None


class RouterFactory:
    """Factory for creating routers."""
    
    @staticmethod
    def create(strategy: RoutingStrategy) -> Router:
        """Create a router with the specified strategy."""
        if strategy == RoutingStrategy.PATH_BASED:
            return PathBasedRouter()
        elif strategy == RoutingStrategy.CONTENT_BASED:
            return ContentBasedRouter()
        elif strategy == RoutingStrategy.HEADER_BASED:
            return HeaderBasedRouter()
        elif strategy == RoutingStrategy.MODEL_BASED:
            return ModelBasedRouter()
        else:
            return PathBasedRouter()


@dataclass
class RoutingStats:
    """Statistics for routing."""
    total_requests: int = 0
    matched_requests: int = 0
    unmatched_requests: int = 0
    route_hits: Dict[str, int] = field(default_factory=dict)
    avg_routing_time_us: float = 0.0
    
    def record_match(self, route_id: str, routing_time_us: float) -> None:
        """Record a successful route match."""
        self.total_requests += 1
        self.matched_requests += 1
        self.route_hits[route_id] = self.route_hits.get(route_id, 0) + 1
        
        # Update running average
        n = self.total_requests
        self.avg_routing_time_us = (
            (self.avg_routing_time_us * (n - 1) + routing_time_us) / n
        )
    
    def record_miss(self, routing_time_us: float) -> None:
        """Record a routing miss."""
        self.total_requests += 1
        self.unmatched_requests += 1


class RoutingMiddleware:
    """Middleware for request routing with statistics."""
    
    def __init__(self, router: Router):
        self.router = router
        self.stats = RoutingStats()
    
    def route(self, request: RoutingRequest) -> Optional[RouteMatch]:
        """Route a request and collect stats."""
        start = time.perf_counter()
        match = self.router.route(request)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        
        if match and match.matched:
            self.stats.record_match(match.route_id, elapsed_us)
        else:
            self.stats.record_miss(elapsed_us)
        
        return match
    
    def get_stats(self) -> RoutingStats:
        """Get routing statistics."""
        return self.stats
