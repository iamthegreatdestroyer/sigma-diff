"""
Serving and request handling.

Sprint 3.4 - Serving Layer components for distributed inference.
"""

from .unified_pipeline import (
    UnifiedInferencePipeline,
    InferencePipelineExecutor,
    PipelineConfig,
    GenerationRequest,
    GenerationOutput,
    create_inference_pipeline,
)

from .load_balancer import (
    LoadBalancerStrategy,
    BackendNode,
    LoadBalancerConfig,
    LoadBalancer,
    RoundRobinBalancer,
    LeastConnectionsBalancer,
    WeightedRoundRobinBalancer,
    ConsistentHashBalancer,
    LatencyBasedBalancer,
    LoadBalancerFactory,
    HealthChecker,
)

from .request_router import (
    RoutingStrategy,
    Route,
    RoutingRequest,
    RouteMatch,
    Router,
    PathBasedRouter,
    ContentBasedRouter,
    HeaderBasedRouter,
    ModelBasedRouter,
    CompositeRouter,
    RouterFactory,
    RoutingMiddleware,
)

from .api_gateway import (
    RateLimitStrategy,
    RateLimiter,
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    RateLimiterFactory,
    AuthProvider,
    APIKeyAuthProvider,
    JWTAuthProvider,
    GatewayRequest,
    GatewayResponse,
    GatewayConfig,
    APIGateway,
)

from .grpc_server import (
    GRPCStatusCode,
    GRPCRequest,
    GRPCResponse,
    GRPCMetadata,
    GenerateRequest,
    GenerateResponse,
    StreamToken,
    InferenceServicer,
    ModelServicer,
    HealthServicer,
    GRPCServer,
    GRPCServerConfig,
    GRPCServerStats,
)

__all__ = [
    # Unified Pipeline
    "UnifiedInferencePipeline",
    "InferencePipelineExecutor",
    "PipelineConfig",
    "GenerationRequest",
    "GenerationOutput",
    "create_inference_pipeline",
    # Load Balancer
    "LoadBalancerStrategy",
    "BackendNode",
    "LoadBalancerConfig",
    "LoadBalancer",
    "RoundRobinBalancer",
    "LeastConnectionsBalancer",
    "WeightedRoundRobinBalancer",
    "ConsistentHashBalancer",
    "LatencyBasedBalancer",
    "LoadBalancerFactory",
    "HealthChecker",
    # Request Router
    "RoutingStrategy",
    "Route",
    "RoutingRequest",
    "RouteMatch",
    "Router",
    "PathBasedRouter",
    "ContentBasedRouter",
    "HeaderBasedRouter",
    "ModelBasedRouter",
    "CompositeRouter",
    "RouterFactory",
    "RoutingMiddleware",
    # API Gateway
    "RateLimitStrategy",
    "RateLimiter",
    "TokenBucketRateLimiter",
    "SlidingWindowRateLimiter",
    "RateLimiterFactory",
    "AuthProvider",
    "APIKeyAuthProvider",
    "JWTAuthProvider",
    "GatewayRequest",
    "GatewayResponse",
    "GatewayConfig",
    "APIGateway",
    # gRPC Server
    "GRPCStatusCode",
    "GRPCRequest",
    "GRPCResponse",
    "GRPCMetadata",
    "GenerateRequest",
    "GenerateResponse",
    "StreamToken",
    "InferenceServicer",
    "ModelServicer",
    "HealthServicer",
    "GRPCServer",
    "GRPCServerConfig",
    "GRPCServerStats",
]
