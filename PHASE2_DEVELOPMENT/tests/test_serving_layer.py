"""
Tests for Sprint 3.4 - Serving Layer.

Tests:
- Load Balancer (Round Robin, Least Connections, Weighted, Consistent Hash, Latency)
- Request Router (Path-based, Content-based, Header-based, Model-based)
- API Gateway (Rate Limiting, Authentication, Caching)
- gRPC Server (Inference, Health, Streaming)

Sprint 3.4 - Serving Layer
Created: 2026-01-06
"""

import pytest
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Any

# Import serving components directly to avoid transitive import errors from __init__.py
import sys
import os
import importlib.util

# Get the path to the src/serving directory
serving_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'serving')

def load_module_direct(name, filepath):
    """Load a module directly from file, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load modules directly
load_balancer = load_module_direct('load_balancer', os.path.join(serving_dir, 'load_balancer.py'))
request_router = load_module_direct('request_router', os.path.join(serving_dir, 'request_router.py'))
api_gateway = load_module_direct('api_gateway', os.path.join(serving_dir, 'api_gateway.py'))
grpc_server = load_module_direct('grpc_server', os.path.join(serving_dir, 'grpc_server.py'))

# Import classes from loaded modules
LoadBalancerStrategy = load_balancer.LoadBalancerStrategy
LoadBalancerConfig = load_balancer.LoadBalancerConfig
BackendNode = load_balancer.BackendNode
RoundRobinBalancer = load_balancer.RoundRobinBalancer
LeastConnectionsBalancer = load_balancer.LeastConnectionsBalancer
WeightedRoundRobinBalancer = load_balancer.WeightedRoundRobinBalancer
ConsistentHashBalancer = load_balancer.ConsistentHashBalancer
LatencyBasedBalancer = load_balancer.LatencyBasedBalancer
LoadBalancerFactory = load_balancer.LoadBalancerFactory

RoutingStrategy = request_router.RoutingStrategy
Route = request_router.Route
RoutingRequest = request_router.RoutingRequest
PathBasedRouter = request_router.PathBasedRouter
ContentBasedRouter = request_router.ContentBasedRouter
HeaderBasedRouter = request_router.HeaderBasedRouter
ModelBasedRouter = request_router.ModelBasedRouter
CompositeRouter = request_router.CompositeRouter
RouterFactory = request_router.RouterFactory
RoutingMiddleware = request_router.RoutingMiddleware

RateLimitStrategy = api_gateway.RateLimitStrategy
RateLimitConfig = api_gateway.RateLimitConfig
TokenBucketRateLimiter = api_gateway.TokenBucketRateLimiter
SlidingWindowRateLimiter = api_gateway.SlidingWindowRateLimiter
RateLimiterFactory = api_gateway.RateLimiterFactory
APIGateway = api_gateway.APIGateway
GatewayConfig = api_gateway.GatewayConfig
GatewayRequest = api_gateway.GatewayRequest
GatewayResponse = api_gateway.GatewayResponse
APIKeyAuthProvider = api_gateway.APIKeyAuthProvider

GRPCServer = grpc_server.GRPCServer
GRPCServerConfig = grpc_server.GRPCServerConfig
GRPCRequest = grpc_server.GRPCRequest
GRPCResponse = grpc_server.GRPCResponse
GRPCStatusCode = grpc_server.GRPCStatusCode
GenerateRequest = grpc_server.GenerateRequest
HealthCheckRequest = grpc_server.HealthCheckRequest
DefaultInferenceServicer = grpc_server.DefaultInferenceServicer
DefaultHealthServicer = grpc_server.DefaultHealthServicer


# ============= Load Balancer Tests =============

class TestRoundRobinBalancer:
    """Tests for round-robin load balancer."""
    
    def test_empty_returns_none(self):
        balancer = RoundRobinBalancer()
        assert balancer.select_node() is None
    
    def test_single_node(self):
        balancer = RoundRobinBalancer()
        node = BackendNode(id="node1", host="localhost", port=8080)
        balancer.add_node(node)
        
        selected = balancer.select_node()
        assert selected is not None
        assert selected.id == "node1"
    
    def test_round_robin_distribution(self):
        balancer = RoundRobinBalancer()
        for i in range(3):
            balancer.add_node(BackendNode(id=f"node{i}", host="localhost", port=8080+i))
        
        # Select 6 times, should cycle through all nodes twice
        selections = [balancer.select_node().id for _ in range(6)]
        assert selections.count("node0") == 2
        assert selections.count("node1") == 2
        assert selections.count("node2") == 2
    
    def test_unhealthy_node_skipped(self):
        balancer = RoundRobinBalancer()
        balancer.add_node(BackendNode(id="healthy", host="localhost", port=8080))
        balancer.add_node(BackendNode(id="unhealthy", host="localhost", port=8081))
        
        balancer.mark_unhealthy("unhealthy")
        
        # All selections should be healthy node
        for _ in range(5):
            assert balancer.select_node().id == "healthy"


class TestLeastConnectionsBalancer:
    """Tests for least-connections load balancer."""
    
    def test_selects_lowest_connections(self):
        balancer = LeastConnectionsBalancer()
        
        node1 = BackendNode(id="node1", host="localhost", port=8080, active_connections=5)
        node2 = BackendNode(id="node2", host="localhost", port=8081, active_connections=2)
        node3 = BackendNode(id="node3", host="localhost", port=8082, active_connections=8)
        
        balancer.add_node(node1)
        balancer.add_node(node2)
        balancer.add_node(node3)
        
        selected = balancer.select_node()
        assert selected.id == "node2"  # Lowest connections
    
    def test_acquire_release_connection(self):
        balancer = LeastConnectionsBalancer()
        node = BackendNode(id="node1", host="localhost", port=8080)
        balancer.add_node(node)
        
        assert balancer.acquire_connection("node1")
        assert balancer.nodes["node1"].active_connections == 1
        
        balancer.release_connection("node1")
        assert balancer.nodes["node1"].active_connections == 0


class TestWeightedRoundRobinBalancer:
    """Tests for weighted round-robin load balancer."""
    
    def test_weighted_distribution(self):
        balancer = WeightedRoundRobinBalancer()
        
        # Node with weight 2 should get twice as many requests
        balancer.add_node(BackendNode(id="heavy", host="localhost", port=8080, weight=2))
        balancer.add_node(BackendNode(id="light", host="localhost", port=8081, weight=1))
        
        selections = [balancer.select_node().id for _ in range(9)]
        
        heavy_count = selections.count("heavy")
        light_count = selections.count("light")
        
        # Heavy should get roughly 2x the requests
        assert heavy_count >= light_count


class TestConsistentHashBalancer:
    """Tests for consistent hash load balancer."""
    
    def test_same_key_same_node(self):
        balancer = ConsistentHashBalancer()
        balancer.add_node(BackendNode(id="node1", host="localhost", port=8080))
        balancer.add_node(BackendNode(id="node2", host="localhost", port=8081))
        balancer.add_node(BackendNode(id="node3", host="localhost", port=8082))
        
        # Same key should always go to same node
        key = "user_session_123"
        first_selection = balancer.select_node(key)
        
        for _ in range(10):
            assert balancer.select_node(key).id == first_selection.id
    
    def test_distribution_across_nodes(self):
        balancer = ConsistentHashBalancer()
        for i in range(3):
            balancer.add_node(BackendNode(id=f"node{i}", host="localhost", port=8080+i))
        
        # Different keys should distribute across nodes
        node_counts: Dict[str, int] = {}
        for i in range(100):
            node = balancer.select_node(f"key_{i}")
            node_counts[node.id] = node_counts.get(node.id, 0) + 1
        
        # All nodes should get some traffic
        assert len(node_counts) > 1


class TestLatencyBasedBalancer:
    """Tests for latency-based load balancer."""
    
    def test_selects_lowest_latency(self):
        balancer = LatencyBasedBalancer()
        
        balancer.add_node(BackendNode(id="slow", host="localhost", port=8080))
        balancer.add_node(BackendNode(id="fast", host="localhost", port=8081))
        balancer.add_node(BackendNode(id="medium", host="localhost", port=8082))
        
        # Record latencies
        for _ in range(10):
            balancer.record_latency("slow", 100.0)
            balancer.record_latency("fast", 10.0)
            balancer.record_latency("medium", 50.0)
        
        selected = balancer.select_node()
        assert selected.id == "fast"


class TestLoadBalancerFactory:
    """Tests for load balancer factory."""
    
    def test_creates_round_robin(self):
        balancer = LoadBalancerFactory.create(LoadBalancerStrategy.ROUND_ROBIN)
        assert isinstance(balancer, RoundRobinBalancer)
    
    def test_creates_least_connections(self):
        balancer = LoadBalancerFactory.create(LoadBalancerStrategy.LEAST_CONNECTIONS)
        assert isinstance(balancer, LeastConnectionsBalancer)


# ============= Request Router Tests =============

class TestPathBasedRouter:
    """Tests for path-based router."""
    
    def test_exact_match(self):
        router = PathBasedRouter()
        router.add_route(Route(id="gen", pattern="/v1/generate", backend="inference"))
        
        request = RoutingRequest(path="/v1/generate", method="POST")
        match = router.route(request)
        
        assert match is not None
        assert match.matched
        assert match.backend == "inference"
    
    def test_path_params(self):
        router = PathBasedRouter()
        router.add_route(Route(
            id="model",
            pattern="/v1/models/{model_id}",
            backend="model_service"
        ))
        
        request = RoutingRequest(path="/v1/models/llama-7b", method="GET")
        match = router.route(request)
        
        assert match is not None
        assert match.params["model_id"] == "llama-7b"
    
    def test_method_filtering(self):
        router = PathBasedRouter()
        router.add_route(Route(
            id="create",
            pattern="/v1/models",
            backend="model_service",
            methods=["POST"]
        ))
        
        # POST should match
        match = router.route(RoutingRequest(path="/v1/models", method="POST"))
        assert match is not None
        
        # GET should not match
        match = router.route(RoutingRequest(path="/v1/models", method="GET"))
        assert match is None
    
    def test_priority_ordering(self):
        router = PathBasedRouter()
        router.add_route(Route(id="low", pattern="/api/test", backend="low", priority=1))
        router.add_route(Route(id="high", pattern="/api/test", backend="high", priority=10))
        
        match = router.route(RoutingRequest(path="/api/test"))
        assert match.backend == "high"  # Higher priority wins


class TestContentBasedRouter:
    """Tests for content-based router."""
    
    def test_condition_matching(self):
        router = ContentBasedRouter()
        router.add_route(Route(
            id="llama",
            pattern="",
            backend="llama_backend",
            conditions={"model": "llama"}
        ))
        
        request = RoutingRequest(
            path="/generate",
            body={"model": "llama", "prompt": "Hello"}
        )
        match = router.route(request)
        
        assert match is not None
        assert match.backend == "llama_backend"


class TestHeaderBasedRouter:
    """Tests for header-based router."""
    
    def test_header_matching(self):
        router = HeaderBasedRouter()
        router.add_route(Route(
            id="v2",
            pattern="",
            backend="v2_backend",
            headers={"X-API-Version": "2.0"}
        ))
        
        request = RoutingRequest(
            path="/api",
            headers={"X-API-Version": "2.0"}
        )
        match = router.route(request)
        
        assert match is not None
        assert match.backend == "v2_backend"
    
    def test_header_regex(self):
        router = HeaderBasedRouter()
        router.add_route(Route(
            id="beta",
            pattern="",
            backend="beta_backend",
            headers={"X-API-Version": "~2\\..*"}  # Regex pattern
        ))
        
        request = RoutingRequest(
            path="/api",
            headers={"X-API-Version": "2.1"}
        )
        match = router.route(request)
        
        assert match is not None


class TestModelBasedRouter:
    """Tests for model-based router."""
    
    def test_model_routing(self):
        router = ModelBasedRouter()
        router.register_model("llama-7b", "gpu_cluster_1")
        router.register_model("mistral", "gpu_cluster_2")
        
        request = RoutingRequest(
            path="/generate",
            model_id="llama-7b"
        )
        match = router.route(request)
        
        assert match is not None
        assert match.backend == "gpu_cluster_1"
    
    def test_default_backend(self):
        router = ModelBasedRouter()
        router.set_default_backend("default_cluster")
        
        request = RoutingRequest(path="/generate", model_id="unknown_model")
        match = router.route(request)
        
        assert match is not None
        assert match.backend == "default_cluster"


class TestCompositeRouter:
    """Tests for composite router."""
    
    def test_multiple_routers(self):
        composite = CompositeRouter()
        
        path_router = PathBasedRouter()
        path_router.add_route(Route(id="api", pattern="/api/.*", backend="api"))
        
        model_router = ModelBasedRouter()
        model_router.register_model("llama", "llama_backend")
        
        composite.add_router(path_router, priority=10)
        composite.add_router(model_router, priority=5)
        
        # Path router should win due to higher priority
        match = composite.route(RoutingRequest(path="/api/test", model_id="llama"))
        assert match is not None


class TestRoutingMiddleware:
    """Tests for routing middleware."""
    
    def test_stats_collection(self):
        router = PathBasedRouter()
        router.add_route(Route(id="test", pattern="/test", backend="test"))
        
        middleware = RoutingMiddleware(router)
        
        middleware.route(RoutingRequest(path="/test"))
        middleware.route(RoutingRequest(path="/test"))
        middleware.route(RoutingRequest(path="/unknown"))
        
        stats = middleware.get_stats()
        assert stats.total_requests == 3
        assert stats.matched_requests == 2
        assert stats.unmatched_requests == 1


# ============= API Gateway Tests =============

class TestTokenBucketRateLimiter:
    """Tests for token bucket rate limiter."""
    
    def test_allows_under_limit(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=5)
        limiter = TokenBucketRateLimiter(config)
        
        for _ in range(5):
            result = limiter.allow("client1")
            assert result.allowed
    
    def test_rejects_over_limit(self):
        config = RateLimitConfig(requests_per_second=1, burst_size=2)
        limiter = TokenBucketRateLimiter(config)
        
        # Use up burst
        limiter.allow("client1")
        limiter.allow("client1")
        
        # Third should be rejected
        result = limiter.allow("client1")
        assert not result.allowed
        assert result.retry_after is not None
    
    def test_tokens_refill(self):
        config = RateLimitConfig(requests_per_second=100, burst_size=1)
        limiter = TokenBucketRateLimiter(config)
        
        limiter.allow("client1")
        time.sleep(0.02)  # Wait for refill
        
        result = limiter.allow("client1")
        assert result.allowed


class TestSlidingWindowRateLimiter:
    """Tests for sliding window rate limiter."""
    
    def test_allows_under_limit(self):
        config = RateLimitConfig(requests_per_second=10)
        limiter = SlidingWindowRateLimiter(config)
        
        for _ in range(5):
            result = limiter.allow("client1")
            assert result.allowed


class TestAPIGateway:
    """Tests for API Gateway."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        config = GatewayConfig(
            rate_limit=RateLimitConfig(requests_per_second=1, burst_size=2)
        )
        # Disable auth for this test
        config.auth.enabled = False
        gateway = APIGateway(config)
        
        async def handler(req):
            return GatewayResponse(status_code=200, body={"ok": True})
        
        # First two should pass
        req = GatewayRequest(request_id="1", path="/test", method="GET", headers={}, client_ip="1.2.3.4")
        resp = await gateway.handle(req, handler)
        assert resp.status_code == 200
        
        resp = await gateway.handle(req, handler)
        assert resp.status_code == 200
        
        # Third should be rate limited
        resp = await gateway.handle(req, handler)
        assert resp.status_code == 429
    
    @pytest.mark.asyncio
    async def test_authentication(self):
        config = GatewayConfig()
        config.auth.enabled = True
        config.auth.allow_anonymous = False
        
        gateway = APIGateway(config)
        
        # Add auth provider
        provider = APIKeyAuthProvider()
        provider.register_key("valid_key", {"name": "test_user"})
        gateway.add_auth_provider(provider)
        
        async def handler(req):
            return GatewayResponse(status_code=200, body={"ok": True})
        
        # Without auth should fail
        req = GatewayRequest(
            request_id="1", path="/test", method="GET",
            headers={}, client_ip="1.2.3.4"
        )
        resp = await gateway.handle(req, handler)
        assert resp.status_code == 401
        
        # With valid auth should pass
        req = GatewayRequest(
            request_id="2", path="/test", method="GET",
            headers={"X-API-Key": "valid_key"}, client_ip="1.2.3.4"
        )
        resp = await gateway.handle(req, handler)
        assert resp.status_code == 200
    
    @pytest.mark.asyncio
    async def test_caching(self):
        config = GatewayConfig(enable_caching=True, cache_ttl=60.0)
        # Disable auth for this test
        config.auth.enabled = False
        gateway = APIGateway(config)
        
        call_count = 0
        
        async def handler(req):
            nonlocal call_count
            call_count += 1
            return GatewayResponse(status_code=200, body={"count": call_count})
        
        req = GatewayRequest(
            request_id="1", path="/test", method="GET",
            headers={}, client_ip="1.2.3.4"
        )
        
        # First call
        resp1 = await gateway.handle(req, handler)
        assert resp1.body["count"] == 1
        
        # Second call should hit cache
        resp2 = await gateway.handle(req, handler)
        assert resp2.body["count"] == 1  # Same as cached
        
        assert gateway.stats.cache_hits == 1


# ============= gRPC Server Tests =============

class TestGRPCServer:
    """Tests for gRPC server."""
    
    @pytest.mark.asyncio
    async def test_server_start_stop(self):
        server = GRPCServer()
        await server.start()
        assert server.running
        
        await server.stop()
        assert not server.running
    
    @pytest.mark.asyncio
    async def test_inference_service(self):
        server = GRPCServer()
        servicer = DefaultInferenceServicer()
        server.register_inference_service(servicer)
        
        request = GRPCRequest(
            method="inference.Generate",
            data=GenerateRequest(prompt="Hello, world!", max_tokens=50)
        )
        
        response = await server.handle_request(request)
        
        assert response.status == GRPCStatusCode.OK
        assert response.data is not None
        assert response.data.request_id is not None
    
    @pytest.mark.asyncio
    async def test_health_service(self):
        server = GRPCServer()
        health = DefaultHealthServicer()
        health.set_service_status("inference", "SERVING")
        server.register_health_service(health)
        
        request = GRPCRequest(
            method="grpc.health.v1.Health/Check",
            data=HealthCheckRequest()
        )
        
        response = await server.handle_request(request)
        
        assert response.status == GRPCStatusCode.OK
        assert response.data.status == "SERVING"
    
    @pytest.mark.asyncio
    async def test_unimplemented_method(self):
        server = GRPCServer()
        
        request = GRPCRequest(
            method="unknown.Method",
            data={}
        )
        
        response = await server.handle_request(request)
        
        assert response.status == GRPCStatusCode.UNIMPLEMENTED


class TestGRPCServerStats:
    """Tests for gRPC server statistics."""
    
    @pytest.mark.asyncio
    async def test_stats_collection(self):
        server = GRPCServer()
        servicer = DefaultInferenceServicer()
        server.register_inference_service(servicer)
        
        # Make some requests
        for i in range(5):
            request = GRPCRequest(
                method="inference.Generate",
                data=GenerateRequest(prompt=f"Test {i}")
            )
            await server.handle_request(request)
        
        assert server.stats.total_requests == 5
        assert server.stats.successful_requests == 5
        assert server.stats.avg_latency_ms > 0


# ============= Integration Tests =============

class TestServingLayerIntegration:
    """Integration tests for the serving layer."""
    
    @pytest.mark.asyncio
    async def test_load_balancer_with_gateway(self):
        """Test load balancer integrated with API gateway."""
        # Create load balancer with nodes
        balancer = RoundRobinBalancer()
        balancer.add_node(BackendNode(id="node1", host="localhost", port=8080))
        balancer.add_node(BackendNode(id="node2", host="localhost", port=8081))
        
        # Create gateway with auth disabled and caching disabled for testing
        config = GatewayConfig()
        config.auth.enabled = False
        config.enable_caching = False  # Disable caching so each request goes through
        gateway = APIGateway(config)
        
        async def handler(req):
            node = balancer.select_node()
            return GatewayResponse(
                status_code=200,
                body={"node": node.id if node else None}
            )
        
        # Make requests
        nodes_used = set()
        for i in range(4):
            req = GatewayRequest(
                request_id=str(i), path="/test", method="GET",
                headers={}, client_ip=f"1.2.3.{i}"
            )
            resp = await gateway.handle(req, handler)
            if resp.body.get("node"):
                nodes_used.add(resp.body["node"])
        
        # Both nodes should be used
        assert len(nodes_used) == 2
    
    @pytest.mark.asyncio
    async def test_router_with_grpc_server(self):
        """Test router directing to gRPC services."""
        router = ModelBasedRouter()
        router.register_model("llama", "llama_server")
        router.register_model("mistral", "mistral_server")
        
        servers = {
            "llama_server": GRPCServer(),
            "mistral_server": GRPCServer(),
        }
        
        for server in servers.values():
            server.register_inference_service(DefaultInferenceServicer())
        
        # Route request
        request = RoutingRequest(path="/generate", model_id="llama")
        match = router.route(request)
        
        assert match is not None
        assert match.backend == "llama_server"
        
        # Handle with matched server
        server = servers[match.backend]
        grpc_req = GRPCRequest(
            method="inference.Generate",
            data=GenerateRequest(prompt="Hello")
        )
        response = await server.handle_request(grpc_req)
        
        assert response.status == GRPCStatusCode.OK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
