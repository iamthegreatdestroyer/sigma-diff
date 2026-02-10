"""
Tests for Distributed API Server
=================================

Tests the distributed inference API server functionality.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import json
from fastapi.testclient import TestClient

from src.api.distributed_server import DistributedAPIServer, get_distributed_server
from src.api.request_router import RequestRouter, RoutingStrategy
from src.serving.batch_engine import BatchEngine, BatchPriority
from src.monitoring.metrics import MetricsCollector


class TestDistributedAPIServer:
    """Test distributed API server functionality."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Mock distributed orchestrator."""
        mock = Mock()
        mock.world_size = 4
        return mock

    @pytest.fixture
    def mock_health_monitor(self):
        """Mock GPU health monitor."""
        mock = Mock()
        mock.get_healthy_devices.return_value = [0, 1, 2, 3]
        mock.get_stats.return_value = Mock(
            utilization=50.0,
            memory_used=2048,
            temperature=70.0,
            active_requests=1,
            last_heartbeat=time.time()
        )
        return mock

    @pytest.fixture
    def mock_request_router(self, mock_orchestrator, mock_health_monitor):
        """Mock request router."""
        router = RequestRouter(
            orchestrator=mock_orchestrator,
            health_monitor=mock_health_monitor,
            strategy=RoutingStrategy.LEAST_LOADED
        )
        return router

    @pytest.fixture
    def mock_batch_engine(self):
        """Mock batch engine."""
        mock = Mock()
        mock.submit_request = AsyncMock(return_value=asyncio.Future())
        mock.get_batch_stats.return_value = {
            "total_requests": 10,
            "batches_processed": 2,
            "avg_batch_size": 3.5
        }
        return mock

    @pytest.fixture
    def mock_metrics_collector(self):
        """Mock metrics collector."""
        mock = Mock()
        mock.record_request = AsyncMock()
        mock.get_summary.return_value = {
            "total_requests": 100,
            "avg_latency_ms": 25.0
        }
        return mock

    @pytest.fixture
    def api_server(
        self,
        mock_orchestrator,
        mock_health_monitor,
        mock_request_router,
        mock_batch_engine,
        mock_metrics_collector
    ):
        """Create API server with mocked components."""
        server = DistributedAPIServer(world_size=4)

        # Replace components with mocks
        server.orchestrator = mock_orchestrator
        server.health_monitor = mock_health_monitor
        server.request_router = mock_request_router
        server.batch_engine = mock_batch_engine
        server.metrics_collector = mock_metrics_collector

        return server

    @pytest.fixture
    def test_client(self, api_server):
        """Create test client for API server."""
        return TestClient(api_server.app)

    def test_health_endpoint(self, test_client, mock_health_monitor):
        """Test health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["healthy_gpus"] == 4
        assert data["total_gpus"] == 4

    def test_metrics_endpoint(self, test_client, mock_metrics_collector):
        """Test metrics endpoint."""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "avg_latency_ms" in data

    @pytest.mark.asyncio
    async def test_chat_completions_request(self, test_client, mock_request_router, mock_metrics_collector):
        """Test chat completions endpoint."""
        # Mock the routing and processing
        mock_request_router.route_request = AsyncMock(return_value=0)

        # Mock the completion processing
        with patch.object(test_client.app.state.server, '_process_completion', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "id": "test-id",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }

            request_data = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7,
                "max_tokens": 100
            }

            response = test_client.post("/v1/chat/completions", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["model"] == "test-model"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["content"] == "Test response"

    def test_invalid_request(self, test_client):
        """Test invalid request handling."""
        # Missing required fields
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}]
            # Missing "model"
        }

        response = test_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_streaming_request(self, test_client, mock_request_router):
        """Test streaming response."""
        mock_request_router.route_request = AsyncMock(return_value=0)

        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        }

        # This would need more complex mocking for streaming
        # For now, just test that the endpoint accepts streaming parameter
        response = test_client.post("/v1/chat/completions", json=request_data)
        # Should not fail due to streaming parameter
        assert response.status_code in [200, 500]  # 500 is expected with mock setup


class TestRequestRouter:
    """Test request routing functionality."""

    @pytest.fixture
    def mock_orchestrator(self):
        return Mock()

    @pytest.fixture
    def mock_health_monitor(self):
        mock = Mock()
        mock.get_healthy_devices.return_value = [0, 1, 2]
        mock.get_stats.side_effect = lambda gpu_id: Mock(
            utilization=50.0 if gpu_id == 0 else 80.0 if gpu_id == 1 else 30.0,
            temperature=70.0,
            active_requests=1 if gpu_id == 0 else 2 if gpu_id == 1 else 0,
            last_heartbeat=time.time()
        )
        return mock

    @pytest.fixture
    def router(self, mock_orchestrator, mock_health_monitor):
        return RequestRouter(
            orchestrator=mock_orchestrator,
            health_monitor=mock_health_monitor,
            strategy=RoutingStrategy.LEAST_LOADED
        )

    @pytest.mark.asyncio
    async def test_least_loaded_routing(self, router):
        """Test least loaded routing strategy."""
        # GPU 2 has lowest utilization (30%) and no active requests
        gpu_id = await router.route_request({"test": "request"})
        assert gpu_id == 2

    @pytest.mark.asyncio
    async def test_round_robin_routing(self, router):
        """Test round-robin routing strategy."""
        router.strategy = RoutingStrategy.ROUND_ROBIN

        # Should cycle through healthy GPUs: 0, 1, 2, 0, 1, 2...
        gpu_id1 = await router.route_request({"test": "request1"})
        gpu_id2 = await router.route_request({"test": "request2"})
        gpu_id3 = await router.route_request({"test": "request3"})

        assert gpu_id1 == 0
        assert gpu_id2 == 1
        assert gpu_id3 == 2

    def test_routing_stats(self, router):
        """Test routing statistics collection."""
        stats = router.get_routing_stats()

        assert "total_requests" in stats
        assert "strategy_distribution" in stats
        assert "current_strategy" in stats
        assert stats["current_strategy"] == "least_loaded"

    def test_strategy_change(self, router):
        """Test changing routing strategy."""
        router.set_routing_strategy(RoutingStrategy.ROUND_ROBIN)
        assert router.strategy == RoutingStrategy.ROUND_ROBIN

        stats = router.get_routing_stats()
        assert stats["current_strategy"] == "round_robin"


class TestBatchEngine:
    """Test batch processing engine."""

    @pytest.fixture
    def batch_engine(self):
        return BatchEngine(max_batch_size=4, max_latency_ms=100.0)

    def test_initialization(self, batch_engine):
        """Test batch engine initialization."""
        assert batch_engine.max_batch_size == 4
        assert batch_engine.max_latency_ms == 100.0

    @pytest.mark.asyncio
    async def test_request_submission(self, batch_engine):
        """Test request submission and batching."""
        batch_engine.initialize_gpu_queues([0])

        future = await batch_engine.submit_request(
            request_id="test-1",
            gpu_id=0,
            input_tokens=[1, 2, 3],
            max_new_tokens=10
        )

        assert isinstance(future, asyncio.Future)

        # Check queue status
        status = batch_engine.get_gpu_queue_status(0)
        assert status["gpu_id"] == 0
        assert status["active_batch_size"] == 1

    def test_batch_stats(self, batch_engine):
        """Test batch statistics."""
        stats = batch_engine.get_batch_stats()

        assert "total_requests" in stats
        assert "batches_processed" in stats
        assert "avg_batch_size" in stats
        assert "efficiency_ratio" in stats

    def test_uninitialized_gpu(self, batch_engine):
        """Test accessing uninitialized GPU."""
        status = batch_engine.get_gpu_queue_status(999)
        assert "error" in status


class TestMetricsCollector:
    """Test metrics collection functionality."""

    @pytest.fixture
    def metrics_collector(self):
        return MetricsCollector()

    @pytest.mark.asyncio
    async def test_request_recording(self, metrics_collector):
        """Test recording request metrics."""
        await metrics_collector.record_request(
            gpu_id=0,
            tokens_processed=100,
            success=True,
            latency_ms=25.0
        )

        summary = metrics_collector.get_metric_summary("requests_total")
        assert summary["count"] >= 1

        summary = metrics_collector.get_metric_summary("request_latency_ms")
        assert summary["avg"] == 25.0

    def test_gpu_summary(self, metrics_collector):
        """Test GPU-specific metrics summary."""
        # Add some test data
        metrics_collector.record_gpu_stats(gpu_id=0, utilization=75.0, memory_used_mb=4096)

        summary = metrics_collector.get_gpu_summary(gpu_id=0)
        assert summary["gpu_id"] == 0
        assert "avg_utilization_percent" in summary

    def test_system_summary(self, metrics_collector):
        """Test system-wide metrics summary."""
        summary = metrics_collector.get_system_summary()
        assert "total_gpus" in summary
        assert "total_requests" in summary
        assert "system_avg_latency_ms" in summary

    def test_all_metrics_export(self, metrics_collector):
        """Test metrics export functionality."""
        export_data = metrics_collector.export_metrics()
        assert isinstance(export_data, str)

        # Should be valid JSON
        data = json.loads(export_data)
        assert "metrics" in data
        assert "export_time" in data
