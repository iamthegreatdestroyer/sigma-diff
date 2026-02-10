"""
Production Hardening Test Suite
================================

Comprehensive test suite for production hardening components,
validating error handling, monitoring, benchmarking, and security
functionality for production deployment.
"""

import torch
import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from .production_hardening import (
    ProductionHardeningSuite, ProductionReadinessLevel
)
from .error_handling import ProductionErrorHandler, ErrorSeverity, ErrorCategory
from .monitoring import MetricsCollector, DistributedTracer, HealthMonitor
from .benchmarking import PerformanceBenchmarker, BenchmarkType
from .security import (
    EncryptionManager, AccessControlManager, SecurityScanner,
    AuditLogger, SecurityHardeningManager
)


class TestProductionHardeningSuite:
    """Test suite for ProductionHardeningSuite."""

    @pytest.fixture
    def hardening_suite(self):
        """Create a test hardening suite."""
        return ProductionHardeningSuite(world_size=1, rank=0, enable_distributed=False)

    def test_initialization(self, hardening_suite):
        """Test suite initialization."""
        assert hardening_suite.world_size == 1
        assert hardening_suite.rank == 0
        assert not hardening_suite.production_mode
        assert hardening_suite.readiness_level == ProductionReadinessLevel.DEVELOPMENT

    def test_start_production_mode(self, hardening_suite):
        """Test starting production mode."""
        hardening_suite.start_production_mode()

        assert hardening_suite.production_mode
        assert hardening_suite.running

        # Cleanup
        hardening_suite.stop_production_mode()

    def test_production_health_assessment(self, hardening_suite):
        """Test production health assessment."""
        health_status = hardening_suite.get_production_health()

        assert "overall_health" in health_status
        assert "error_handler_status" in health_status
        assert "monitoring_status" in health_status
        assert "security_status" in health_status
        assert "readiness_level" in health_status

    def test_inference_with_hardening(self, hardening_suite):
        """Test inference execution with hardening."""
        def mock_inference(data):
            return {"result": "success", "input": data}

        result = hardening_suite.run_inference_with_hardening(
            mock_inference,
            input_data={"test": "data"},
            user_id="test_user"
        )

        assert "result" in result
        assert "metadata" in result
        assert result["metadata"]["user_id"] == "test_user"

    def test_comprehensive_benchmark(self, hardening_suite):
        """Test comprehensive benchmarking."""
        results = hardening_suite.run_comprehensive_benchmark()

        # Should have some results even if mock
        assert isinstance(results, dict)
        assert "latency" in results or "error" in results

    def test_production_report_generation(self, hardening_suite):
        """Test production report generation."""
        report = hardening_suite.generate_production_report()

        assert "timestamp" in report
        assert "readiness_level" in report
        assert "health_status" in report
        assert "performance_summary" in report
        assert "security_assessment" in report

    def test_configuration_export(self, hardening_suite):
        """Test configuration export."""
        config = hardening_suite.export_configuration()

        config_data = json.loads(config)
        assert "production_hardening" in config_data
        assert "error_handling" in config_data
        assert "monitoring" in config_data
        assert "security" in config_data

    def test_deployment_validation(self, hardening_suite):
        """Test deployment validation."""
        validation = hardening_suite.validate_production_deployment()

        assert "checks" in validation
        assert "passed" in validation
        assert "critical_issues" in validation
        assert "warnings" in validation


class TestProductionErrorHandler:
    """Test suite for ProductionErrorHandler."""

    @pytest.fixture
    def error_handler(self):
        """Create a test error handler."""
        return ProductionErrorHandler()

    def test_error_classification(self, error_handler):
        """Test error classification."""
        # Test memory error
        oom_error = RuntimeError("CUDA out of memory")
        error_context = error_handler._classify_error(
            oom_error, "inference", "forward", {}
        )

        assert error_context.category == ErrorCategory.MEMORY
        assert error_context.severity == ErrorSeverity.HIGH

    def test_circuit_breaker(self, error_handler):
        """Test circuit breaker functionality."""
        # Initially should allow
        assert error_handler._check_circuit_breaker(
            error_handler._classify_error(
                RuntimeError("test"), "test", "op", {}
            )
        )

        # After failures, should trip
        for _ in range(error_handler.circuit_breaker_threshold + 1):
            error_handler._record_circuit_failure(
                error_handler._classify_error(
                    RuntimeError("test"), "test", "op", {}
                )
            )

        assert not error_handler._check_circuit_breaker(
            error_handler._classify_error(
                RuntimeError("test"), "test", "op", {}
            )
        )

    def test_error_recovery(self, error_handler):
        """Test error recovery mechanisms."""
        # Test memory error recovery
        memory_error = RuntimeError("CUDA out of memory")
        error_context = error_handler._classify_error(
            memory_error, "inference", "forward", {}
        )

        recovered = error_handler._recover_memory_error(error_context)
        assert recovered  # Should succeed

    def test_error_stats(self, error_handler):
        """Test error statistics collection."""
        # Generate some errors
        for i in range(3):
            error_handler.handle_error(
                RuntimeError(f"test error {i}"),
                "test_component",
                "test_operation"
            )

        stats = error_handler.get_error_stats()
        assert stats["total_errors"] >= 3
        assert "test_component.test_operation" in stats["error_counts"]


class TestMetricsCollector:
    """Test suite for MetricsCollector."""

    @pytest.fixture
    def metrics_collector(self):
        """Create a test metrics collector."""
        collector = MetricsCollector(collection_interval=0.1, retention_period=60)
        return collector

    def test_counter_operations(self, metrics_collector):
        """Test counter operations."""
        metrics_collector.increment_counter("test_counter")
        metrics_collector.increment_counter("test_counter", 5)

        summary = metrics_collector.get_metrics_summary()
        assert summary["counters"]["test_counter"] == 6

    def test_gauge_operations(self, metrics_collector):
        """Test gauge operations."""
        metrics_collector.set_gauge("test_gauge", 42.0)
        metrics_collector.set_gauge("test_gauge", 24.0)

        summary = metrics_collector.get_metrics_summary()
        assert summary["gauges"]["test_gauge"] == 24.0

    def test_histogram_operations(self, metrics_collector):
        """Test histogram operations."""
        for value in [1.0, 2.0, 3.0, 2.0, 1.0]:
            metrics_collector.observe_histogram("test_histogram", value)

        summary = metrics_collector.get_metrics_summary()
        assert summary["histograms"]["test_histogram"]["count"] == 5
        assert summary["histograms"]["test_histogram"]["avg"] == 1.8

    def test_alert_thresholds(self, metrics_collector):
        """Test alert threshold functionality."""
        metrics_collector.set_alert_threshold(
            "test_gauge", "high_value", 10.0, "gt", "Value too high"
        )

        metrics_collector.set_gauge("test_gauge", 15.0)

        # Run collection to trigger alerts
        metrics_collector._check_alerts()

        assert "test_gauge:high_value" in metrics_collector.active_alerts

    def test_prometheus_export(self, metrics_collector):
        """Test Prometheus format export."""
        metrics_collector.set_gauge("test_metric", 42.0)

        prometheus_output = metrics_collector.export_metrics("prometheus")
        assert "test_metric 42.0" in prometheus_output


class TestDistributedTracer:
    """Test suite for DistributedTracer."""

    @pytest.fixture
    def tracer(self):
        """Create a test tracer."""
        return DistributedTracer("test_service")

    def test_span_creation(self, tracer):
        """Test span creation and management."""
        span_id = tracer.start_span("test_operation", tags={"key": "value"})

        assert span_id in tracer.active_traces
        span = tracer.active_traces[span_id]
        assert span.operation == "test_operation"
        assert span.tags["key"] == "value"

    def test_span_completion(self, tracer):
        """Test span completion."""
        span_id = tracer.start_span("test_operation")
        time.sleep(0.01)  # Small delay
        tracer.end_span(span_id, {"result": "success"})

        assert span_id not in tracer.active_traces
        assert len(tracer.completed_traces) == 1

        span = tracer.completed_traces[0]
        assert span.duration is not None
        assert span.duration > 0

    def test_performance_stats(self, tracer):
        """Test performance statistics calculation."""
        # Create and complete spans
        for i in range(5):
            span_id = tracer.start_span("test_op")
            time.sleep(0.001)
            tracer.end_span(span_id)

        stats = tracer.get_performance_stats()
        assert "test_op" in stats
        assert stats["test_op"]["count"] == 5
        assert stats["test_op"]["avg_duration"] > 0

    def test_trace_correlation(self, tracer):
        """Test trace correlation across spans."""
        trace_id = "test-trace-123"

        span1_id = tracer.start_span("operation1", trace_id=trace_id)
        span2_id = tracer.start_span("operation2", trace_id=trace_id, parent_span_id=span1_id)

        tracer.end_span(span1_id)
        tracer.end_span(span2_id)

        trace_spans = tracer.get_trace(trace_id)
        assert len(trace_spans) == 2

        # Check parent-child relationship
        child_span = next(span for span in trace_spans if span.span_id == span2_id)
        assert child_span.parent_span_id == span1_id


class TestPerformanceBenchmarker:
    """Test suite for PerformanceBenchmarker."""

    @pytest.fixture
    def benchmarker(self):
        """Create a test benchmarker."""
        return PerformanceBenchmarker(
            warmup_iterations=1,
            benchmark_iterations=2,
            enable_distributed=False
        )

    def test_latency_benchmark(self, benchmarker):
        """Test latency benchmarking."""
        def mock_inference(data):
            time.sleep(0.001)  # 1ms delay
            return "result"

        result = benchmarker.run_latency_benchmark(mock_inference)

        assert result.benchmark_type == BenchmarkType.LATENCY
        assert result.value > 0  # Should have some latency
        assert result.unit == "seconds"
        assert "p50_latency" in result.metadata

    def test_throughput_benchmark(self, benchmarker):
        """Test throughput benchmarking."""
        def mock_inference(data):
            return "result"

        result = benchmarker.run_throughput_benchmark(mock_inference)

        assert result.benchmark_type == BenchmarkType.THROUGHPUT
        assert result.value > 0  # Should have some throughput
        assert result.unit == "requests/second"

    def test_memory_benchmark(self, benchmarker):
        """Test memory benchmarking."""
        def mock_inference(data):
            # Allocate some memory
            tensor = torch.randn(10, 10)
            return tensor

        results = benchmarker.run_memory_benchmark(mock_inference)

        assert len(results) > 0
        for result in results:
            assert result.benchmark_type == BenchmarkType.MEMORY
            assert result.unit == "MB"

    def test_bottleneck_analysis(self, benchmarker):
        """Test bottleneck analysis."""
        # Create mock results
        results = [
            Mock(benchmark_type=BenchmarkType.LATENCY, value=2.0),  # High latency
            Mock(benchmark_type=BenchmarkType.THROUGHPUT, value=5.0)  # Low throughput
        ]

        analysis = benchmarker.analyze_bottlenecks(results)

        assert "bottlenecks" in analysis
        assert "recommendations" in analysis
        assert len(analysis["bottlenecks"]) > 0


class TestEncryptionManager:
    """Test suite for EncryptionManager."""

    @pytest.fixture
    def encryption_manager(self):
        """Create a test encryption manager."""
        return EncryptionManager(key_rotation_days=1)  # Short for testing

    def test_data_encryption_decryption(self, encryption_manager):
        """Test data encryption and decryption."""
        test_data = "sensitive information"

        encrypted = encryption_manager.encrypt_data(test_data)
        decrypted = encryption_manager.decrypt_data(encrypted)

        assert decrypted == test_data
        assert encrypted != test_data  # Should be encrypted

    def test_key_rotation(self, encryption_manager):
        """Test key rotation functionality."""
        old_key = encryption_manager.current_key

        encryption_manager.rotate_key()

        assert encryption_manager.current_key != old_key
        assert encryption_manager.key_created is not None

    def test_key_rotation_trigger(self, encryption_manager):
        """Test key rotation trigger logic."""
        # Should not rotate immediately
        assert not encryption_manager.should_rotate_key()

        # Mock old creation date
        encryption_manager.key_created = datetime.now() - timedelta(days=2)

        # Should trigger rotation
        assert encryption_manager.should_rotate_key()


class TestAccessControlManager:
    """Test suite for AccessControlManager."""

    @pytest.fixture
    def access_manager(self):
        """Create a test access control manager."""
        return AccessControlManager()

    def test_user_creation(self, access_manager):
        """Test user creation and role assignment."""
        access_manager.add_user("testuser", "password123", ["developer"])

        assert "testuser" in access_manager.users
        assert "developer" in access_manager.user_roles["testuser"]

    def test_authentication(self, access_manager):
        """Test user authentication."""
        access_manager.add_user("testuser", "password123", ["developer"])

        session_token = access_manager.authenticate_user("testuser", "password123")
        assert session_token is not None

        # Wrong password should fail
        bad_token = access_manager.authenticate_user("testuser", "wrongpass")
        assert bad_token is None

    def test_authorization(self, access_manager):
        """Test request authorization."""
        access_manager.add_user("testuser", "password123", ["developer"])

        session_token = access_manager.authenticate_user("testuser", "password123")

        # Should allow developer permissions
        assert access_manager.authorize_request(session_token, "inference", "read")

        # Should deny admin permissions
        assert not access_manager.authorize_request(session_token, "system", "admin")

    def test_policy_evaluation(self, access_manager):
        """Test policy-based authorization."""
        from .security import AccessPolicy

        policy = AccessPolicy(
            resource="inference",
            action="write",
            conditions={"user_id": "admin"},
            effect="allow"
        )
        access_manager.add_policy(policy)

        access_manager.add_user("admin", "password123", ["admin"])
        session_token = access_manager.authenticate_user("admin", "password123")

        # Admin should be allowed due to policy
        assert access_manager.authorize_request(session_token, "inference", "write")


class TestSecurityScanner:
    """Test suite for SecurityScanner."""

    @pytest.fixture
    def security_scanner(self):
        """Create a test security scanner."""
        return SecurityScanner()

    def test_vulnerability_scanning(self, security_scanner):
        """Test vulnerability scanning."""
        vulnerabilities = security_scanner.scan_for_vulnerabilities("test_component")

        assert isinstance(vulnerabilities, list)
        # Should find some mock vulnerabilities
        assert len(vulnerabilities) > 0

    def test_compliance_checking(self, security_scanner):
        """Test compliance checking."""
        result = security_scanner.check_compliance("soc2")

        assert "framework" in result
        assert "checks" in result
        assert "overall_compliant" in result

    @pytest.mark.parametrize("framework", ["soc2", "gdpr", "hipaa"])
    def test_all_compliance_frameworks(self, security_scanner, framework):
        """Test all compliance frameworks."""
        result = security_scanner.check_compliance(framework)

        assert result["framework"] == framework
        assert "checks" in result
        assert "overall_compliant" in result


class TestAuditLogger:
    """Test suite for AuditLogger."""

    @pytest.fixture
    def audit_logger(self):
        """Create a test audit logger."""
        return AuditLogger(log_retention_days=1)  # Short for testing

    def test_event_logging(self, audit_logger):
        """Test security event logging."""
        from .security import SecurityEvent

        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type="authentication",
            severity="info",
            user_id="testuser",
            resource="system",
            action="login",
            success=True
        )

        audit_logger.log_event(event)

        # Should have the event
        assert len(audit_logger.audit_events) == 1
        assert audit_logger.audit_events[0].event_type == "authentication"

    def test_log_integrity(self, audit_logger):
        """Test audit log integrity verification."""
        from .security import SecurityEvent

        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type="test",
            severity="info",
            user_id="test",
            resource="test",
            action="test",
            success=True
        )

        audit_logger.log_event(event)

        # Should verify integrity
        assert audit_logger.verify_log_integrity()

    def test_event_querying(self, audit_logger):
        """Test audit event querying."""
        from .security import SecurityEvent

        # Add multiple events
        for i in range(3):
            event = SecurityEvent(
                timestamp=datetime.now(),
                event_type="test",
                severity="info",
                user_id=f"user{i}",
                resource="system",
                action="action",
                success=True
            )
            audit_logger.log_event(event)

        # Query events
        events = audit_logger.query_events(event_type="test")
        assert len(events) == 3

        # Query by user
        user_events = audit_logger.query_events(user_id="user1")
        assert len(user_events) == 1


class TestSecurityHardeningManager:
    """Test suite for SecurityHardeningManager."""

    @pytest.fixture
    def hardening_manager(self):
        """Create a test security hardening manager."""
        return SecurityHardeningManager()

    def test_hardening_checklist(self, hardening_manager):
        """Test security hardening checklist execution."""
        results = hardening_manager.run_hardening_checklist()

        assert "checks" in results
        assert "overall_score" in results
        assert "recommendations" in results

        # Should have expected checks
        expected_checks = ["network_security", "data_encryption", "access_controls",
                          "logging_monitoring", "patch_management"]
        for check in expected_checks:
            assert check in results["checks"]

    def test_incident_response(self, hardening_manager):
        """Test incident response activation."""
        response = hardening_manager.activate_incident_response(
            "security_breach",
            {"affected_systems": ["inference", "cache"], "severity": "high"}
        )

        assert "incident_type" in response
        assert "response_actions" in response
        assert "escalation_contacts" in response


# Integration Tests
class TestProductionHardeningIntegration:
    """Integration tests for production hardening components."""

    @pytest.fixture
    def full_suite(self):
        """Create a full production hardening suite."""
        return ProductionHardeningSuite(
            world_size=1,
            rank=0,
            enable_distributed=False,
            security_level="high"
        )

    def test_end_to_end_inference(self, full_suite):
        """Test end-to-end inference with all hardening components."""
        def mock_inference(data):
            return {"prediction": "positive", "confidence": 0.95}

        # Start production mode
        full_suite.start_production_mode()

        try:
            # Run inference with hardening
            result = full_suite.run_inference_with_hardening(
                mock_inference,
                input_data={"text": "test input"},
                user_id="test_user"
            )

            assert "result" in result
            assert "metadata" in result
            assert result["metadata"]["user_id"] == "test_user"

            # Check that monitoring captured the request
            metrics = full_suite.metrics_collector.get_metrics_summary()
            assert metrics["counters"].get("inference_requests_total", 0) > 0

        finally:
            full_suite.stop_production_mode()

    def test_comprehensive_health_check(self, full_suite):
        """Test comprehensive health check across all components."""
        full_suite.start_production_mode()

        try:
            health = full_suite.get_production_health()

            # Should have all component statuses
            required_keys = [
                "overall_health", "error_handler_status", "monitoring_status",
                "benchmarking_status", "security_status", "readiness_level",
                "issues", "recommendations"
            ]

            for key in required_keys:
                assert key in health

        finally:
            full_suite.stop_production_mode()

    def test_production_readiness_assessment(self, full_suite):
        """Test production readiness level assessment."""
        # Initially should be development
        assert full_suite.readiness_level == ProductionReadinessLevel.DEVELOPMENT

        # After starting production mode, should assess
        full_suite.start_production_mode()

        try:
            # Should have assessed readiness
            assert full_suite.readiness_level in [
                ProductionReadinessLevel.DEVELOPMENT,
                ProductionReadinessLevel.STAGING,
                ProductionReadinessLevel.PRODUCTION_READY,
                ProductionReadinessLevel.PRODUCTION_OPTIMIZED
            ]

        finally:
            full_suite.stop_production_mode()

    def test_security_integration(self, full_suite):
        """Test security component integration."""
        full_suite.start_production_mode()

        try:
            # Test encryption
            test_data = "sensitive production data"
            encrypted = full_suite.encryption_manager.encrypt_data(test_data)
            decrypted = full_suite.encryption_manager.decrypt_data(encrypted)
            assert decrypted == test_data

            # Test access control
            full_suite.access_control.add_user("testuser", "password", ["user"])
            session = full_suite.access_control.authenticate_user("testuser", "password")
            assert session is not None

            # Test security scanning
            vulnerabilities = full_suite.security_scanner.scan_for_vulnerabilities("inference")
            assert isinstance(vulnerabilities, list)

        finally:
            full_suite.stop_production_mode()


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running production hardening smoke tests...")

    suite = ProductionHardeningSuite(world_size=1, rank=0, enable_distributed=False)

    # Test basic functionality
    health = suite.get_production_health()
    print(f"Health status: {health.overall_health}")

    # Test inference
    def mock_inference(data):
        return {"result": "success"}

    result = suite.run_inference_with_hardening(mock_inference, {"test": "data"})
    print(f"Inference result: {result['result']}")

    # Test report generation
    report = suite.generate_production_report()
    print(f"Readiness level: {report['readiness_level']}")

    print("Smoke tests completed successfully!")
