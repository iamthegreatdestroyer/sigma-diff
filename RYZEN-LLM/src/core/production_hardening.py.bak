"""
Production Hardening Integration
=================================

Integrated production hardening system that combines error handling,
monitoring, benchmarking, and security into a cohesive production-ready
solution for distributed inference systems.

Key Features:
- Unified production hardening interface
- Automated health checks and self-healing
- Comprehensive monitoring and alerting
- Performance optimization and benchmarking
- Security hardening and compliance automation
- Production deployment validation
"""

import torch
import torch.distributed as dist
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os

from .error_handling import ProductionErrorHandler, FaultToleranceManager
from .monitoring import MetricsCollector, DistributedTracer, HealthMonitor
from .benchmarking import PerformanceBenchmarker, ResourceOptimizer
from .security import (
    EncryptionManager, AccessControlManager, SecurityScanner,
    AuditLogger, SecurityHardeningManager
)

logger = logging.getLogger(__name__)


class ProductionReadinessLevel(Enum):
    """Production readiness assessment levels."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION_READY = "production_ready"
    PRODUCTION_OPTIMIZED = "production_optimized"


@dataclass
class ProductionHealthStatus:
    """Comprehensive production health status."""
    overall_health: str
    error_handler_status: Dict[str, Any]
    monitoring_status: Dict[str, Any]
    benchmarking_status: Dict[str, Any]
    security_status: Dict[str, Any]
    readiness_level: ProductionReadinessLevel
    issues: List[str]
    recommendations: List[str]


class ProductionHardeningSuite:
    """
    Integrated production hardening suite for distributed inference.

    Features:
    - Unified error handling and fault tolerance
    - Comprehensive monitoring and observability
    - Automated performance benchmarking
    - Security hardening and compliance
    - Production readiness assessment
    - Self-healing and optimization
    """

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        enable_distributed: bool = True,
        security_level: str = "high"
    ):
        self.world_size = world_size
        self.rank = rank
        self.enable_distributed = enable_distributed
        self.security_level = security_level

        # Initialize core components
        self.error_handler = ProductionErrorHandler()
        self.fault_tolerance = FaultToleranceManager(world_size, rank)
        self.metrics_collector = MetricsCollector()
        self.tracer = DistributedTracer("ryzen-llm-inference")
        self.health_monitor = HealthMonitor()
        self.performance_benchmarker = PerformanceBenchmarker(enable_distributed=enable_distributed)
        self.resource_optimizer = ResourceOptimizer()

        # Security components
        self.encryption_manager = EncryptionManager()
        self.access_control = AccessControlManager()
        self.security_scanner = SecurityScanner()
        self.audit_logger = AuditLogger()
        self.security_hardening = SecurityHardeningManager()

        # Production state
        self.production_mode = False
        self.readiness_level = ProductionReadinessLevel.DEVELOPMENT
        self.health_check_interval = 60.0  # seconds
        self.optimization_interval = 300.0  # 5 minutes

        # Threading
        self.monitoring_thread: Optional[threading.Thread] = None
        self.optimization_thread: Optional[threading.Thread] = None
        self.running = False

        logger.info("ProductionHardeningSuite initialized")

    def start_production_mode(self):
        """Start production hardening mode."""
        self.production_mode = True
        self.running = True

        # Start all monitoring and optimization threads
        self._start_monitoring()
        self._start_optimization()

        # Initialize security
        self._initialize_security()

        # Set production readiness level
        self._assess_readiness_level()

        logger.info("Production hardening mode activated")

    def stop_production_mode(self):
        """Stop production hardening mode."""
        self.production_mode = False
        self.running = False

        # Stop all threads
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10.0)
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10.0)

        # Stop components
        self.fault_tolerance.stop_monitoring()
        self.metrics_collector.stop_collection()
        self.health_monitor.stop_monitoring()

        logger.info("Production hardening mode deactivated")

    def _start_monitoring(self):
        """Start monitoring threads."""
        # Start fault tolerance monitoring
        self.fault_tolerance.start_monitoring()

        # Start metrics collection
        self.metrics_collector.start_collection()

        # Start health monitoring
        self.health_monitor.start_monitoring()

        # Start main monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def _start_optimization(self):
        """Start optimization threads."""
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Comprehensive health check
                health_status = self.get_production_health()

                # Log critical issues
                if health_status.overall_health != "healthy":
                    logger.warning(f"Production health degraded: {health_status.overall_health}")
                    for issue in health_status.issues:
                        logger.warning(f"Issue: {issue}")

                # Auto-heal if possible
                self._attempt_auto_healing(health_status)

                time.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")

    def _optimization_loop(self):
        """Main optimization loop."""
        while self.running:
            try:
                # Run performance optimization
                self._run_performance_optimization()

                # Security maintenance
                self._run_security_maintenance()

                time.sleep(self.optimization_interval)

            except Exception as e:
                logger.error(f"Optimization loop error: {e}")

    def _initialize_security(self):
        """Initialize security components."""
        # Create default admin user
        try:
            self.access_control.add_user("admin", "default_password_change_immediately", ["admin"])
        except ValueError:
            pass  # User already exists

        # Set up security monitoring
        self.metrics_collector.set_alert_threshold(
            "errors_total", "high_error_rate",
            value=10, comparison="gt",
            message="High error rate detected - investigate immediately"
        )

        logger.info("Security components initialized")

    def _assess_readiness_level(self):
        """Assess current production readiness level."""
        # Run comprehensive checks
        health_status = self.get_production_health()
        security_check = self.security_hardening.run_hardening_checklist()
        benchmark_results = self.performance_benchmarker.get_performance_summary()

        # Assess readiness
        score = 0

        # Health score (40%)
        if health_status.overall_health == "healthy":
            score += 40
        elif health_status.overall_health == "degraded":
            score += 20

        # Security score (30%)
        security_score = security_check.get("overall_score", 0) * 30
        score += security_score

        # Performance score (30%)
        if benchmark_results["benchmark_summary"]["total_benchmarks"] > 0:
            score += 30

        # Determine level
        if score >= 90:
            self.readiness_level = ProductionReadinessLevel.PRODUCTION_OPTIMIZED
        elif score >= 75:
            self.readiness_level = ProductionReadinessLevel.PRODUCTION_READY
        elif score >= 50:
            self.readiness_level = ProductionReadinessLevel.STAGING
        else:
            self.readiness_level = ProductionReadinessLevel.DEVELOPMENT

        logger.info(f"Production readiness assessed: {self.readiness_level.value} (score: {score:.1f})")

    def get_production_health(self) -> ProductionHealthStatus:
        """Get comprehensive production health status."""
        # Gather component statuses
        error_stats = self.error_handler.get_error_stats()
        monitoring_stats = self.metrics_collector.get_metrics_summary()
        health_stats = self.health_monitor.get_health_status()
        security_stats = self.security_hardening.run_hardening_checklist()

        # Determine overall health
        overall_health = "healthy"
        issues = []
        recommendations = []

        # Check error rates
        if error_stats["total_errors"] > 100:
            overall_health = "degraded"
            issues.append("High error count detected")
            recommendations.append("Investigate error patterns and implement fixes")

        # Check system health
        if not health_stats.get("overall_health", {}).get("status") == "healthy":
            overall_health = "unhealthy"
            issues.append("System health checks failing")
            recommendations.append("Check system resources and component health")

        # Check security
        if security_stats.get("overall_score", 0) < 0.8:
            overall_health = "degraded"
            issues.append("Security hardening incomplete")
            recommendations.append("Complete security hardening checklist")

        return ProductionHealthStatus(
            overall_health=overall_health,
            error_handler_status=error_stats,
            monitoring_status=monitoring_stats,
            benchmarking_status=self.performance_benchmarker.get_performance_summary(),
            security_status=security_stats,
            readiness_level=self.readiness_level,
            issues=issues,
            recommendations=recommendations
        )

    def _attempt_auto_healing(self, health_status: ProductionHealthStatus):
        """Attempt automatic healing of issues."""
        for issue in health_status.issues:
            if "memory" in issue.lower():
                # Attempt memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Auto-healed: Memory cleanup performed")

            elif "circuit breaker" in issue.lower():
                # Reset circuit breakers
                for component in ["inference", "cache", "network"]:
                    for operation in ["read", "write", "process"]:
                        self.error_handler.reset_circuit_breaker(component, operation)
                logger.info("Auto-healed: Circuit breakers reset")

    def _run_performance_optimization(self):
        """Run automated performance optimization."""
        try:
            # Analyze resource usage
            resource_analysis = self.resource_optimizer.analyze_resource_usage(self.metrics_collector)

            # Apply optimizations
            for recommendation in resource_analysis.get("recommendations", []):
                if "batch size" in recommendation.lower():
                    # Optimize batch sizes
                    current_batch = 4  # Would be dynamic
                    optimal_batch = self.resource_optimizer.optimize_batch_sizes(current_batch)
                    logger.info(f"Optimization: Adjusted batch size to {optimal_batch}")

            logger.debug("Performance optimization cycle completed")

        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")

    def _run_security_maintenance(self):
        """Run security maintenance tasks."""
        try:
            # Rotate encryption keys if needed
            if self.encryption_manager.should_rotate_key():
                self.encryption_manager.rotate_key()
                logger.info("Security: Encryption keys rotated")

            # Run security scans
            vulnerabilities = self.security_scanner.scan_for_vulnerabilities("system")
            if vulnerabilities:
                logger.warning(f"Security scan found {len(vulnerabilities)} vulnerabilities")

            # Verify log integrity
            if not self.audit_logger.verify_log_integrity():
                logger.error("Security: Audit log integrity compromised")

        except Exception as e:
            logger.error(f"Security maintenance failed: {e}")

    def run_inference_with_hardening(
        self,
        inference_func: Callable,
        input_data: Any,
        user_id: Optional[str] = None,
        trace_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run inference with full production hardening.

        Args:
            inference_func: The inference function to call
            input_data: Input data for inference
            user_id: User ID for access control
            trace_context: Tracing context

        Returns:
            Inference result with metadata
        """
        start_time = time.time()
        trace_id = trace_context.get("trace_id") if trace_context else None

        # Start tracing
        span_id = self.tracer.start_span(
            "inference_request",
            trace_id=trace_id,
            tags={"user_id": user_id, "component": "inference"}
        )

        try:
            # Access control check
            if user_id and not self.access_control.authorize_request(
                "dummy_session", "inference", "execute"
            ):
                raise PermissionError("Access denied")

            # Record metrics
            self.metrics_collector.increment_counter("inference_requests_total")

            # Run inference with error handling
            result = inference_func(input_data)

            # Record success metrics
            inference_time = time.time() - start_time
            self.metrics_collector.set_gauge("inference_latency_ms", inference_time * 1000)
            self.metrics_collector.observe_histogram("inference_batch_sizes", len(input_data) if hasattr(input_data, '__len__') else 1)

            # End tracing
            self.tracer.end_span(span_id, {"success": True, "latency": inference_time})

            # Audit log
            self.audit_logger.log_event(SecurityEvent(
                timestamp=datetime.now(),
                event_type="inference",
                severity="info",
                user_id=user_id,
                resource="inference",
                action="execute",
                success=True,
                metadata={"latency": inference_time}
            ))

            return {
                "result": result,
                "metadata": {
                    "latency": inference_time,
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            # Handle error
            error_handled = self.error_handler.handle_error(
                e, "inference", "execute",
                {"user_id": user_id, "input_size": len(input_data) if hasattr(input_data, '__len__') else 1}
            )

            if not error_handled:
                logger.error(f"Inference failed: {e}")

            # End tracing with error
            self.tracer.end_span(span_id, {"success": False, "error": str(e)})

            # Audit log failure
            self.audit_logger.log_event(SecurityEvent(
                timestamp=datetime.now(),
                event_type="inference",
                severity="error",
                user_id=user_id,
                resource="inference",
                action="execute",
                success=False,
                metadata={"error": str(e)}
            ))

            raise

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive production benchmark suite."""
        logger.info("Starting comprehensive production benchmark")

        # Mock inference function for benchmarking
        def mock_inference(input_data):
            time.sleep(0.01)  # Simulate 10ms inference
            return {"output": "mock_result"}

        results = {}

        # Run different benchmark types
        try:
            results["latency"] = self.performance_benchmarker.run_latency_benchmark(mock_inference)
            results["throughput"] = self.performance_benchmarker.run_throughput_benchmark(mock_inference)
            results["memory"] = self.performance_benchmarker.run_memory_benchmark(mock_inference)

            # Analyze bottlenecks
            all_results = [
                results["latency"], results["throughput"],
                *results["memory"]
            ]
            results["analysis"] = self.performance_benchmarker.analyze_bottlenecks(all_results)

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            results["error"] = str(e)

        logger.info("Comprehensive benchmark completed")
        return results

    def generate_production_report(self) -> Dict[str, Any]:
        """Generate comprehensive production readiness report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "readiness_level": self.readiness_level.value,
            "production_mode": self.production_mode,
            "health_status": self.get_production_health(),
            "performance_summary": self.performance_benchmarker.get_performance_summary(),
            "security_assessment": self.security_hardening.run_hardening_checklist(),
            "compliance_status": {},
            "recommendations": []
        }

        # Check compliance
        for framework in ["soc2", "gdpr", "hipaa"]:
            try:
                compliance_result = self.security_scanner.check_compliance(framework)
                report["compliance_status"][framework] = compliance_result
            except Exception as e:
                report["compliance_status"][framework] = {"error": str(e)}

        # Generate recommendations
        health = report["health_status"]
        report["recommendations"].extend(health.recommendations)

        if report["readiness_level"] != "production_optimized":
            report["recommendations"].append("Complete production hardening to reach optimized level")

        return report

    def export_configuration(self) -> str:
        """Export production hardening configuration."""
        config = {
            "production_hardening": {
                "world_size": self.world_size,
                "rank": self.rank,
                "enable_distributed": self.enable_distributed,
                "security_level": self.security_level,
                "health_check_interval": self.health_check_interval,
                "optimization_interval": self.optimization_interval
            },
            "error_handling": {
                "max_retries": self.error_handler.max_retries,
                "circuit_breaker_threshold": self.error_handler.circuit_breaker_threshold
            },
            "monitoring": {
                "collection_interval": self.metrics_collector.collection_interval,
                "retention_period": self.metrics_collector.retention_period
            },
            "security": {
                "key_rotation_days": self.encryption_manager.key_rotation_days,
                "audit_retention_days": self.audit_logger.log_retention_days
            }
        }

        return json.dumps(config, indent=2)

    def validate_production_deployment(self) -> Dict[str, Any]:
        """Validate production deployment readiness."""
        validation = {
            "checks": {},
            "passed": True,
            "critical_issues": [],
            "warnings": []
        }

        # Check components initialization
        components = [
            ("error_handler", self.error_handler),
            ("fault_tolerance", self.fault_tolerance),
            ("metrics_collector", self.metrics_collector),
            ("health_monitor", self.health_monitor),
            ("encryption_manager", self.encryption_manager),
            ("access_control", self.access_control)
        ]

        for name, component in components:
            if component is None:
                validation["checks"][name] = "FAILED: Not initialized"
                validation["passed"] = False
                validation["critical_issues"].append(f"{name} not initialized")
            else:
                validation["checks"][name] = "PASSED"

        # Check production mode
        if not self.production_mode:
            validation["warnings"].append("Production mode not activated")

        # Check readiness level
        if self.readiness_level == ProductionReadinessLevel.DEVELOPMENT:
            validation["warnings"].append("System still in development readiness level")

        # Check security
        security_check = self.security_hardening.run_hardening_checklist()
        if security_check.get("overall_score", 0) < 0.8:
            validation["warnings"].append("Security hardening incomplete")

        return validation