#!/usr/bin/env python3
"""
Production Deployment Script for Ryzanstein LLM
===========================================

Complete production deployment and hardening script that sets up
the distributed inference system with full production hardening,
monitoring, security, and benchmarking capabilities.

Usage:
    python deploy_production.py [options]

Options:
    --world-size N        Number of processes in distributed setup (default: auto-detect)
    --rank N             Process rank (default: auto-detect)
    --config FILE        Configuration file path
    --security-level     Security level: low|medium|high (default: high)
    --benchmark-only     Run benchmarks only, don't start service
    --validate-only      Validate deployment only, don't start
    --help               Show this help message
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.production_hardening import ProductionHardeningSuite, ProductionReadinessLevel
from core.error_handling import ProductionErrorHandler
from core.monitoring import MetricsCollector, DistributedTracer, HealthMonitor
from core.benchmarking import PerformanceBenchmarker
from core.security import (
    EncryptionManager, AccessControlManager, SecurityScanner,
    AuditLogger, SecurityHardeningManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ProductionDeployer:
    """
    Production deployment orchestrator for Ryzanstein LLM.

    Handles complete production setup including:
    - Distributed system initialization
    - Security hardening
    - Monitoring setup
    - Performance benchmarking
    - Production readiness validation
    """

    def __init__(
        self,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        config_file: Optional[str] = None,
        security_level: str = "high"
    ):
        self.world_size = world_size
        self.rank = rank
        self.config_file = config_file
        self.security_level = security_level

        # Load configuration
        self.config = self._load_config()

        # Initialize components
        self.hardening_suite: Optional[ProductionHardeningSuite] = None
        self.deployment_status = "initializing"

        logger.info("ProductionDeployer initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        default_config = {
            "distributed": {
                "world_size": self.world_size or self._auto_detect_world_size(),
                "rank": self.rank or self._auto_detect_rank(),
                "enable_distributed": True
            },
            "security": {
                "level": self.security_level,
                "key_rotation_days": 90,
                "audit_retention_days": 2555
            },
            "monitoring": {
                "collection_interval": 60.0,
                "retention_period": 3600.0,
                "health_check_interval": 60.0
            },
            "benchmarking": {
                "warmup_iterations": 10,
                "benchmark_iterations": 100,
                "enable_distributed": True
            },
            "production": {
                "optimization_interval": 300.0,
                "auto_healing": True,
                "incident_response": True
            }
        }

        # Load from file if specified
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
                # Merge with defaults
                self._merge_configs(default_config, file_config)

        return default_config

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def _auto_detect_world_size(self) -> int:
        """Auto-detect world size from environment."""
        # Check for torchrun/distributed environment
        world_size = os.environ.get('WORLD_SIZE')
        if world_size:
            return int(world_size)

        # Check for SLURM
        slurm_world_size = os.environ.get('SLURM_NTASKS')
        if slurm_world_size:
            return int(slurm_world_size)

        # Default to single node
        return 1

    def _auto_detect_rank(self) -> int:
        """Auto-detect rank from environment."""
        # Check for torchrun/distributed environment
        rank = os.environ.get('RANK')
        if rank:
            return int(rank)

        # Check for SLURM
        slurm_rank = os.environ.get('SLURM_PROCID')
        if slurm_rank:
            return int(slurm_rank)

        # Default to rank 0
        return 0

    def validate_prerequisites(self) -> Dict[str, Any]:
        """Validate system prerequisites for production deployment."""
        validation = {
            "checks": {},
            "passed": True,
            "warnings": [],
            "errors": []
        }

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            validation["errors"].append("Python 3.8+ required")
            validation["passed"] = False
        else:
            validation["checks"]["python_version"] = f"✓ Python {python_version.major}.{python_version.minor}"

        # Check PyTorch
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            validation["checks"]["pytorch"] = f"✓ PyTorch {torch.__version__}"
            if cuda_available:
                validation["checks"]["cuda"] = f"✓ CUDA available ({torch.cuda.device_count()} devices)"
            else:
                validation["warnings"].append("CUDA not available - using CPU only")
        except ImportError:
            validation["errors"].append("PyTorch not installed")
            validation["passed"] = False

        # Check distributed setup
        if self.config["distributed"]["enable_distributed"]:
            try:
                import torch.distributed as dist
                validation["checks"]["torch_distributed"] = "✓ torch.distributed available"
            except ImportError:
                validation["errors"].append("torch.distributed not available")
                validation["passed"] = False

        # Check security dependencies
        try:
            from cryptography.fernet import Fernet
            validation["checks"]["cryptography"] = "✓ cryptography library available"
        except ImportError:
            validation["errors"].append("cryptography library not installed")
            validation["passed"] = False

        # Check disk space
        import psutil
        disk_usage = psutil.disk_usage('/')
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 10:
            validation["warnings"].append(".1f")
        else:
            validation["checks"]["disk_space"] = ".1f"

        # Check memory
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        validation["checks"]["memory"] = ".1f"

        if available_gb < 4:
            validation["warnings"].append(".1f")

        return validation

    def initialize_production_suite(self) -> ProductionHardeningSuite:
        """Initialize the production hardening suite."""
        logger.info("Initializing production hardening suite...")

        self.hardening_suite = ProductionHardeningSuite(
            world_size=self.config["distributed"]["world_size"],
            rank=self.config["distributed"]["rank"],
            enable_distributed=self.config["distributed"]["enable_distributed"],
            security_level=self.config["security"]["level"]
        )

        # Configure components based on loaded config
        self._configure_components()

        logger.info("Production hardening suite initialized")
        return self.hardening_suite

    def _configure_components(self):
        """Configure individual components based on config."""
        if not self.hardening_suite:
            return

        # Configure error handling
        self.hardening_suite.error_handler.max_retries = 5
        self.hardening_suite.error_handler.circuit_breaker_threshold = 10

        # Configure monitoring
        self.hardening_suite.metrics_collector.collection_interval = self.config["monitoring"]["collection_interval"]
        self.hardening_suite.metrics_collector.retention_period = self.config["monitoring"]["retention_period"]
        self.hardening_suite.health_check_interval = self.config["monitoring"]["health_check_interval"]

        # Configure benchmarking
        self.hardening_suite.performance_benchmarker.warmup_iterations = self.config["benchmarking"]["warmup_iterations"]
        self.hardening_suite.performance_benchmarker.benchmark_iterations = self.config["benchmarking"]["benchmark_iterations"]

        # Configure security
        self.hardening_suite.encryption_manager.key_rotation_days = self.config["security"]["key_rotation_days"]
        self.hardening_suite.audit_logger.log_retention_days = self.config["security"]["audit_retention_days"]

        # Configure production settings
        self.hardening_suite.optimization_interval = self.config["production"]["optimization_interval"]

    def run_pre_deployment_checks(self) -> Dict[str, Any]:
        """Run comprehensive pre-deployment checks."""
        logger.info("Running pre-deployment checks...")

        checks = {
            "prerequisites": self.validate_prerequisites(),
            "security_hardening": {},
            "benchmarking": {},
            "integration": {}
        }

        # Security hardening checks
        if self.hardening_suite:
            checks["security_hardening"] = self.hardening_suite.security_hardening.run_hardening_checklist()

        # Benchmarking validation
        if self.hardening_suite:
            try:
                benchmark_results = self.hardening_suite.run_comprehensive_benchmark()
                checks["benchmarking"] = {
                    "status": "completed",
                    "results": benchmark_results
                }
            except Exception as e:
                checks["benchmarking"] = {
                    "status": "failed",
                    "error": str(e)
                }

        # Integration tests
        checks["integration"] = self._run_integration_tests()

        # Overall assessment
        checks["overall_ready"] = self._assess_deployment_readiness(checks)

        return checks

    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests for all components."""
        tests = {
            "error_handling": False,
            "monitoring": False,
            "benchmarking": False,
            "security": False,
            "overall": False
        }

        if not self.hardening_suite:
            return tests

        try:
            # Test error handling
            self.hardening_suite.error_handler.handle_error(
                ValueError("test error"), "test", "test"
            )
            tests["error_handling"] = True
        except Exception as e:
            logger.warning(f"Error handling test failed: {e}")

        try:
            # Test monitoring
            self.hardening_suite.metrics_collector.increment_counter("test_counter")
            summary = self.hardening_suite.metrics_collector.get_metrics_summary()
            if "test_counter" in summary.get("counters", {}):
                tests["monitoring"] = True
        except Exception as e:
            logger.warning(f"Monitoring test failed: {e}")

        try:
            # Test security
            encrypted = self.hardening_suite.encryption_manager.encrypt_data("test")
            decrypted = self.hardening_suite.encryption_manager.decrypt_data(encrypted)
            if decrypted == "test":
                tests["security"] = True
        except Exception as e:
            logger.warning(f"Security test failed: {e}")

        tests["overall"] = all(tests.values())
        return tests

    def _assess_deployment_readiness(self, checks: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall deployment readiness."""
        assessment = {
            "ready": True,
            "blocking_issues": [],
            "warnings": [],
            "recommendations": []
        }

        # Check prerequisites
        prereqs = checks.get("prerequisites", {})
        if not prereqs.get("passed", False):
            assessment["ready"] = False
            assessment["blocking_issues"].extend(prereqs.get("errors", []))

        assessment["warnings"].extend(prereqs.get("warnings", []))

        # Check security
        security = checks.get("security_hardening", {})
        security_score = security.get("overall_score", 0)
        if security_score < 0.7:
            assessment["ready"] = False
            assessment["blocking_issues"].append("Security hardening incomplete")
        elif security_score < 0.9:
            assessment["warnings"].append("Security hardening could be improved")

        # Check integration
        integration = checks.get("integration", {})
        if not integration.get("overall", False):
            assessment["warnings"].append("Some integration tests failed")

        return assessment

    def deploy_production_system(self) -> Dict[str, Any]:
        """Deploy the complete production system."""
        logger.info("Starting production deployment...")

        deployment_result = {
            "status": "starting",
            "timestamp": datetime.now().isoformat(),
            "phases": {},
            "final_status": "unknown"
        }

        try:
            # Phase 1: Pre-deployment validation
            logger.info("Phase 1: Pre-deployment validation")
            validation = self.run_pre_deployment_checks()
            deployment_result["phases"]["validation"] = validation

            if not validation["overall_ready"]["ready"]:
                deployment_result["status"] = "failed"
                deployment_result["final_status"] = "validation_failed"
                logger.error("Pre-deployment validation failed")
                return deployment_result

            # Phase 2: Component initialization
            logger.info("Phase 2: Component initialization")
            if not self.hardening_suite:
                self.initialize_production_suite()

            deployment_result["phases"]["initialization"] = {
                "status": "completed",
                "components_initialized": True
            }

            # Phase 3: Production mode activation
            logger.info("Phase 3: Production mode activation")
            if self.hardening_suite:
                self.hardening_suite.start_production_mode()

                # Wait for system to stabilize
                time.sleep(5)

                # Check health
                health = self.hardening_suite.get_production_health()
                deployment_result["phases"]["activation"] = {
                    "status": "completed",
                    "health_status": health.overall_health,
                    "readiness_level": health.readiness_level.value
                }

            # Phase 4: Final validation
            logger.info("Phase 4: Final validation")
            final_validation = self.hardening_suite.validate_production_deployment() if self.hardening_suite else {}
            deployment_result["phases"]["final_validation"] = final_validation

            # Success
            deployment_result["status"] = "completed"
            deployment_result["final_status"] = "production_ready"
            logger.info("Production deployment completed successfully")

        except Exception as e:
            deployment_result["status"] = "failed"
            deployment_result["final_status"] = "deployment_error"
            deployment_result["error"] = str(e)
            logger.error(f"Production deployment failed: {e}")

        return deployment_result

    def run_production_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive production benchmarks."""
        if not self.hardening_suite:
            return {"error": "Production suite not initialized"}

        logger.info("Running production benchmarks...")

        try:
            results = self.hardening_suite.run_comprehensive_benchmark()

            # Generate benchmark report
            report = {
                "timestamp": datetime.now().isoformat(),
                "system_info": self.hardening_suite.system_info,
                "benchmark_results": results,
                "analysis": self.hardening_suite.performance_benchmarker.analyze_bottlenecks(
                    [results.get("latency"), results.get("throughput")] +
                    results.get("memory", [])
                ) if "latency" in results else {}
            }

            logger.info("Production benchmarks completed")
            return report

        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            return {"error": str(e)}

    def generate_deployment_report(self, deployment_result: Dict[str, Any]) -> str:
        """Generate comprehensive deployment report."""
        report = f"""
# Production Deployment Report
Generated: {datetime.now().isoformat()}

## Deployment Status
Status: {deployment_result.get('status', 'unknown').upper()}
Final Status: {deployment_result.get('final_status', 'unknown').upper()}

## Deployment Phases
"""

        for phase_name, phase_data in deployment_result.get('phases', {}).items():
            report += f"\n### {phase_name.title()}\n"
            if isinstance(phase_data, dict):
                for key, value in phase_data.items():
                    report += f"- **{key}**: {value}\n"
            else:
                report += f"- Status: {phase_data}\n"

        # Add configuration summary
        report += f"""
## Configuration Summary
- World Size: {self.config['distributed']['world_size']}
- Rank: {self.config['distributed']['rank']}
- Security Level: {self.config['security']['level']}
- Distributed: {self.config['distributed']['enable_distributed']}

## Recommendations
"""

        if deployment_result.get('final_status') == 'production_ready':
            report += "- System is production-ready\n"
            report += "- Monitor performance metrics regularly\n"
            report += "- Keep security patches up to date\n"
        else:
            report += "- Address blocking issues before production deployment\n"
            report += "- Review warnings and implement improvements\n"

        return report

    def cleanup_deployment(self):
        """Clean up deployment resources."""
        logger.info("Cleaning up deployment resources...")

        if self.hardening_suite:
            self.hardening_suite.stop_production_mode()

        # Additional cleanup as needed
        logger.info("Deployment cleanup completed")


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Ryzanstein LLM Production Deployment")
    parser.add_argument("--world-size", type=int, help="Number of processes in distributed setup")
    parser.add_argument("--rank", type=int, help="Process rank")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--security-level", choices=["low", "medium", "high"], default="high",
                       help="Security level")
    parser.add_argument("--benchmark-only", action="store_true",
                       help="Run benchmarks only, don't start service")
    parser.add_argument("--validate-only", action="store_true",
                       help="Validate deployment only, don't start")
    parser.add_argument("--help", action="help", help="Show this help message")

    args = parser.parse_args()

    # Initialize deployer
    deployer = ProductionDeployer(
        world_size=args.world_size,
        rank=args.rank,
        config_file=args.config,
        security_level=args.security_level
    )

    try:
        if args.validate_only:
            # Validation only
            logger.info("Running validation only...")
            validation = deployer.run_pre_deployment_checks()

            print("\n=== DEPLOYMENT VALIDATION RESULTS ===")
            print(json.dumps(validation, indent=2))

            if validation["overall_ready"]["ready"]:
                print("\n✅ System is ready for production deployment")
                sys.exit(0)
            else:
                print("\n❌ System is NOT ready for production deployment")
                sys.exit(1)

        elif args.benchmark_only:
            # Benchmarks only
            logger.info("Running benchmarks only...")
            deployer.initialize_production_suite()

            if deployer.hardening_suite:
                results = deployer.run_production_benchmarks()

                print("\n=== PRODUCTION BENCHMARK RESULTS ===")
                print(json.dumps(results, indent=2))
            else:
                print("Failed to initialize production suite")
                sys.exit(1)

        else:
            # Full deployment
            logger.info("Starting full production deployment...")

            # Run deployment
            result = deployer.deploy_production_system()

            # Generate and save report
            report = deployer.generate_deployment_report(result)

            with open("production_deployment_report.md", "w") as f:
                f.write(report)

            print("\n=== DEPLOYMENT RESULTS ===")
            print(f"Status: {result['status']}")
            print(f"Final Status: {result['final_status']}")
            print("Report saved to: production_deployment_report.md"

            if result['final_status'] == 'production_ready':
                print("\n🎉 Production deployment successful!")
                print("System is now running in production mode with full hardening.")
            else:
                print(f"\n❌ Deployment failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        deployer.cleanup_deployment()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Deployment failed with error: {e}")
        deployer.cleanup_deployment()
        sys.exit(1)
    finally:
        deployer.cleanup_deployment()


if __name__ == "__main__":
    main()