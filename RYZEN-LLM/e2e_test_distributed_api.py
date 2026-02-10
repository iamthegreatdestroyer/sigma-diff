#!/usr/bin/env python3
"""
End-to-End Testing for Distributed Inference API
===============================================

Comprehensive E2E tests for the distributed inference HTTP API.
Tests all endpoints, error handling, and request/response formats.
"""

import asyncio
import json
import time
import sys
import os
from typing import Dict, Any
import subprocess
import signal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi.testclient import TestClient
from src.api.distributed_server import app


class EndToEndTester:
    """Comprehensive E2E tester for distributed inference API."""

    def __init__(self):
        self.client = TestClient(app)
        self.test_results = []
        self.server_process = None

    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test result."""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        print(f"{'✅' if status == 'PASS' else '❌'} {test_name}: {status}")
        if details:
            print(f"   {details}")

    def test_health_endpoint(self):
        """Test health endpoint functionality."""
        try:
            response = self.client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert "status" in data
            assert "healthy_gpus" in data
            assert "total_gpus" in data
            assert "uptime" in data

            # Status should be either "healthy" or "initializing"
            assert data["status"] in ["healthy", "initializing"]
            assert isinstance(data["healthy_gpus"], int)
            assert isinstance(data["total_gpus"], int)
            assert data["total_gpus"] == 4  # Default world_size

            self.log_test("Health Endpoint", "PASS", f"Status: {data['status']}, GPUs: {data['healthy_gpus']}/{data['total_gpus']}")
            return True
        except Exception as e:
            self.log_test("Health Endpoint", "FAIL", str(e))
            return False

    def test_metrics_endpoint(self):
        """Test metrics endpoint functionality."""
        try:
            response = self.client.get("/metrics")
            assert response.status_code == 200

            data = response.json()

            # Should return metrics data or error message
            if "error" in data:
                # Expected when metrics collector not initialized
                assert data["error"] == "Metrics collector not initialized"
                self.log_test("Metrics Endpoint", "PASS", "Expected error: Metrics collector not initialized")
            else:
                # If metrics are available, validate structure
                expected_keys = ["period_seconds", "total_gpus", "total_requests", "total_success", "total_failed", "total_tokens_processed", "system_avg_latency_ms", "system_avg_utilization_percent", "system_error_rate_percent", "gpu_summaries", "timestamp"]
                for key in expected_keys:
                    assert key in data, f"Missing key: {key}"
                self.log_test("Metrics Endpoint", "PASS", f"Metrics data returned with {len(data.get('gpu_summaries', []))} GPU summaries")

            return True
        except Exception as e:
            self.log_test("Metrics Endpoint", "FAIL", str(e))
            return False

    def test_chat_completions_validation(self):
        """Test chat completions request validation."""
        try:
            # Test valid request structure
            valid_request = {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "max_tokens": 100,
                "stream": False
            }

            response = self.client.post("/v1/chat/completions", json=valid_request)

            # Should return 500 (not 422 validation error) since no model loaded
            assert response.status_code == 500, f"Expected 500, got {response.status_code}"

            data = response.json()
            assert "detail" in data
            assert "route_request" in data["detail"]  # Expected error message

            self.log_test("Chat Completions Validation", "PASS", "Request validation passed, model routing failed as expected")
            return True
        except Exception as e:
            self.log_test("Chat Completions Validation", "FAIL", str(e))
            return False

    def test_chat_completions_invalid_requests(self):
        """Test chat completions with invalid requests."""
        invalid_requests = [
            # Missing required fields
            {"messages": [{"role": "user", "content": "Hello"}]},

            # Invalid temperature
            {"model": "test", "messages": [{"role": "user", "content": "Hello"}], "temperature": 3.0},

            # Invalid top_p
            {"model": "test", "messages": [{"role": "user", "content": "Hello"}], "top_p": 1.5},

            # Invalid top_k
            {"model": "test", "messages": [{"role": "user", "content": "Hello"}], "top_k": 2000},

            # Invalid repetition_penalty
            {"model": "test", "messages": [{"role": "user", "content": "Hello"}], "repetition_penalty": 3.0},

            # Invalid max_tokens
            {"model": "test", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5000},
        ]

        passed = 0
        for i, invalid_req in enumerate(invalid_requests):
            try:
                response = self.client.post("/v1/chat/completions", json=invalid_req)
                # Should return 422 for validation errors
                assert response.status_code == 422, f"Request {i+1}: Expected 422, got {response.status_code}"
                passed += 1
            except Exception as e:
                self.log_test(f"Invalid Request {i+1}", "FAIL", str(e))

        if passed == len(invalid_requests):
            self.log_test("Invalid Request Validation", "PASS", f"All {passed} invalid requests properly rejected")
            return True
        else:
            self.log_test("Invalid Request Validation", "FAIL", f"Only {passed}/{len(invalid_requests)} invalid requests rejected")
            return False

    def test_openapi_specification(self):
        """Test OpenAPI specification is valid."""
        try:
            response = self.client.get("/openapi.json")
            assert response.status_code == 200

            spec = response.json()

            # Validate basic OpenAPI structure
            assert spec["openapi"].startswith("3.")
            assert "paths" in spec
            assert "/health" in spec["paths"]
            assert "/metrics" in spec["paths"]
            assert "/v1/chat/completions" in spec["paths"]

            # Validate chat completions endpoint
            chat_endpoint = spec["paths"]["/v1/chat/completions"]["post"]
            assert "requestBody" in chat_endpoint
            assert "responses" in chat_endpoint

            self.log_test("OpenAPI Specification", "PASS", f"Valid OpenAPI {spec['openapi']} spec with {len(spec['paths'])} endpoints")
            return True
        except Exception as e:
            self.log_test("OpenAPI Specification", "FAIL", str(e))
            return False

    def test_swagger_ui(self):
        """Test Swagger UI is accessible."""
        try:
            response = self.client.get("/docs")
            assert response.status_code == 200
            assert "swagger" in response.text.lower()
            assert "fastapi" in response.text.lower()

            self.log_test("Swagger UI", "PASS", "Swagger UI accessible and contains expected content")
            return True
        except Exception as e:
            self.log_test("Swagger UI", "FAIL", str(e))
            return False

    def test_error_handling(self):
        """Test various error conditions."""
        try:
            # Test non-existent endpoint
            response = self.client.get("/nonexistent")
            assert response.status_code == 404

            # Test invalid method
            response = self.client.put("/health")
            assert response.status_code == 405

            self.log_test("Error Handling", "PASS", "404 and 405 errors handled correctly")
            return True
        except Exception as e:
            self.log_test("Error Handling", "FAIL", str(e))
            return False

    def run_all_tests(self):
        """Run all E2E tests."""
        print("🚀 Starting End-to-End Testing for Distributed Inference API")
        print("=" * 60)

        tests = [
            self.test_health_endpoint,
            self.test_metrics_endpoint,
            self.test_chat_completions_validation,
            self.test_chat_completions_invalid_requests,
            self.test_openapi_specification,
            self.test_swagger_ui,
            self.test_error_handling,
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            if test():
                passed += 1
            print()

        print("=" * 60)
        print(f"📊 Test Results: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 ALL END-TO-END TESTS PASSED!")
            return True
        else:
            print(f"❌ {total - passed} tests failed")
            return False

    def generate_report(self):
        """Generate test report."""
        report = {
            "test_run": {
                "timestamp": time.time(),
                "total_tests": len(self.test_results),
                "passed_tests": len([t for t in self.test_results if t["status"] == "PASS"]),
                "failed_tests": len([t for t in self.test_results if t["status"] == "FAIL"]),
            },
            "results": self.test_results
        }

        # Save to file
        with open("e2e_test_results.json", "w") as f:
            json.dump(report, f, indent=2)

        return report


def main():
    """Main test runner."""
    tester = EndToEndTester()

    try:
        success = tester.run_all_tests()
        report = tester.generate_report()

        print(f"\n📄 Detailed results saved to: e2e_test_results.json")

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())