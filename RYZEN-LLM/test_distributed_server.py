#!/usr/bin/env python3
"""
Test Distributed API Server Startup
===================================

Simple test script to verify the distributed API server can start up
and handle basic requests.
"""

import asyncio
import time
import requests
import threading
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from api.distributed_server import DistributedAPIServer


def test_server_startup():
    """Test that the server can start up successfully."""
    print("Testing distributed API server startup...")

    # Create server instance
    server = DistributedAPIServer(world_size=2, port=8001)

    # Start server in background thread
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait for server to start and initialize
    time.sleep(5)

    try:
        # Test health endpoint
        response = requests.get("http://localhost:8001/health", timeout=5)
        print(f"Health response status: {response.status_code}")
        print(f"Health response: {response.text}")
        assert response.status_code == 200

        data = response.json()
        print(f"Parsed health data: {data}")
        # Accept both "healthy" and "initializing" as valid startup states
        assert data["status"] in ["healthy", "initializing"]
        assert "healthy_gpus" in data
        assert data["total_gpus"] == 2

        print("✓ Health check passed")

        # Test metrics endpoint
        response = requests.get("http://localhost:8001/metrics", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert "total_requests" in data

        print("✓ Metrics endpoint passed")

        # Test chat completions endpoint
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello, test message"}],
            "temperature": 0.7,
            "max_tokens": 50
        }

        response = requests.post(
            "http://localhost:8001/v1/chat/completions",
            json=payload,
            timeout=10
        )
        assert response.status_code == 200

        data = response.json()
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert "content" in data["choices"][0]["message"]

        print("✓ Chat completions endpoint passed")

        print("🎉 All distributed API server tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_server_startup()
    sys.exit(0 if success else 1)