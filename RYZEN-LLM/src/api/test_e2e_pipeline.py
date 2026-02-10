#!/usr/bin/env python3
"""
End-to-End Pipeline Test for Ryzanstein LLM
Tests: Python API Server → MCP Server (external)

This script:
1. Starts the Python API server with mock engine
2. Tests the API server directly
3. Provides instructions for testing MCP server
"""

import subprocess
import sys
import time
import json
import os

def test_api_server():
    """Test the Python API server using in-process TestClient"""
    print("=" * 60)
    print("RYZANSTEIN LLM - E2E Pipeline Test")
    print("=" * 60)
    print()
    
    # Use in-process testing
    from fastapi.testclient import TestClient
    from server import app, engine, engine_type, USING_MOCK
    
    print("[Python API Server Status]")
    print(f"  Engine Type: {engine_type}")
    print(f"  Engine Loaded: {engine is not None}")
    print(f"  Using Mock: {USING_MOCK}")
    print()
    
    client = TestClient(app)
    
    # Test 1: Health check
    print("[Test 1] Health Check...")
    response = client.get("/health")
    assert response.status_code == 200
    health = response.json()
    print(f"  Status: {health['status']}")
    print(f"  Engine Type: {health['engine_type']}")
    print(f"  ✓ Health check passed")
    print()
    
    # Test 2: Models list
    print("[Test 2] Models List...")
    response = client.get("/v1/models")
    assert response.status_code == 200
    models = response.json()
    print(f"  Available Models: {[m['id'] for m in models['data']]}")
    print(f"  ✓ Models list passed")
    print()
    
    # Test 3: Chat completion
    print("[Test 3] Chat Completion...")
    response = client.post("/v1/chat/completions", json={
        "model": "bitnet-1.58b",
        "messages": [
            {"role": "system", "content": "You are Ryzanstein, a helpful AI assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    })
    assert response.status_code == 200
    chat = response.json()
    print(f"  Response ID: {chat['id']}")
    print(f"  Model: {chat['model']}")
    print(f"  Message: {chat['choices'][0]['message']['content'][:100]}...")
    print(f"  ✓ Chat completion passed")
    print()
    
    print("=" * 60)
    print("All Python API tests passed!")
    print("=" * 60)
    print()
    
    return True

def print_mcp_test_instructions():
    """Print instructions for testing MCP server"""
    print("=" * 60)
    print("MCP Server Testing Instructions")
    print("=" * 60)
    print()
    print("To test the full MCP → Python API pipeline:")
    print()
    print("1. Start the Python API server (in a new terminal):")
    print("   cd s:\\Ryot\\RYZEN-LLM\\src\\api")
    print("   python -m uvicorn server:app --host 127.0.0.1 --port 8000")
    print()
    print("2. Test the API server health (in another terminal):")
    print("   curl http://127.0.0.1:8000/health")
    print()
    print("3. Start the MCP server (in another terminal):")
    print("   cd s:\\Ryot\\mcp")
    print("   .\\mcp-server.exe")
    print()
    print("4. Test the MCP server using grpcurl:")
    print("   grpcurl -plaintext -d '{\"prompt\": \"Hello\"}' localhost:50051 ryzanstein.RyzansteinService/GenerateText")
    print()
    print("Or test using the Python gRPC client:")
    print("   python test_mcp_client.py")
    print()

def main():
    """Main entry point"""
    try:
        # Run API server tests
        test_api_server()
        
        # Print MCP testing instructions
        print_mcp_test_instructions()
        
        return 0
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
