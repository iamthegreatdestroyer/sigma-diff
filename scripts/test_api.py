#!/usr/bin/env python3
"""
Test script for verifying the Ryzanstein LLM API stack.

This script tests the entire inference pipeline:
1. Python API server health
2. Chat completion endpoint
3. Model listing

Usage:
    python test_api.py [--url URL]
"""

import sys
import json
import argparse
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests


def test_health(base_url: str) -> bool:
    """Test the health endpoint."""
    print("\n" + "="*60)
    print("Testing /health endpoint...")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(data, indent=2)}")
        
        if response.status_code == 200 and data.get("status") == "healthy":
            print("✓ Health check PASSED")
            return True
        else:
            print("✗ Health check FAILED - unexpected response")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Health check FAILED - Cannot connect to API server")
        print(f"  Make sure the API server is running at {base_url}")
        return False
    except Exception as e:
        print(f"✗ Health check FAILED - {e}")
        return False


def test_models(base_url: str) -> bool:
    """Test the models listing endpoint."""
    print("\n" + "="*60)
    print("Testing /v1/models endpoint...")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(data, indent=2)}")
        
        if response.status_code == 200 and "data" in data:
            print(f"✓ Models endpoint PASSED - Found {len(data.get('data', []))} model(s)")
            return True
        else:
            print("✗ Models endpoint FAILED - unexpected response")
            return False
            
    except Exception as e:
        print(f"✗ Models endpoint FAILED - {e}")
        return False


def test_chat_completion(base_url: str) -> bool:
    """Test the chat completion endpoint."""
    print("\n" + "="*60)
    print("Testing /v1/chat/completions endpoint...")
    print("="*60)
    
    payload = {
        "model": "ryzanstein-bitnet-3b",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Hello! Please respond with a short greeting."}
        ],
        "max_tokens": 64,
        "temperature": 0.7,
        "stream": False
    }
    
    print(f"Request payload:\n{json.dumps(payload, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        elapsed = time.time() - start_time
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response:\n{json.dumps(data, indent=2)}")
            
            # Check for expected fields
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                if content:
                    print(f"\n✓ Chat completion PASSED")
                    print(f"  Generated content: {content[:200]}...")
                    return True
                else:
                    print("✗ Chat completion FAILED - Empty response content")
                    return False
            else:
                print("✗ Chat completion FAILED - No choices in response")
                return False
        else:
            print(f"Response: {response.text}")
            print("✗ Chat completion FAILED - Non-200 status")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Chat completion FAILED - Request timed out")
        return False
    except Exception as e:
        print(f"✗ Chat completion FAILED - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Ryzanstein LLM API")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--skip-inference", action="store_true", help="Skip chat completion test")
    args = parser.parse_args()
    
    print("="*60)
    print("     RYZANSTEIN LLM API TEST SUITE")
    print("="*60)
    print(f"Testing API at: {args.url}")
    
    results = {}
    
    # Test health
    results["health"] = test_health(args.url)
    
    # Only continue if health passes
    if results["health"]:
        # Test models
        results["models"] = test_models(args.url)
        
        # Test chat completion (if not skipped)
        if not args.skip_inference:
            results["chat"] = test_chat_completion(args.url)
        else:
            print("\n[Skipping chat completion test]")
            results["chat"] = None
    else:
        print("\n⚠ Skipping remaining tests - API server not available")
        results["models"] = None
        results["chat"] = None
    
    # Summary
    print("\n" + "="*60)
    print("     TEST SUMMARY")
    print("="*60)
    
    total = 0
    passed = 0
    
    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED"
            color = "⚪"
        elif result:
            status = "PASSED"
            color = "✓"
            passed += 1
            total += 1
        else:
            status = "FAILED"
            color = "✗"
            total += 1
        
        print(f"  {color} {test_name}: {status}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if total > 0 and passed == total:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
