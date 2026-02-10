#!/usr/bin/env python
"""
Test the API server in-process without needing HTTP.
"""

import sys
import os

# Add API directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("RYZANSTEIN LLM - API Server Test (In-Process)")
print("=" * 60)

# Import test client from FastAPI
from fastapi.testclient import TestClient

# Import our server
print("\n[Loading Server]")
import server
print(f"  Engine type: {server.engine_type}")
print(f"  Engine loaded: {server.engine is not None}")
print(f"  Using mock: {server.USING_MOCK}")

# Create test client
client = TestClient(server.app)

print("\n[Test 1] Root Endpoint...")
response = client.get("/")
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}")

print("\n[Test 2] Health Endpoint...")
response = client.get("/health")
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}")

print("\n[Test 3] Models Endpoint...")
response = client.get("/v1/models")
print(f"  Status: {response.status_code}")
print(f"  Response: {response.json()}")

print("\n[Test 4] Chat Completion...")
response = client.post("/v1/chat/completions", json={
    "model": "bitnet-1.58b",
    "messages": [
        {"role": "user", "content": "Hello, who are you?"}
    ],
    "max_tokens": 100
})
print(f"  Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"  ID: {data.get('id')}")
    print(f"  Model: {data.get('model')}")
    print(f"  Response: {data['choices'][0]['message']['content'][:150]}...")
    print(f"  Usage: {data.get('usage')}")
else:
    print(f"  Error: {response.text}")

print("\n[Test 5] Different prompts...")
prompts = [
    "What can you help me with?",
    "Explain how a hash table works"
]
for prompt in prompts:
    response = client.post("/v1/chat/completions", json={
        "model": "bitnet-1.58b", 
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50
    })
    if response.status_code == 200:
        data = response.json()
        print(f"\n  Q: {prompt}")
        print(f"  A: {data['choices'][0]['message']['content'][:100]}...")
    else:
        print(f"\n  Q: {prompt}")
        print(f"  Error: {response.status_code}")

print("\n" + "=" * 60)
print("API Server Test Complete!")
print("=" * 60)
