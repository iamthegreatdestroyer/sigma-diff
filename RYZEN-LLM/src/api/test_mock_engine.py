#!/usr/bin/env python
"""
Quick test of the mock engine and API server components.
"""

import sys
import os

# Add API directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("RYZANSTEIN LLM - Mock Engine & API Test")
print("=" * 60)

# Test 1: Mock Engine
print("\n[Test 1] Mock Engine Import...")
try:
    import mock_engine
    print("  ✓ Mock engine module imported")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\n[Test 2] Create Mock Engine...")
try:
    config = mock_engine.create_bitnet_1_58b_config()
    engine = mock_engine.MockBitNetEngine(config)
    print(f"  ✓ Engine created: {engine._model_name}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\n[Test 3] Generate Text...")
try:
    response = engine.generate_text("Hello, who are you?")
    print(f"  ✓ Response: {response[:100]}...")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\n[Test 4] Generate with Different Prompts...")
prompts = [
    "What can you help with?",
    "Write some code for me",
    "Tell me about algorithms"
]
for prompt in prompts:
    try:
        response = engine.generate_text(prompt)
        print(f"  Prompt: '{prompt}'")
        print(f"  Response: {response[:80]}...")
        print()
    except Exception as e:
        print(f"  ✗ Failed for '{prompt}': {e}")

print("\n[Test 5] Model Info...")
try:
    info = engine.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
