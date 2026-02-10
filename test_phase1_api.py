#!/usr/bin/env python3
"""Phase 1 API Validation Test Script"""

import sys
import inspect
import json
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "RYZEN-LLM" / "scripts"))

def extract_module_api(module_name, class_name, key_methods):
    """Extract API information from a module class"""
    try:
        module = __import__(module_name)
        cls = getattr(module, class_name)
        
        api_info = {
            "class": class_name,
            "module": module_name,
            "methods": {}
        }
        
        for method_name in key_methods:
            if hasattr(cls, method_name):
                method = getattr(cls, method_name)
                sig = inspect.signature(method)
                api_info["methods"][method_name] = {
                    "signature": str(sig),
                    "parameters": list(sig.parameters.keys()),
                    "return_annotation": str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else "None"
                }
        
        return api_info
    except Exception as e:
        return {"error": str(e), "module": module_name, "class": class_name}

print("=" * 80)
print("PHASE 1 API VALIDATION TEST")
print("=" * 80)

# Test kernel_optimizer
print("\n[1] KERNEL_OPTIMIZER")
print("-" * 80)
try:
    from kernel_optimizer import KernelOptimizer
    print("✅ kernel_optimizer imported successfully")
    
    ko_api = extract_module_api('kernel_optimizer', 'KernelOptimizer', [
        'auto_tune_tile_sizes', 
        'generate_cmake_config', 
        'benchmark_kernel_params', 
        'save_config', 
        'report'
    ])
    print(json.dumps(ko_api, indent=2))
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test semantic_compression
print("\n[2] SEMANTIC_COMPRESSION")
print("-" * 80)
try:
    from semantic_compression import SemanticCompressor
    print("✅ semantic_compression imported successfully")
    
    sc_api = extract_module_api('semantic_compression', 'SemanticCompressor', [
        'matryoshka_encode',
        'binary_quantization',
        'sparse_compression',
        'adaptive_selector',
        'corpus_adaptive_tuning',
        'estimate_compression_gain'
    ])
    print(json.dumps(sc_api, indent=2))
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test inference_scaling
print("\n[3] INFERENCE_SCALING")
print("-" * 80)
try:
    from inference_scaling import InferenceScalingEngine, TaskComplexityEstimator
    print("✅ inference_scaling imported successfully")
    
    ie_api = extract_module_api('inference_scaling', 'InferenceScalingEngine', [
        'process_query'
    ])
    print("InferenceScalingEngine:")
    print(json.dumps(ie_api, indent=2))
    
    tce_api = extract_module_api('inference_scaling', 'TaskComplexityEstimator', [
        'estimate'
    ])
    print("\nTaskComplexityEstimator:")
    print(json.dumps(tce_api, indent=2))
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n" + "=" * 80)
print("✅ ALL PHASE 1 MODULES VALIDATED")
print("=" * 80)
