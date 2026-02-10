#!/usr/bin/env python3
"""Test C++ bindings via ctypes (bypasses module init)."""
import ctypes
import os

# Load the DLL via ctypes
dll_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'python', 'ryzen_llm', 'ryzen_llm_bindings.pyd')
print(f"Loading: {dll_path}")
print(f"Exists: {os.path.exists(dll_path)}")

try:
    dll = ctypes.WinDLL(dll_path)
    print("SUCCESS: DLL loaded via ctypes")
    
    # Test exported functions
    functions = [
        'test_function',
        'test_quantize_scalar',
        'test_simple_loop',
        'test_nested_loops',
        'test_quantization_steps',
        'test_vector_allocation',
        'test_vector_access'
    ]
    
    for func_name in functions:
        try:
            func = getattr(dll, func_name)
            result = func()
            print(f"  {func_name}(): {result}")
        except Exception as e:
            print(f"  {func_name}(): ERROR - {e}")
            
except Exception as e:
    print(f"ERROR loading DLL: {type(e).__name__}: {e}")
