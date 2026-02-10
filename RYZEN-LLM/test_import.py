import sys
import os
import ctypes
import numpy as np

print("Current directory:", os.getcwd())
print("Python path before:", sys.path)

sys.path.insert(0, 'build/python')

print("Python path after:", sys.path)

# Try loading the DLL directly with ctypes
dll_path = 'build/python/ryzanstein_llm/ryzen_llm_bindings.pyd'
print(f"Trying to load DLL directly: {dll_path}")
try:
    dll = ctypes.CDLL(dll_path)
    print("DLL loaded successfully with ctypes")

    # Test basic function
    try:
        test_function = dll.test_function
        test_function.restype = ctypes.c_int
        result = test_function()
        print(f"C function test_function() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_function: {e}")

    # Test minimal quantization function
    try:
        test_quantize_scalar = dll.test_quantize_scalar
        test_quantize_scalar.restype = ctypes.c_int
        result = test_quantize_scalar()
        print(f"C function test_quantize_scalar() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_quantize_scalar: {e}")

    # Test weights-only computation
    try:
        test_quantize_weights_only = dll.test_quantize_weights_only
        test_quantize_weights_only.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32]
        test_quantize_weights_only.restype = ctypes.c_int
        
        # Create test weights
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype=np.float32)
        weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        result = test_quantize_weights_only(weights_ptr, 4, 4)
        expected_sum = np.sum(weights) * 1000
        print(f"C function test_quantize_weights_only() returned: {result}, expected: {int(expected_sum)}")
    except Exception as e:
        print(f"Failed to call test_quantize_weights_only: {e}")

    # Test object creation
    try:
        test_create_ternary_weight = dll.test_create_ternary_weight
        test_create_ternary_weight.restype = ctypes.c_int
        result = test_create_ternary_weight()
        print(f"C function test_create_ternary_weight() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_create_ternary_weight: {e}")

    # Test QuantConfig creation
    try:
        test_create_quant_config = dll.test_create_quant_config
        test_create_quant_config.restype = ctypes.c_int
        result = test_create_quant_config()
        print(f"C function test_create_quant_config() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_create_quant_config: {e}")

    # Test direct scalar quantization
    try:
        test_scalar_quantize_direct = dll.test_scalar_quantize_direct
        test_scalar_quantize_direct.restype = ctypes.c_int
        result = test_scalar_quantize_direct()
        print(f"C function test_scalar_quantize_direct() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_scalar_quantize_direct: {e}")

    # Test quantization function without object creation
    try:
        test_quantize_weights_only_scalar = dll.test_quantize_weights_only_scalar
        test_quantize_weights_only_scalar.restype = ctypes.c_int
        result = test_quantize_weights_only_scalar()
        print(f"C function test_quantize_weights_only_scalar() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_quantize_weights_only_scalar: {e}")

    # Test basic quantization operations
    try:
        test_basic_quantize_ops = dll.test_basic_quantize_ops
        test_basic_quantize_ops.restype = ctypes.c_int
        result = test_basic_quantize_ops()
        print(f"C function test_basic_quantize_ops() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_basic_quantize_ops: {e}")

    # Test quantization without std::vector
    try:
        test_quantize_no_vector = dll.test_quantize_no_vector
        test_quantize_no_vector.restype = ctypes.c_int
        result = test_quantize_no_vector()
        print(f"C function test_quantize_no_vector() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_quantize_no_vector: {e}")

    # Test basic quantization with larger data
    try:
        test_basic_quantize_large = dll.test_basic_quantize_large
        test_basic_quantize_large.restype = ctypes.c_int
        result = test_basic_quantize_large()
        print(f"C function test_basic_quantize_large() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_basic_quantize_large: {e}")

    # Test floating-point accumulation operations
    try:
        test_floating_point_accumulation = dll.test_floating_point_accumulation
        test_floating_point_accumulation.restype = ctypes.c_int
        result = test_floating_point_accumulation()
        print(f"C function test_floating_point_accumulation() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_floating_point_accumulation: {e}")

    # Test division operations
    try:
        test_division_operations = dll.test_division_operations
        test_division_operations.restype = ctypes.c_int
        result = test_division_operations()
        print(f"C function test_division_operations() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_division_operations: {e}")

    # Test std::vector operations
    try:
        test_vector_operations = dll.test_vector_operations
        test_vector_operations.restype = ctypes.c_int
        result = test_vector_operations()
        print(f"C function test_vector_operations() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_vector_operations: {e}")

    # Test avoiding std::min_element and std::max_element
    try:
        test_min_max_avoidance = dll.test_min_max_avoidance
        test_min_max_avoidance.restype = ctypes.c_int
        result = test_min_max_avoidance()
        print(f"C function test_min_max_avoidance() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_min_max_avoidance: {e}")

    # Test int8_t vector operations
    try:
        test_int8_vector_operations = dll.test_int8_vector_operations
        test_int8_vector_operations.restype = ctypes.c_int
        result = test_int8_vector_operations()
        print(f"C function test_int8_vector_operations() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_int8_vector_operations: {e}")

    # Test simple loop with fabs
    try:
        test_simple_loop = dll.test_simple_loop
        test_simple_loop.restype = ctypes.c_int
        result = test_simple_loop()
        print(f"C function test_simple_loop() returned: {result}, expected: 136")
    except Exception as e:
        print(f"Failed to call test_simple_loop: {e}")

    # Test nested loops
    try:
        test_nested_loops = dll.test_nested_loops
        test_nested_loops.restype = ctypes.c_int
        result = test_nested_loops()
        print(f"C function test_nested_loops() returned: {result}, expected: 16")
    except Exception as e:
        print(f"Failed to call test_nested_loops: {e}")

    # Test quantization steps
    try:
        test_quantization_steps = dll.test_quantization_steps
        test_quantization_steps.restype = ctypes.c_int
        result = test_quantization_steps()
        print(f"C function test_quantization_steps() returned: {result}, expected: 8000")
    except Exception as e:
        print(f"Failed to call test_quantization_steps: {e}")

    # Test vector allocation
    try:
        test_vector_allocation = dll.test_vector_allocation
        test_vector_allocation.restype = ctypes.c_int
        result = test_vector_allocation()
        print(f"C function test_vector_allocation() returned: {result}, expected: 16")
    except Exception as e:
        print(f"Failed to call test_vector_allocation: {e}")

    # Test vector access
    try:
        test_vector_access = dll.test_vector_access
        test_vector_access.restype = ctypes.c_int
        result = test_vector_access()
        print(f"C function test_vector_access() returned: {result}, expected: 15")
    except Exception as e:
        print(f"Failed to call test_vector_access: {e}")

    # Test weight quantization only
    try:
        test_weight_quantize_only = dll.test_weight_quantize_only
        test_weight_quantize_only.restype = ctypes.c_int
        result = test_weight_quantize_only()
        print(f"C function test_weight_quantize_only() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_weight_quantize_only: {e}")

    # Test activation quantization only
    try:
        test_activation_quantize_only = dll.test_activation_quantize_only
        test_activation_quantize_only.restype = ctypes.c_int
        result = test_activation_quantize_only()
        print(f"C function test_activation_quantize_only() returned: {result}")
    except Exception as e:
        print(f"Failed to call test_activation_quantize_only: {e}")

    print("\n=== Testing Quantization Functions ===")

    # Create test data
    rows, cols = 4, 4
    weights = np.random.randn(rows, cols).astype(np.float32)
    activations = np.random.randn(rows).astype(np.float32)

    print(f"Test weights shape: {weights.shape}")
    print(f"Test activations shape: {activations.shape}")

    # Test weight quantization
    try:
        print("Testing weight quantization...")
        quantize_weights = dll.quantize_weights_ternary_c
        quantize_weights.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32]
        quantize_weights.restype = ctypes.c_void_p

        weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ternary_ptr = quantize_weights(weights_ptr, rows, cols)

        if ternary_ptr:
            print("Weight quantization successful")

            # Get dimensions
            get_rows = dll.get_ternary_weight_rows
            get_rows.argtypes = [ctypes.c_void_p]
            get_rows.restype = ctypes.c_uint32

            get_cols = dll.get_ternary_weight_cols
            get_cols.argtypes = [ctypes.c_void_p]
            get_cols.restype = ctypes.c_uint32

            q_rows = get_rows(ternary_ptr)
            q_cols = get_cols(ternary_ptr)
            print(f"Quantized weight dimensions: {q_rows} x {q_cols}")

            # Test dequantization
            dequantize_weights = dll.dequantize_weights_c
            dequantize_weights.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32]

            output_weights = np.zeros((rows, cols), dtype=np.float32)
            output_ptr = output_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            dequantize_weights(ternary_ptr, output_ptr, rows, cols)

            print("Weight dequantization successful")

            # Compute error
            compute_error = dll.compute_quantization_error_c
            compute_error.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
            compute_error.restype = ctypes.c_float

            error = compute_error(weights_ptr, output_ptr, rows * cols)
            print(f"Weight quantization error: {error}")

            # Free memory
            free_ternary = dll.free_ternary_weight
            free_ternary.argtypes = [ctypes.c_void_p]
            free_ternary(ternary_ptr)

        else:
            print("Weight quantization failed")

    except Exception as e:
        print(f"Weight quantization test failed: {e}")
        # Don't print traceback to avoid script exit

    # Test activation quantization
    try:
        print("\nTesting activation quantization...")
        quantize_activations = dll.quantize_activations_int8_c
        quantize_activations.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        quantize_activations.restype = ctypes.c_void_p

        activations_ptr = activations.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        quantized_ptr = quantize_activations(activations_ptr, len(activations))

        if quantized_ptr:
            print("Activation quantization successful")

            # Get size
            get_size = dll.get_quantized_activation_size
            get_size.argtypes = [ctypes.c_void_p]
            get_size.restype = ctypes.c_size_t

            q_size = get_size(quantized_ptr)
            print(f"Quantized activation size: {q_size}")

            # Test dequantization
            dequantize_activations = dll.dequantize_activations_c
            dequantize_activations.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]

            output_activations = np.zeros(len(activations), dtype=np.float32)
            output_ptr = output_activations.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            dequantize_activations(quantized_ptr, output_ptr, len(activations))

            print("Activation dequantization successful")

            # Compute error
            compute_error = dll.compute_quantization_error_c
            compute_error.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
            compute_error.restype = ctypes.c_float

            error = compute_error(activations_ptr, output_ptr, len(activations))
            print(f"Activation quantization error: {error}")

            # Free memory
            free_quantized = dll.free_quantized_activation
            free_quantized.argtypes = [ctypes.c_void_p]
            free_quantized(quantized_ptr)

        else:
            print("Activation quantization failed")

    except Exception as e:
        print(f"Activation quantization test failed: {e}")
        # Don't print traceback to avoid script exit

except Exception as e:
    print(f"DLL load failed: {e}")

# Test CPU-compatible quantization functions
print("\n=== Testing CPU-Compatible Quantization Functions ===")

try:
    # Test CPU-compatible activation quantization
    try:
        print("Testing CPU-compatible activation quantization...")
        test_quantize_activations_cpu = dll.test_quantize_activations_cpu
        test_quantize_activations_cpu.restype = ctypes.c_int
        result = test_quantize_activations_cpu()
        print(f"test_quantize_activations_cpu returned: {result}")
    except Exception as e:
        print(f"test_quantize_activations_cpu failed: {e}")

    # Test full CPU-compatible quantization pipeline
    try:
        print("Testing full CPU-compatible quantization pipeline...")
        test_full_quantization_cpu = dll.test_full_quantization_cpu
        test_full_quantization_cpu.restype = ctypes.c_int
        result = test_full_quantization_cpu()
        print(f"test_full_quantization_cpu returned: {result}")
    except Exception as e:
        print(f"test_full_quantization_cpu failed: {e}")

except Exception as e:
    print(f"CPU-compatible quantization tests failed: {e}")

# Test CPU-compatible functions
print("\n=== Testing CPU-Compatible Functions ===")

# Test TernaryWeightCPU constructor
try:
    test_ternary_weight_cpu_constructor = dll.test_ternary_weight_cpu_constructor
    test_ternary_weight_cpu_constructor.restype = ctypes.c_int
    result = test_ternary_weight_cpu_constructor()
    print(f"C function test_ternary_weight_cpu_constructor() returned: {result}, expected: 42")
except Exception as e:
    print(f"Failed to call test_ternary_weight_cpu_constructor: {e}")

# Test simple quantize weights CPU
try:
    test_quantize_weights_cpu_simple = dll.test_quantize_weights_cpu_simple
    test_quantize_weights_cpu_simple.restype = ctypes.c_int
    result = test_quantize_weights_cpu_simple()
    print(f"C function test_quantize_weights_cpu_simple() returned: {result}, expected: 1")
except Exception as e:
    print(f"Failed to call test_quantize_weights_cpu_simple: {e}")

# Try Python imports (these will likely fail due to pybind11 issues)
try:
    print("\n=== Testing Python Imports ===")
    print("Attempting to import ryzen_llm_bindings directly...")
    import ryzen_llm_bindings
    print('Direct import successful')
    print('Available functions:', [attr for attr in dir(ryzen_llm_bindings) if not attr.startswith('_')])
    print(f"test_function result: {ryzen_llm_bindings.test_function()}")
except ImportError as e:
    print(f'Direct import failed: {e}')

    # Try importing as a package module
    try:
        print("Attempting to import as ryzanstein_llm.ryzen_llm_bindings...")
        import ryzanstein_llm.ryzen_llm_bindings
        print('Package import successful')
        print('Available functions:', [attr for attr in dir(ryzanstein_llm.ryzen_llm_bindings) if not attr.startswith('_')])
        print(f"test_function result: {ryzanstein_llm.ryzen_llm_bindings.test_function()}")
    except ImportError as e2:
        print(f'Package import failed: {e2}')

try:
    print("Attempting to import ryzanstein_llm package...")
    import ryzanstein_llm
    print('Package import successful')
    print('Available functions:', [attr for attr in dir(ryzanstein_llm) if not attr.startswith('_')])
except ImportError as e:
    print(f'Package import failed: {e}')