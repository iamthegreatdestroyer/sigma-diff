"""
RYZEN-LLM Python Bindings

High-performance LLM inference with 1.58-bit quantization (BitNet b1.58)
optimized for AMD Ryzen processors.
"""

try:
    from . import ryzen_llm_bindings as _bindings
    
    # Re-export key classes and functions
    # Core components
    if hasattr(_bindings, 'BitNetTokenizer'):
        BitNetTokenizer = _bindings.BitNetTokenizer
    if hasattr(_bindings, 'BitNetModel'):
        BitNetModel = _bindings.BitNetModel
    if hasattr(_bindings, 'BitNetInference'):
        BitNetInference = _bindings.BitNetInference
    if hasattr(_bindings, 'BitNetPipeline'):
        BitNetPipeline = _bindings.BitNetPipeline
    
    # T-MAC components
    if hasattr(_bindings, 'LutGemm'):
        LutGemm = _bindings.LutGemm
    if hasattr(_bindings, 'TernaryWeight'):
        TernaryWeight = _bindings.TernaryWeight
    
    # Utility functions
    if hasattr(_bindings, 'get_version'):
        get_version = _bindings.get_version
    if hasattr(_bindings, 'get_simd_info'):
        get_simd_info = _bindings.get_simd_info
    
    __version__ = "2.0.0"
    __all__ = ['_bindings', '__version__']
    
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import ryzen_llm_bindings: {e}")
    __version__ = "2.0.0-no-bindings"
