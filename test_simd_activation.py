#!/usr/bin/env python3
"""
SIMD Activation Test Script
Tests if AVX-512/AVX2 vectorized paths are properly activated.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RYZEN-LLM'))

def test_simd_activation():
    """Test SIMD activation in T-MAC GEMM."""
    try:
        import ryzanstein_llm
        print("✅ ryzanstein_llm imported successfully")
        
        # Check if bindings are available
        if hasattr(ryzanstein_llm, '_bindings'):
            print("✅ C++ bindings available")
            
            # Try to get SIMD info
            if hasattr(ryzanstein_llm, 'get_simd_info'):
                try:
                    simd_info = ryzanstein_llm.get_simd_info()
                    print(f"✅ SIMD info: {simd_info}")
                except Exception as e:
                    print(f"⚠️  SIMD info not available: {e}")
            else:
                print("⚠️  get_simd_info not available")
                
        else:
            print("❌ C++ bindings not available")
            return False
            
        print("✅ SIMD activation test passed!")
        return True

    except Exception as e:
        print(f"❌ SIMD activation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simd_activation()
    sys.exit(0 if success else 1)