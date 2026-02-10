import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path('build/python')))
sys.path.insert(0, str(Path('build/python/ryzanstein_llm')))

try:
    import test_module
    print('✓ Import successful')
    print('hasattr test_function:', hasattr(test_module, 'test_function'))

    if hasattr(test_module, 'test_function'):
        result = test_module.test_function()
        print('✓ test_function result:', result)
    else:
        print('Available attrs:', [attr for attr in dir(test_module) if not attr.startswith('_')])

except Exception as e:
    print('❌ Error:', e)
    import traceback
    traceback.print_exc()