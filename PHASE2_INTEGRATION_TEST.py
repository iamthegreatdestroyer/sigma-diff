#!/usr/bin/env python3
"""
Phase 2 Integration Test: Verify OptOptimizationOrchestrator integration into training_loop.py

Tests:
1. Syntax validation
2. Import validation 
3. Class instantiation
4. Orchestrator initialization
5. Integration point verification
"""

import sys
import ast
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('INTEGRATION_TEST')


def test_syntax_validation():
    """Test 1: Verify Python syntax is valid"""
    logger.info("═" * 70)
    logger.info("TEST 1: SYNTAX VALIDATION")
    logger.info("═" * 70)
    
    training_loop_path = Path('RYZEN-LLM/scripts/training_loop.py')
    
    try:
        with open(training_loop_path) as f:
            code = f.read()
        
        ast.parse(code)
        logger.info("✅ PASS: training_loop.py syntax is valid")
        return True
    except SyntaxError as e:
        logger.error(f"❌ FAIL: Syntax error in training_loop.py: {e}")
        return False


def test_import_orchestrator():
    """Test 2: Verify OptOptimizationOrchestrator can be imported"""
    logger.info("\n" + "═" * 70)
    logger.info("TEST 2: IMPORT ORCHESTRATOR")
    logger.info("═" * 70)
    
    sys.path.insert(0, 'RYZEN-LLM/scripts')
    
    try:
        from optimization_orchestrator import OptOptimizationOrchestrator
        logger.info("✅ PASS: OptOptimizationOrchestrator imported successfully")
        return True, OptOptimizationOrchestrator
    except ImportError as e:
        logger.error(f"❌ FAIL: Cannot import OptOptimizationOrchestrator: {e}")
        return False, None


def test_import_training_loop():
    """Test 3: Verify training_loop.py imports (without running training)"""
    logger.info("\n" + "═" * 70)
    logger.info("TEST 3: IMPORT TRAINING LOOP")
    logger.info("═" * 70)
    
    sys.path.insert(0, 'RYZEN-LLM/scripts')
    
    try:
        # This will test that all imports in training_loop.py work
        import training_loop
        logger.info("✅ PASS: training_loop module imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ FAIL: Cannot import training_loop: {e}")
        return False


def test_orchestrator_instantiation():
    """Test 4: Verify OptOptimizationOrchestrator can be instantiated"""
    logger.info("\n" + "═" * 70)
    logger.info("TEST 4: ORCHESTRATOR INSTANTIATION")
    logger.info("═" * 70)
    
    sys.path.insert(0, 'RYZEN-LLM/scripts')
    
    try:
        from optimization_orchestrator import OptOptimizationOrchestrator
        
        config = {
            'kernel_optimizer': {'tile_size': 64, 'block_size': 64},
            'semantic_compression': {'compression_ratio': 0.3, 'block_size': 64},
            'inference_scaling': {'path_selection_threshold': 0.7, 'sparsity_threshold': 0.5}
        }
        
        orchestrator = OptOptimizationOrchestrator(config)
        logger.info(f"✅ PASS: OptOptimizationOrchestrator instantiated")
        logger.info(f"  - Config keys: {list(config.keys())}")
        logger.info(f"  - State snapshots: {len(orchestrator.optimization_states)}")
        return True
    except Exception as e:
        logger.error(f"❌ FAIL: Cannot instantiate OptOptimizationOrchestrator: {e}")
        return False


def test_integration_points():
    """Test 5: Verify all 5 integration points are present in training_loop.py"""
    logger.info("\n" + "═" * 70)
    logger.info("TEST 5: INTEGRATION POINTS VERIFICATION")
    logger.info("═" * 70)
    
    with open('RYZEN-LLM/scripts/training_loop.py') as f:
        code = f.read()
    
    integration_points = {
        '1. Import statement': 'from optimization_orchestrator import OptOptimizationOrchestrator',
        '2. Initialization': 'self.orchestrator = None',
        '3. In setup_optimizations': 'self.orchestrator = OptOptimizationOrchestrator',
        '4a. Parameter adaptation': 'self.orchestrator.adapt_parameters',
        '4b. Safety gate validation': 'self.orchestrator.validate_safety_gates',
        '5. Checkpoint snapshot': 'self.orchestrator.snapshot_configuration'
    }
    
    results = {}
    for point_name, search_string in integration_points.items():
        found = search_string in code
        results[point_name] = found
        status = "✅" if found else "❌"
        logger.info(f"{status} {point_name}: {search_string}")
    
    all_present = all(results.values())
    
    if all_present:
        logger.info(f"\n✅ PASS: All 5 integration points verified")
    else:
        logger.error(f"\n❌ FAIL: Some integration points missing")
    
    return all_present


def main():
    """Run all integration tests"""
    logger.info("\n")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  PHASE 2 INTEGRATION VALIDATION - TASK 1: ORCHESTRATOR INTEGRATION  ║")
    logger.info("╚" + "═" * 68 + "╝")
    
    results = {
        'Syntax Validation': test_syntax_validation(),
        'Import Orchestrator': test_import_orchestrator()[0],
        'Import Training Loop': test_import_training_loop(),
        'Orchestrator Instantiation': test_orchestrator_instantiation(),
        'Integration Points': test_integration_points()
    }
    
    # Summary
    logger.info("\n" + "═" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("═" * 70)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    logger.info("═" * 70)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED - INTEGRATION SUCCESSFUL")
        logger.info("═" * 70)
        logger.info("\nNext Steps:")
        logger.info("  1. Run unit tests: pytest tests/test_training_loop.py -v -k 'orchestrator'")
        logger.info("  2. Run integration tests: pytest tests/test_integration.py -v")
        logger.info("  3. Run smoke test: python -c \"from training_loop import TrainingLoop; TrainingLoop()\"")
        logger.info("  4. Ready for Phase 4: Testing")
        return 0
    else:
        logger.error("INTEGRATION VALIDATION FAILED - Check errors above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
