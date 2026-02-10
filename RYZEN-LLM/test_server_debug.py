"""
Debug script to isolate distributed server crash issue.
Tests server startup and request handling step by step.
"""
import sys
import asyncio
import logging

# Set up detailed logging BEFORE any other imports
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Capture all uvicorn/fastapi logs
for logger_name in ['uvicorn', 'uvicorn.access', 'uvicorn.error', 'fastapi', 'asyncio']:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

logger = logging.getLogger("DEBUG")

def test_imports():
    """Test that all imports work."""
    logger.info("=" * 60)
    logger.info("STEP 1: Testing imports...")
    
    try:
        logger.info("  Importing FastAPI components...")
        from fastapi import FastAPI
        logger.info("  ✓ FastAPI imported")
        
        logger.info("  Importing DistributedOrchestrator...")
        from src.distributed.multi_gpu_orchestrator import DistributedOrchestrator, GPUHealthMonitor, GPUStats
        logger.info("  ✓ DistributedOrchestrator, GPUHealthMonitor, GPUStats imported")
        
        logger.info("  Importing BatchEngine...")
        from src.serving.batch_engine import BatchEngine
        logger.info("  ✓ BatchEngine imported")
        
        logger.info("  Importing RequestRouter...")
        from src.api.request_router import RequestRouter
        logger.info("  ✓ RequestRouter imported")
        
        logger.info("  Importing MetricsCollector...")
        from src.monitoring.metrics import MetricsCollector
        logger.info("  ✓ MetricsCollector imported")
        
        logger.info("  Importing RyotEngine...")
        from src.core.engine.inference import RyotEngine
        logger.info("  ✓ RyotEngine imported")
        
        logger.info("STEP 1 PASSED: All imports successful")
        return True
        
    except Exception as e:
        logger.error(f"STEP 1 FAILED: Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_component_creation():
    """Test component instantiation."""
    logger.info("=" * 60)
    logger.info("STEP 2: Testing component creation...")
    
    try:
        from src.distributed.multi_gpu_orchestrator import DistributedOrchestrator, GPUHealthMonitor
        from src.serving.batch_engine import BatchEngine
        from src.api.request_router import RequestRouter
        from src.monitoring.metrics import MetricsCollector
        
        world_size = 4
        
        logger.info(f"  Creating DistributedOrchestrator(world_size={world_size})...")
        orchestrator = DistributedOrchestrator(world_size=world_size)
        logger.info(f"  ✓ DistributedOrchestrator created: {type(orchestrator)}")
        
        logger.info(f"  Creating GPUHealthMonitor(device_count={world_size})...")
        health_monitor = GPUHealthMonitor(device_count=world_size)
        logger.info(f"  ✓ GPUHealthMonitor created: {type(health_monitor)}")
        
        logger.info(f"  Creating BatchEngine(max_batch_size=16)...")
        batch_engine = BatchEngine(max_batch_size=16)
        logger.info(f"  ✓ BatchEngine created: {type(batch_engine)}")
        
        logger.info(f"  Creating RequestRouter(orchestrator, health_monitor)...")
        request_router = RequestRouter(orchestrator, health_monitor)
        logger.info(f"  ✓ RequestRouter created: {type(request_router)}")
        
        logger.info(f"  Creating MetricsCollector()...")
        metrics_collector = MetricsCollector()
        logger.info(f"  ✓ MetricsCollector created: {type(metrics_collector)}")
        
        logger.info("STEP 2 PASSED: All components created successfully")
        return True
        
    except Exception as e:
        logger.error(f"STEP 2 FAILED: Component creation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fastapi_app():
    """Test FastAPI app creation."""
    logger.info("=" * 60)
    logger.info("STEP 3: Testing FastAPI app creation...")
    
    try:
        from fastapi import FastAPI
        
        logger.info("  Creating FastAPI app...")
        app = FastAPI(title="Test App", version="1.0.0")
        logger.info(f"  ✓ FastAPI app created: {type(app)}")
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        logger.info("  ✓ Test endpoint registered")
        logger.info("STEP 3 PASSED: FastAPI app created successfully")
        return app
        
    except Exception as e:
        logger.error(f"STEP 3 FAILED: FastAPI creation error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_distributed_server_import():
    """Test importing the DistributedAPIServer."""
    logger.info("=" * 60)
    logger.info("STEP 4: Testing DistributedAPIServer import...")
    
    try:
        logger.info("  Importing DistributedAPIServer...")
        from src.api.distributed_server import DistributedAPIServer, get_distributed_server
        logger.info(f"  ✓ DistributedAPIServer imported: {DistributedAPIServer}")
        logger.info(f"  ✓ get_distributed_server imported: {get_distributed_server}")
        
        logger.info("STEP 4 PASSED: DistributedAPIServer imported successfully")
        return True
        
    except Exception as e:
        logger.error(f"STEP 4 FAILED: Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_server_creation():
    """Test creating the server instance."""
    logger.info("=" * 60)
    logger.info("STEP 5: Testing server instance creation...")
    
    try:
        from src.api.distributed_server import DistributedAPIServer
        
        logger.info("  Creating DistributedAPIServer(world_size=4, port=8080)...")
        server = DistributedAPIServer(world_size=4, host="127.0.0.1", port=8080)
        logger.info(f"  ✓ Server created: {type(server)}")
        logger.info(f"  ✓ Server.app: {type(server.app)}")
        
        logger.info("STEP 5 PASSED: Server instance created successfully")
        return server
        
    except Exception as e:
        logger.error(f"STEP 5 FAILED: Server creation error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_startup_event(server):
    """Test the startup event."""
    logger.info("=" * 60)
    logger.info("STEP 6: Testing startup event (initialization)...")
    
    try:
        logger.info("  Calling _initialize_distributed_system()...")
        await server._initialize_distributed_system()
        logger.info("  ✓ Distributed system initialized")
        
        logger.info(f"  Orchestrator: {type(server.orchestrator)}")
        logger.info(f"  Health Monitor: {type(server.health_monitor)}")
        logger.info(f"  Batch Engine: {type(server.batch_engine)}")
        logger.info(f"  Request Router: {type(server.request_router)}")
        logger.info(f"  Metrics Collector: {type(server.metrics_collector)}")
        
        logger.info("STEP 6 PASSED: Startup event successful")
        return True
        
    except Exception as e:
        logger.error(f"STEP 6 FAILED: Startup event error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_health_endpoint(server):
    """Test the health endpoint directly."""
    logger.info("=" * 60)
    logger.info("STEP 7: Testing health endpoint directly...")
    
    try:
        from fastapi.testclient import TestClient
        from httpx import ASGITransport, AsyncClient
        
        logger.info("  Creating AsyncClient...")
        async with AsyncClient(transport=ASGITransport(app=server.app), base_url="http://test") as client:
            logger.info("  Sending GET /health request...")
            response = await client.get("/health")
            logger.info(f"  ✓ Response status: {response.status_code}")
            logger.info(f"  ✓ Response body: {response.json()}")
        
        logger.info("STEP 7 PASSED: Health endpoint works")
        return True
        
    except Exception as e:
        logger.error(f"STEP 7 FAILED: Health endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests in sequence."""
    logger.info("=" * 60)
    logger.info("DISTRIBUTED SERVER DEBUG TEST SUITE")
    logger.info("=" * 60)
    
    # Step 1: Test imports
    if not test_imports():
        return False
    
    # Step 2: Test component creation
    if not test_component_creation():
        return False
    
    # Step 3: Test FastAPI app
    if not test_fastapi_app():
        return False
    
    # Step 4: Test distributed server import
    if not test_distributed_server_import():
        return False
    
    # Step 5: Test server creation
    server = test_server_creation()
    if not server:
        return False
    
    # Step 6: Test startup event
    if not await test_startup_event(server):
        return False
    
    # Step 7: Test health endpoint
    if not await test_health_endpoint(server):
        return False
    
    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED!")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting Distributed Server Debug Tests")
    print("=" * 60 + "\n")
    
    result = asyncio.run(run_all_tests())
    
    print("\n" + "=" * 60)
    if result:
        print("SUCCESS: All tests passed!")
        print("The server components work correctly in isolation.")
        print("The issue may be in the uvicorn/server lifecycle.")
    else:
        print("FAILURE: Some tests failed!")
        print("Check the error messages above to identify the issue.")
    print("=" * 60)
    
    sys.exit(0 if result else 1)
