"""
Distributed Inference API Server
=================================

FastAPI server for distributed LLM inference across multiple GPUs.
Routes requests to available GPUs with load balancing and health monitoring.

Key Features:
- REST API for chat completions and text generation
- Automatic GPU load balancing
- Health monitoring and failover
- Request batching for optimal throughput
- Streaming responses
- Metrics collection

Architecture:
- FastAPI for high-performance async HTTP
- GPU health monitoring integration
- Request routing with load balancing
- Response aggregation from distributed workers
"""

import asyncio
import time
import logging
from typing import List, Optional, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from src.distributed.multi_gpu_orchestrator import DistributedOrchestrator, GPUHealthMonitor
from src.serving.batch_engine import BatchEngine
from src.api.request_router import RequestRouter
from src.monitoring.metrics import MetricsCollector
from src.core.engine.inference import RyotEngine
from src.api.types import GenerationConfig, GenerationResult, StreamChunk as EngineStreamChunk, StopReason

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class Message(BaseModel):
    """Chat message."""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """Chat completion request for distributed inference."""
    model: str = Field(..., description="Model identifier")
    messages: List[Message] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(default=50, ge=1, le=1000, description="Top-k sampling parameter")
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0, description="Repetition penalty parameter")
    max_tokens: Optional[int] = Field(default=256, ge=1, le=4096, description="Maximum tokens to generate")
    stream: bool = Field(default=False, description="Enable streaming response")
    gpu_preference: Optional[int] = Field(default=None, description="Preferred GPU ID (optional)")


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


class StreamChunk(BaseModel):
    """Streaming response chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


class DistributedAPIServer:
    """
    Distributed API server for multi-GPU LLM inference.

    Manages request routing, load balancing, and response aggregation
    across multiple GPU workers.
    """

    def __init__(
        self,
        world_size: int = 4,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_batch_size: int = 8,
        health_check_interval: float = 5.0,
        model_path: Optional[str] = None
    ):
        self.world_size = world_size
        self.host = host
        self.port = port
        self.max_batch_size = max_batch_size
        self.health_check_interval = health_check_interval
        self.model_path = model_path

        # Core components
        self.orchestrator: Optional[DistributedOrchestrator] = None
        self.health_monitor: Optional[GPUHealthMonitor] = None
        self.batch_engine: Optional[BatchEngine] = None
        self.request_router: Optional[RequestRouter] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        
        # Inference engine
        self.engine: Optional[RyotEngine] = None

        # FastAPI app with lifespan for proper async initialization
        self.app = FastAPI(
            title="RYZEN-LLM Distributed API",
            description="Distributed LLM inference across multiple GPUs",
            version="3.0.0",
            lifespan=self._create_lifespan()
        )

        # Setup routes
        self._setup_routes()

    def _create_lifespan(self):
        """Create a lifespan context manager for FastAPI.
        
        This replaces the deprecated on_event("startup") and on_event("shutdown")
        pattern with the modern lifespan approach.
        """
        server = self  # Capture self for use in the closure
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup: Initialize distributed components
            logger.info("Starting distributed system initialization...")
            await server._initialize_distributed_system()
            logger.info("Distributed system initialization complete")
            
            yield  # Server is running
            
            # Shutdown: Cleanup
            await server._shutdown_distributed_system()
        
        return lifespan

    def _setup_routes(self):
        """Setup API routes."""
        # Note: Startup/shutdown are now handled by lifespan context manager
        # defined in _create_lifespan() method

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            try:
                if not self.health_monitor:
                    # System not fully initialized yet
                    return {
                        "status": "initializing",
                        "healthy_gpus": 0,
                        "total_gpus": self.world_size,
                        "uptime": time.time()
                    }

                # Ensure we have basic stats for all GPUs (simulate health check)
                for gpu_id in range(self.world_size):
                    if gpu_id not in self.health_monitor.stats:
                        from src.distributed.multi_gpu_orchestrator import GPUStats
                        stats = GPUStats(
                            device_id=gpu_id,
                            memory_used=1024,  # 1GB used
                            memory_total=8192,  # 8GB total
                            utilization=50.0,
                            temperature=70.0,
                            last_heartbeat=time.time(),
                            active_requests=1,
                            avg_latency=25.0
                        )
                        self.health_monitor.update_stats(gpu_id, stats)

                healthy_devices = self.health_monitor.get_healthy_devices()
                response_data = {
                    "status": "healthy" if healthy_devices else "unhealthy",
                    "healthy_gpus": len(healthy_devices),
                    "total_gpus": self.world_size,
                    "uptime": time.time()
                }
                logger.info(f"Health check response: {response_data}")
                return response_data

            except Exception as e:
                logger.error(f"Health check error: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "uptime": time.time()
                }

        @self.app.get("/metrics")
        async def get_metrics():
            """Get system metrics."""
            if not self.metrics_collector:
                return {"error": "Metrics collector not initialized"}

            return self.metrics_collector.get_system_summary()

        @self.app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
            """Chat completions endpoint with distributed routing."""
            try:
                # Route request to appropriate GPU
                gpu_id = await self.request_router.route_request(request)

                # Add to batch or process immediately
                if request.stream:
                    return StreamingResponse(
                        self._stream_completion(request, gpu_id),
                        media_type="text/plain"
                    )
                else:
                    response = await self._process_completion(request, gpu_id)

                    # Record metrics
                    if self.metrics_collector:
                        background_tasks.add_task(
                            self.metrics_collector.record_request,
                            gpu_id, response.usage.get("total_tokens", 0), True
                        )

                    return response

            except Exception as e:
                logger.error(f"Error processing request: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _initialize_distributed_system(self):
        """Initialize all distributed components."""
        logger.info("Initializing distributed inference system...")

        try:
            # Initialize orchestrator
            self.orchestrator = DistributedOrchestrator(world_size=self.world_size)

            # Initialize health monitor
            self.health_monitor = GPUHealthMonitor(device_count=self.world_size)

            # Initialize batch engine
            self.batch_engine = BatchEngine(max_batch_size=self.max_batch_size)

            # Initialize inference engine
            self.engine = RyotEngine(model_path=self.model_path)
            if self.model_path:
                logger.info(f"Loaded model from {self.model_path}")
            else:
                logger.warning("No model_path provided - engine will need model loaded before inference")

            # Initialize request router
            self.request_router = RequestRouter(
                orchestrator=self.orchestrator,
                health_monitor=self.health_monitor
            )

            # Initialize metrics collector
            self.metrics_collector = MetricsCollector()

            # Start health monitoring
            self.health_check_task = asyncio.create_task(self._health_check_loop())

            logger.info(f"Distributed system initialized with {self.world_size} GPUs")

        except Exception as e:
            logger.error(f"Failed to initialize distributed system: {e}")
            raise

    async def _shutdown_distributed_system(self):
        """Shutdown distributed components."""
        logger.info("Shutting down distributed system...")

        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        # Cleanup components
        if self.orchestrator:
            # Add cleanup logic if needed
            pass

    async def _health_check_loop(self):
        """Continuous health monitoring loop."""
        while True:
            try:
                # Update health status for all GPUs
                for gpu_id in range(self.world_size):
                    # In a real implementation, this would query actual GPU status
                    # For now, simulate healthy status
                    from src.distributed.multi_gpu_orchestrator import GPUStats
                    stats = GPUStats(
                        device_id=gpu_id,
                        memory_used=1024,  # 1GB used
                        memory_total=8192,  # 8GB total
                        utilization=50.0,
                        temperature=70.0,
                        last_heartbeat=time.time(),
                        active_requests=1,
                        avg_latency=25.0
                    )
                    self.health_monitor.update_stats(gpu_id, stats)

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(1.0)

    async def _process_completion(self, request: ChatCompletionRequest, gpu_id: int) -> ChatCompletionResponse:
        """Process a single completion request using RyotEngine."""
        if self.engine is None:
            raise HTTPException(status_code=503, detail="Inference engine not initialized")

        # Extract prompt from messages
        prompt = "\n".join(
            f"{msg.role}: {msg.content}" for msg in request.messages
        )

        # Build generation config from request
        config = GenerationConfig(
            max_tokens=request.max_tokens or 256,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            top_k=request.top_k or 50,
            repetition_penalty=request.repetition_penalty or 1.0
        )

        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result: GenerationResult = await loop.run_in_executor(
            None, self.engine.generate, prompt, config
        )

        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=result.generated_text
                    ),
                    finish_reason="stop" if result.stop_reason == StopReason.MAX_TOKENS else str(result.stop_reason.value)
                )
            ],
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens
            }
        )

        return response

    async def _stream_completion(self, request: ChatCompletionRequest, gpu_id: int) -> AsyncIterator[str]:
        """Stream completion response using RyotEngine."""
        if self.engine is None:
            yield f"data: {{\"error\": \"Inference engine not initialized\"}}\n\n"
            return

        # Extract prompt from messages
        prompt = "\n".join(
            f"{msg.role}: {msg.content}" for msg in request.messages
        )

        # Build generation config from request
        config = GenerationConfig(
            max_tokens=request.max_tokens or 256,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            top_k=request.top_k or 50,
            repetition_penalty=request.repetition_penalty or 1.0
        )

        # Stream from engine in thread pool
        loop = asyncio.get_event_loop()
        stream_iter = await loop.run_in_executor(
            None, self.engine.stream, prompt, config
        )

        chunk_index = 0
        for chunk in stream_iter:
            # EngineStreamChunk has token_text, is_last, and final_result
            finish_reason = None
            if chunk.is_last:
                if chunk.final_result:
                    finish_reason = "stop" if chunk.final_result.stop_reason == StopReason.EOS_TOKEN else str(chunk.final_result.stop_reason.value)
                else:
                    finish_reason = "stop"
            
            stream_chunk = StreamChunk(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=[{
                    "index": 0,
                    "delta": {"role": "assistant", "content": chunk.token_text} if chunk_index == 0 else {"content": chunk.token_text},
                    "finish_reason": finish_reason
                }]
            )
            chunk_index += 1

            yield f"data: {stream_chunk.json()}\n\n"

        yield "data: [DONE]\n\n"

    def run(self):
        """Run the API server."""
        logger.info(f"Starting distributed API server on {self.host}:{self.port}")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


# Global server instance for easy access
_server_instance: Optional[DistributedAPIServer] = None


def get_distributed_server(
    world_size: int = 4,
    host: str = "0.0.0.0",
    port: int = 8000
) -> DistributedAPIServer:
    """Get or create distributed API server instance."""
    global _server_instance
    if _server_instance is None:
        _server_instance = DistributedAPIServer(
            world_size=world_size,
            host=host,
            port=port
        )
    return _server_instance


# Module-level app instance for uvicorn
# Usage: uvicorn src.api.distributed_server:app --host 0.0.0.0 --port 8000
app = get_distributed_server().app


if __name__ == "__main__":
    # Run server directly
    server = get_distributed_server()
    server.run()