"""
gRPC Server for Distributed Inference.

Provides high-performance gRPC interface for:
- Inference requests
- Model management
- Health checks
- Streaming responses

Sprint 3.4 - Serving Layer
Created: 2026-01-06
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, AsyncIterator, Callable
import uuid

logger = logging.getLogger(__name__)


class GRPCStatusCode(Enum):
    """gRPC status codes."""
    OK = 0
    CANCELLED = 1
    UNKNOWN = 2
    INVALID_ARGUMENT = 3
    DEADLINE_EXCEEDED = 4
    NOT_FOUND = 5
    ALREADY_EXISTS = 6
    PERMISSION_DENIED = 7
    RESOURCE_EXHAUSTED = 8
    FAILED_PRECONDITION = 9
    ABORTED = 10
    OUT_OF_RANGE = 11
    UNIMPLEMENTED = 12
    INTERNAL = 13
    UNAVAILABLE = 14
    DATA_LOSS = 15
    UNAUTHENTICATED = 16


@dataclass
class GRPCMetadata:
    """gRPC request/response metadata."""
    values: Dict[str, str] = field(default_factory=dict)
    
    def get(self, key: str, default: str = "") -> str:
        return self.values.get(key, default)
    
    def set(self, key: str, value: str) -> None:
        self.values[key] = value


@dataclass
class GRPCRequest:
    """Generic gRPC request wrapper."""
    method: str
    data: Any
    metadata: GRPCMetadata = field(default_factory=GRPCMetadata)
    deadline: Optional[float] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class GRPCResponse:
    """Generic gRPC response wrapper."""
    data: Any
    status: GRPCStatusCode = GRPCStatusCode.OK
    message: str = ""
    metadata: GRPCMetadata = field(default_factory=GRPCMetadata)
    latency_ms: float = 0.0


# ============= Inference Service Messages =============

@dataclass
class GenerateRequest:
    """Request for text generation."""
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    model_id: Optional[str] = None
    stream: bool = False
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class GenerateResponse:
    """Response for text generation."""
    request_id: str
    text: str
    token_ids: List[int]
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str
    latency_ms: float


@dataclass
class StreamToken:
    """Single token in streaming response."""
    token_id: int
    token_text: str
    is_final: bool = False
    finish_reason: Optional[str] = None


@dataclass
class BatchGenerateRequest:
    """Batch generation request."""
    requests: List[GenerateRequest]


@dataclass
class BatchGenerateResponse:
    """Batch generation response."""
    responses: List[GenerateResponse]
    total_latency_ms: float


@dataclass
class ModelInfoRequest:
    """Request for model information."""
    model_id: str


@dataclass
class ModelInfoResponse:
    """Model information response."""
    model_id: str
    name: str
    version: str
    parameters: int
    context_length: int
    loaded: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckRequest:
    """Health check request."""
    service: str = ""


@dataclass
class HealthCheckResponse:
    """Health check response."""
    status: str  # "SERVING", "NOT_SERVING", "UNKNOWN"
    details: Dict[str, str] = field(default_factory=dict)


# ============= Service Interfaces =============

class InferenceServicer(ABC):
    """Abstract inference service interface."""
    
    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self, request: GenerateRequest
    ) -> AsyncIterator[StreamToken]:
        """Stream generated tokens."""
        pass
    
    @abstractmethod
    async def batch_generate(
        self, request: BatchGenerateRequest
    ) -> BatchGenerateResponse:
        """Generate for batch of prompts."""
        pass


class ModelServicer(ABC):
    """Abstract model management service interface."""
    
    @abstractmethod
    async def get_model_info(self, request: ModelInfoRequest) -> ModelInfoResponse:
        """Get model information."""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[ModelInfoResponse]:
        """List all available models."""
        pass
    
    @abstractmethod
    async def load_model(self, model_id: str) -> bool:
        """Load a model."""
        pass
    
    @abstractmethod
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model."""
        pass


class HealthServicer(ABC):
    """Abstract health service interface."""
    
    @abstractmethod
    async def check(self, request: HealthCheckRequest) -> HealthCheckResponse:
        """Check service health."""
        pass


# ============= gRPC Server Implementation =============

@dataclass
class GRPCServerConfig:
    """gRPC server configuration."""
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10
    max_message_size: int = 100 * 1024 * 1024  # 100MB
    keepalive_time_ms: int = 10000
    keepalive_timeout_ms: int = 5000
    max_concurrent_rpcs: int = 100
    enable_reflection: bool = True


class GRPCServer:
    """
    gRPC server for inference services.
    
    This is a simulation/interface - in production would use grpcio.
    """
    
    def __init__(self, config: Optional[GRPCServerConfig] = None):
        self.config = config or GRPCServerConfig()
        self.inference_servicer: Optional[InferenceServicer] = None
        self.model_servicer: Optional[ModelServicer] = None
        self.health_servicer: Optional[HealthServicer] = None
        self.interceptors: List[Callable] = []
        self.running = False
        self.stats = GRPCServerStats()
        self._handlers: Dict[str, Callable] = {}
    
    def register_inference_service(self, servicer: InferenceServicer) -> None:
        """Register inference service."""
        self.inference_servicer = servicer
        self._handlers["inference.Generate"] = self._handle_generate
        self._handlers["inference.GenerateStream"] = self._handle_generate_stream
        self._handlers["inference.BatchGenerate"] = self._handle_batch_generate
        logger.info("Registered inference service")
    
    def register_model_service(self, servicer: ModelServicer) -> None:
        """Register model management service."""
        self.model_servicer = servicer
        self._handlers["model.GetInfo"] = self._handle_get_model_info
        self._handlers["model.List"] = self._handle_list_models
        logger.info("Registered model service")
    
    def register_health_service(self, servicer: HealthServicer) -> None:
        """Register health service."""
        self.health_servicer = servicer
        self._handlers["grpc.health.v1.Health/Check"] = self._handle_health_check
        logger.info("Registered health service")
    
    def add_interceptor(self, interceptor: Callable) -> None:
        """Add a gRPC interceptor."""
        self.interceptors.append(interceptor)
    
    async def start(self) -> None:
        """Start the gRPC server."""
        self.running = True
        logger.info(
            f"gRPC server started on {self.config.host}:{self.config.port}"
        )
    
    async def stop(self, grace_period: float = 5.0) -> None:
        """Stop the gRPC server."""
        self.running = False
        await asyncio.sleep(0.1)  # Allow pending requests to complete
        logger.info("gRPC server stopped")
    
    async def handle_request(self, request: GRPCRequest) -> GRPCResponse:
        """Handle a gRPC request."""
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            # Check deadline
            if request.deadline and time.time() > request.deadline:
                self.stats.deadline_exceeded += 1
                return GRPCResponse(
                    data=None,
                    status=GRPCStatusCode.DEADLINE_EXCEEDED,
                    message="Request deadline exceeded"
                )
            
            # Find handler
            handler = self._handlers.get(request.method)
            if not handler:
                self.stats.unimplemented += 1
                return GRPCResponse(
                    data=None,
                    status=GRPCStatusCode.UNIMPLEMENTED,
                    message=f"Method not found: {request.method}"
                )
            
            # Apply interceptors
            for interceptor in self.interceptors:
                request = await interceptor(request)
            
            # Execute handler
            response = await handler(request)
            
            latency_ms = (time.time() - start_time) * 1000
            response.latency_ms = latency_ms
            self.stats.record_success(latency_ms)
            
            return response
            
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"gRPC handler error: {e}")
            return GRPCResponse(
                data=None,
                status=GRPCStatusCode.INTERNAL,
                message=str(e)
            )
    
    async def _handle_generate(self, request: GRPCRequest) -> GRPCResponse:
        """Handle generate request."""
        if not self.inference_servicer:
            return GRPCResponse(
                data=None,
                status=GRPCStatusCode.UNAVAILABLE,
                message="Inference service not available"
            )
        
        gen_request = request.data
        if not isinstance(gen_request, GenerateRequest):
            gen_request = GenerateRequest(**request.data)
        
        response = await self.inference_servicer.generate(gen_request)
        return GRPCResponse(data=response, status=GRPCStatusCode.OK)
    
    async def _handle_generate_stream(
        self, request: GRPCRequest
    ) -> AsyncIterator[GRPCResponse]:
        """Handle streaming generate request."""
        if not self.inference_servicer:
            yield GRPCResponse(
                data=None,
                status=GRPCStatusCode.UNAVAILABLE,
                message="Inference service not available"
            )
            return
        
        gen_request = request.data
        if not isinstance(gen_request, GenerateRequest):
            gen_request = GenerateRequest(**request.data)
        
        async for token in self.inference_servicer.generate_stream(gen_request):
            yield GRPCResponse(data=token, status=GRPCStatusCode.OK)
    
    async def _handle_batch_generate(self, request: GRPCRequest) -> GRPCResponse:
        """Handle batch generate request."""
        if not self.inference_servicer:
            return GRPCResponse(
                data=None,
                status=GRPCStatusCode.UNAVAILABLE,
                message="Inference service not available"
            )
        
        batch_request = request.data
        if not isinstance(batch_request, BatchGenerateRequest):
            batch_request = BatchGenerateRequest(**request.data)
        
        response = await self.inference_servicer.batch_generate(batch_request)
        return GRPCResponse(data=response, status=GRPCStatusCode.OK)
    
    async def _handle_get_model_info(self, request: GRPCRequest) -> GRPCResponse:
        """Handle get model info request."""
        if not self.model_servicer:
            return GRPCResponse(
                data=None,
                status=GRPCStatusCode.UNAVAILABLE,
                message="Model service not available"
            )
        
        info_request = request.data
        if not isinstance(info_request, ModelInfoRequest):
            info_request = ModelInfoRequest(**request.data)
        
        response = await self.model_servicer.get_model_info(info_request)
        return GRPCResponse(data=response, status=GRPCStatusCode.OK)
    
    async def _handle_list_models(self, request: GRPCRequest) -> GRPCResponse:
        """Handle list models request."""
        if not self.model_servicer:
            return GRPCResponse(
                data=None,
                status=GRPCStatusCode.UNAVAILABLE,
                message="Model service not available"
            )
        
        models = await self.model_servicer.list_models()
        return GRPCResponse(data=models, status=GRPCStatusCode.OK)
    
    async def _handle_health_check(self, request: GRPCRequest) -> GRPCResponse:
        """Handle health check request."""
        if not self.health_servicer:
            return GRPCResponse(
                data=HealthCheckResponse(status="SERVING"),
                status=GRPCStatusCode.OK
            )
        
        health_request = request.data
        if not isinstance(health_request, HealthCheckRequest):
            health_request = HealthCheckRequest(**request.data) if request.data else HealthCheckRequest()
        
        response = await self.health_servicer.check(health_request)
        return GRPCResponse(data=response, status=GRPCStatusCode.OK)


@dataclass
class GRPCServerStats:
    """Statistics for gRPC server."""
    total_requests: int = 0
    successful_requests: int = 0
    errors: int = 0
    deadline_exceeded: int = 0
    unimplemented: int = 0
    total_latency_ms: float = 0.0
    
    def record_success(self, latency_ms: float) -> None:
        """Record successful request."""
        self.successful_requests += 1
        self.total_latency_ms += latency_ms
    
    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests
    
    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.errors / self.total_requests


# ============= Default Implementations =============

class DefaultInferenceServicer(InferenceServicer):
    """Default inference servicer for testing."""
    
    def __init__(self, pipeline=None):
        self.pipeline = pipeline
    
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        start_time = time.time()
        
        # Simulated generation
        generated_text = f"Response to: {request.prompt[:50]}..."
        generated_tokens = len(generated_text.split())
        
        latency_ms = (time.time() - start_time) * 1000
        
        return GenerateResponse(
            request_id=request.request_id,
            text=generated_text,
            token_ids=list(range(generated_tokens)),
            prompt_tokens=len(request.prompt.split()),
            generated_tokens=generated_tokens,
            finish_reason="stop",
            latency_ms=latency_ms
        )
    
    async def generate_stream(
        self, request: GenerateRequest
    ) -> AsyncIterator[StreamToken]:
        words = f"Response to: {request.prompt[:30]}...".split()
        
        for i, word in enumerate(words):
            yield StreamToken(
                token_id=i,
                token_text=word + " ",
                is_final=(i == len(words) - 1),
                finish_reason="stop" if i == len(words) - 1 else None
            )
            await asyncio.sleep(0.01)  # Simulate token generation time
    
    async def batch_generate(
        self, request: BatchGenerateRequest
    ) -> BatchGenerateResponse:
        start_time = time.time()
        
        responses = []
        for req in request.requests:
            resp = await self.generate(req)
            responses.append(resp)
        
        return BatchGenerateResponse(
            responses=responses,
            total_latency_ms=(time.time() - start_time) * 1000
        )


class DefaultHealthServicer(HealthServicer):
    """Default health servicer."""
    
    def __init__(self):
        self.services_status: Dict[str, str] = {}
    
    def set_service_status(self, service: str, status: str) -> None:
        """Set status for a service."""
        self.services_status[service] = status
    
    async def check(self, request: HealthCheckRequest) -> HealthCheckResponse:
        if request.service:
            status = self.services_status.get(request.service, "UNKNOWN")
        else:
            # Overall status
            if all(s == "SERVING" for s in self.services_status.values()):
                status = "SERVING"
            elif any(s == "NOT_SERVING" for s in self.services_status.values()):
                status = "NOT_SERVING"
            else:
                status = "SERVING"  # Default to serving
        
        return HealthCheckResponse(
            status=status,
            details=dict(self.services_status)
        )


def create_grpc_server(
    config: Optional[GRPCServerConfig] = None,
    pipeline=None
) -> GRPCServer:
    """Create and configure a gRPC server."""
    server = GRPCServer(config)
    
    # Register default services
    server.register_inference_service(DefaultInferenceServicer(pipeline))
    server.register_health_service(DefaultHealthServicer())
    
    return server
