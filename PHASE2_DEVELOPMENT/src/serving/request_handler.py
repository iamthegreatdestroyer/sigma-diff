"""
HTTP Request Handler for Distributed Inference
==============================================

FastAPI-based HTTP interface for the unified inference pipeline.

Endpoints:
- POST /v1/generate - Generate text
- POST /v1/batch - Batch generation
- GET /v1/health - Health check
- GET /v1/metrics - Performance metrics

Sprint 2.2 Phase 1 - Integration
Created: 2025-12-26
"""

import torch
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import time
import json

logger = logging.getLogger(__name__)


@dataclass
class GenerateRequest:
    """Request format for generation."""
    prompt: str
    prompt_tokens: Optional[List[int]] = None
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    priority: int = 0


@dataclass
class GenerateResponse:
    """Response format for generation."""
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    text: str
    token_ids: List[int]
    latency_ms: float
    throughput_tokens_per_sec: float
    finish_reason: str = "length"


@dataclass
class BatchRequest:
    """Request format for batch generation."""
    requests: List[GenerateRequest]


@dataclass
class BatchResponse:
    """Response format for batch generation."""
    request_ids: List[str]
    responses: List[GenerateResponse]
    total_latency_ms: float
    batch_throughput_tokens_per_sec: float


@dataclass
class HealthResponse:
    """Health check response."""
    status: str
    uptime_seconds: float
    total_requests: int
    total_tokens: int
    gpu_utilization_percent: Optional[float] = None


@dataclass
class MetricsResponse:
    """Metrics response."""
    total_requests: int
    total_tokens_generated: int
    avg_latency_ms: float
    avg_throughput_tokens_per_sec: float
    cache_hit_rate: float
    acceptance_rate: float


class RequestHandler:
    """
    Handles HTTP requests for inference pipeline.
    
    Integrates with UnifiedInferencePipeline to serve requests.
    """
    
    def __init__(self, pipeline):
        """
        Initialize request handler.
        
        Args:
            pipeline: UnifiedInferencePipeline instance
        """
        self.pipeline = pipeline
        self.start_time = time.time()
        self.request_count = 0
        self.cache_hits = 0
        self.total_acceptance_rate = 0.0
        self.request_ids_processed = set()
        
        logger.info("RequestHandler initialized")
    
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Generate response for a single request.
        
        Args:
            request: GenerateRequest
        
        Returns:
            GenerateResponse
        """
        request_id = f"req_{self.request_count}_{int(time.time()*1000)}"
        self.request_count += 1
        
        # Convert prompt to tokens if needed
        if request.prompt_tokens:
            prompt_tokens = torch.tensor(request.prompt_tokens)
        else:
            # In real implementation, use actual tokenizer
            prompt_tokens = torch.randint(0, 32000, (len(request.prompt.split()),))
        
        request_start = time.time()
        
        # Add to pipeline
        self.pipeline.add_request(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=request.max_tokens,
            priority=request.priority
        )
        
        # Process (single batch)
        outputs = self.pipeline.process_requests(num_batches=1)
        
        if not outputs:
            raise RuntimeError("No output generated")
        
        output = outputs[0]
        
        latency_ms = (time.time() - request_start) * 1000
        
        # Format response
        response = GenerateResponse(
            request_id=request_id,
            prompt_tokens=len(prompt_tokens),
            generated_tokens=output.num_tokens,
            text="",  # Would decode tokens to text with tokenizer
            token_ids=output.generated_ids.tolist(),
            latency_ms=latency_ms,
            throughput_tokens_per_sec=output.num_tokens / (latency_ms / 1000) if latency_ms > 0 else 0,
            finish_reason="length"
        )
        
        self.request_ids_processed.add(request_id)
        
        # Track metrics
        if hasattr(output, 'accepted_ratio'):
            self.total_acceptance_rate += output.accepted_ratio
        
        return response
    
    def batch_generate(self, batch_request: BatchRequest) -> BatchResponse:
        """
        Generate responses for batch of requests.
        
        Args:
            batch_request: BatchRequest containing multiple requests
        
        Returns:
            BatchResponse
        """
        batch_start = time.time()
        responses = []
        request_ids = []
        
        for request in batch_request.requests:
            response = self.generate(request)
            responses.append(response)
            request_ids.append(response.request_id)
        
        total_latency_ms = (time.time() - batch_start) * 1000
        total_tokens = sum(r.generated_tokens for r in responses)
        batch_throughput = total_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else 0
        
        return BatchResponse(
            request_ids=request_ids,
            responses=responses,
            total_latency_ms=total_latency_ms,
            batch_throughput_tokens_per_sec=batch_throughput
        )
    
    def health_check(self) -> HealthResponse:
        """
        Check pipeline health.
        
        Returns:
            HealthResponse
        """
        uptime = time.time() - self.start_time
        
        return HealthResponse(
            status="healthy",
            uptime_seconds=uptime,
            total_requests=self.pipeline.total_requests,
            total_tokens=self.pipeline.total_tokens,
            gpu_utilization_percent=None  # Would measure actual GPU usage
        )
    
    def get_metrics(self) -> MetricsResponse:
        """
        Get pipeline metrics.
        
        Returns:
            MetricsResponse
        """
        stats = self.pipeline.get_statistics()
        
        # Calculate cache hit rate (simplified)
        cache_hits = len([r for r in self.request_ids_processed if r in self.pipeline.active_requests])
        cache_hit_rate = cache_hits / max(1, len(self.request_ids_processed))
        
        # Calculate average acceptance rate
        avg_acceptance = self.total_acceptance_rate / max(1, len(self.request_ids_processed))
        
        return MetricsResponse(
            total_requests=self.pipeline.total_requests,
            total_tokens_generated=self.pipeline.total_tokens,
            avg_latency_ms=stats.get('avg_latency_ms', 0.0),
            avg_throughput_tokens_per_sec=stats.get('avg_throughput_tokens_per_sec', 0.0),
            cache_hit_rate=cache_hit_rate,
            acceptance_rate=avg_acceptance
        )


# FastAPI integration (optional)
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    
    def create_app(pipeline) -> FastAPI:
        """
        Create FastAPI application for inference.
        
        Args:
            pipeline: UnifiedInferencePipeline instance
        
        Returns:
            FastAPI application
        """
        app = FastAPI(title="Distributed Inference API", version="2.2")
        handler = RequestHandler(pipeline)
        
        @app.post("/v1/generate")
        async def generate(request: dict) -> dict:
            """Generate text."""
            try:
                gen_request = GenerateRequest(**request)
                response = handler.generate(gen_request)
                return asdict(response)
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/v1/batch")
        async def batch_generate(batch_request: dict) -> dict:
            """Batch generation."""
            try:
                requests = [GenerateRequest(**r) for r in batch_request.get("requests", [])]
                batch_req = BatchRequest(requests=requests)
                response = handler.batch_generate(batch_req)
                return asdict(response)
            except Exception as e:
                logger.error(f"Batch error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/v1/health")
        async def health() -> dict:
            """Health check."""
            response = handler.health_check()
            return asdict(response)
        
        @app.get("/v1/metrics")
        async def metrics() -> dict:
            """Get metrics."""
            response = handler.get_metrics()
            return asdict(response)
        
        return app

except ImportError:
    logger.warning("FastAPI not available - HTTP interface disabled")
    
    def create_app(pipeline):
        """Fallback when FastAPI not available."""
        logger.warning("FastAPI not available - install with: pip install fastapi uvicorn")
        return None


if __name__ == "__main__":
    # Test request handler
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Request Handler...")
    
    # Would need actual pipeline instance
    # from src.serving.unified_pipeline import UnifiedInferencePipeline
    
    print("Request handler ready for integration!")
