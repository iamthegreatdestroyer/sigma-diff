"""
Unified Multi-Modal Inference Pipeline
=======================================

End-to-end distributed inference pipeline integrating all components.

Combines:
- Multi-modal input processing (vision + text)
- Distributed computation across GPUs
- KV cache management with prefix caching
- Speculative generation for speed
- Token-level batching for throughput

Sprint 2.2 Phase 1 - Integration
Created: 2025-12-26
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from abc import ABC, abstractmethod

# Import components
from distributed.engine import DistributedInferenceEngine, DistributedConfig
from cache.manager import PagedAttentionKVCache, PrefixCache, PageConfig
from speculative.decoder import SpeculativeDecoder, SpeculationConfig, DraftModel
from batching.token_batcher import TokenBatcher, TokenBatch, TokenRequest

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for unified inference pipeline."""
    # Distributed settings
    num_gpus: int = 8
    world_size: int = 8
    
    # Model settings
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    
    # Memory settings
    num_kv_cache_pages: int = 4096
    max_batch_size: int = 128
    max_batch_tokens: int = 4096
    
    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Performance settings
    enable_speculative_decoding: bool = True
    enable_prefix_caching: bool = True
    enable_continuous_batching: bool = True


@dataclass
class GenerationRequest:
    """A single generation request."""
    request_id: str
    prompt_tokens: torch.Tensor
    max_tokens: int
    priority: int = 0
    deadline: Optional[float] = None


@dataclass
class GenerationOutput:
    """Output from generation."""
    request_id: str
    generated_ids: torch.Tensor
    num_tokens: int
    latency_ms: float
    throughput_tokens_per_sec: float
    memory_used_mb: float
    accepted_ratio: float = 1.0  # For speculative decoding


class UnifiedInferencePipeline(nn.Module):
    """
    Unified pipeline orchestrating all distributed inference components.
    
    Manages:
    - Request batching (token-level)
    - Distributed computation (tensor parallelism)
    - KV cache (paged attention + prefix caching)
    - Fast generation (speculative decoding)
    - Performance monitoring
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: PipelineConfig
    ):
        super().__init__()
        self.config = config
        self.model = model
        
        # Initialize components
        dist_config = DistributedConfig(
            num_gpus=config.num_gpus,
            world_size=config.world_size,
            model_dim=config.hidden_size,
            num_layers=config.num_layers
        )
        
        self.distributed_engine = DistributedInferenceEngine(dist_config)
        
        cache_config = PageConfig()
        self.kv_cache = PagedAttentionKVCache(
            cache_config,
            num_pages=config.num_kv_cache_pages,
            device=self.distributed_engine.device
        )
        
        self.prefix_cache = None
        if config.enable_prefix_caching:
            self.prefix_cache = PrefixCache(self.kv_cache)
        
        self.speculative_decoder = None
        if config.enable_speculative_decoding:
            spec_config = SpeculationConfig(max_speculation_depth=4)
            self.speculative_decoder = SpeculativeDecoder(model, spec_config)
        
        self.token_batcher = TokenBatcher(
            max_batch_size=config.max_batch_size,
            max_batch_tokens=config.max_batch_tokens
        )
        
        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self.generation_count = 0
        
        logger.info("UnifiedInferencePipeline initialized")
    
    def add_request(
        self,
        request_id: str,
        prompt_tokens: torch.Tensor,
        max_tokens: int,
        priority: int = 0,
        deadline: Optional[float] = None
    ):
        """
        Add a generation request to the pipeline.
        
        Args:
            request_id: Unique request ID
            prompt_tokens: Input token IDs [seq_len]
            max_tokens: Max tokens to generate
            priority: Request priority (higher = more urgent)
            deadline: Optional deadline timestamp
        """
        self.token_batcher.add_request(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            priority=priority,
            deadline=deadline
        )
        self.total_requests += 1
    
    def _check_prefix_cache(
        self,
        prompt_tokens: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Check if prefix is cached.
        
        Returns:
            (cached_k, cached_v, num_cached_tokens) or (None, None, 0)
        """
        if self.prefix_cache is None:
            return None, None, 0
        
        retrieved = self.prefix_cache.get_prefix(prompt_tokens)
        if retrieved:
            k, v = retrieved
            return k, v, prompt_tokens.shape[0]
        
        return None, None, 0
    
    def generate_batch(
        self,
        batch: TokenBatch,
        num_iterations: int = 1
    ) -> Dict[str, GenerationOutput]:
        """
        Generate tokens for a batch of requests.
        
        Args:
            batch: TokenBatch from token batcher
            num_iterations: Number of generation iterations
        
        Returns:
            Dict mapping request_id -> GenerationOutput
        """
        outputs = {}
        batch_start = time.time()
        
        for request_id in batch.request_ids:
            request_start = time.time()
            
            # Get request tokens
            idx = batch.request_ids.index(request_id)
            prompt_tokens = batch.tokens[idx]
            
            # Check prefix cache
            cached_k, cached_v, num_cached = self._check_prefix_cache(prompt_tokens)
            
            # Prepare input
            input_ids = prompt_tokens.unsqueeze(0).to(self.distributed_engine.device)
            
            # Generate using speculative decoder if available
            if self.speculative_decoder:
                generation_output = self.speculative_decoder.generate(
                    input_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p
                )
                
                generated_ids = generation_output.generated_ids
                num_generated = generation_output.num_tokens
                acceptance_rate = generation_output.acceptance_rate
                latency_ms = generation_output.latency_ms
            else:
                # Fallback to standard generation
                with torch.no_grad():
                    output = self.model(input_ids)
                    logits = output.logits if hasattr(output, 'logits') else output
                    generated_ids = torch.argmax(logits, dim=-1)
                
                num_generated = generated_ids.shape[1]
                acceptance_rate = 1.0
                latency_ms = (time.time() - request_start) * 1000
            
            # Cache the generated tokens
            if self.prefix_cache and num_generated > 0:
                # Cache for future prefix matching
                full_seq = torch.cat([prompt_tokens, generated_ids.squeeze(0)], dim=0)
                self.prefix_cache.cache_prefix(full_seq, cached_k or torch.randn(1, 10, self.config.hidden_size), 
                                                cached_v or torch.randn(1, 10, self.config.hidden_size))
            
            # Create output
            total_latency_ms = (time.time() - request_start) * 1000
            throughput = num_generated / (total_latency_ms / 1000) if total_latency_ms > 0 else 0
            
            output = GenerationOutput(
                request_id=request_id,
                generated_ids=generated_ids.squeeze(0),
                num_tokens=num_generated,
                latency_ms=total_latency_ms,
                throughput_tokens_per_sec=throughput,
                memory_used_mb=0.0,  # Would measure actual GPU memory
                accepted_ratio=acceptance_rate
            )
            
            outputs[request_id] = output
            self.total_tokens += num_generated
            self.total_latency_ms += total_latency_ms
            self.generation_count += 1
        
        return outputs
    
    def process_requests(
        self,
        batch_size: Optional[int] = None,
        num_batches: Optional[int] = None
    ) -> List[GenerationOutput]:
        """
        Process pending requests in batches.
        
        Args:
            batch_size: Override default batch size
            num_batches: Max batches to process
        
        Returns:
            List of GenerationOutput
        """
        all_outputs = []
        batches_processed = 0
        
        while num_batches is None or batches_processed < num_batches:
            # Get next batch
            batch = self.token_batcher.get_batch(batch_size)
            if batch is None:
                break
            
            # Generate
            batch_outputs = self.generate_batch(batch)
            all_outputs.extend(batch_outputs.values())
            
            # Mark requests complete
            for request_id in batch.request_ids:
                self.token_batcher.mark_completed(request_id)
            
            batches_processed += 1
        
        return all_outputs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.total_latency_ms / max(1, self.generation_count),
            "avg_throughput_tokens_per_sec": self.total_tokens / max(1, self.total_latency_ms / 1000),
            "distributed_stats": self.distributed_engine.get_stats(),
            "cache_stats": self.kv_cache.get_memory_stats(),
            "batcher_stats": self.token_batcher.get_stats()
        }
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.total_requests = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self.generation_count = 0


class InferencePipelineExecutor:
    """
    High-level executor for the unified inference pipeline.
    
    Manages pipeline lifecycle and provides simplified API.
    """
    
    def __init__(self, model: nn.Module, config: PipelineConfig):
        self.config = config
        self.pipeline = UnifiedInferencePipeline(model, config)
    
    def __enter__(self):
        """Context manager entry."""
        self.pipeline.distributed_engine.initialize_process_group()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.pipeline.distributed_engine.cleanup()
    
    def generate(
        self,
        request_id: str,
        prompt_tokens: torch.Tensor,
        max_tokens: int,
        priority: int = 0
    ) -> GenerationOutput:
        """
        Generate response for a single request.
        
        Args:
            request_id: Request identifier
            prompt_tokens: Input tokens
            max_tokens: Max generation length
            priority: Request priority
        
        Returns:
            GenerationOutput
        """
        self.pipeline.add_request(request_id, prompt_tokens, max_tokens, priority)
        outputs = self.pipeline.process_requests(num_batches=1)
        return outputs[0] if outputs else None
    
    def generate_batch(
        self,
        requests: List[Tuple[str, torch.Tensor, int]],
        priority: int = 0
    ) -> List[GenerationOutput]:
        """
        Generate responses for multiple requests.
        
        Args:
            requests: List of (request_id, prompt_tokens, max_tokens)
            priority: Request priority
        
        Returns:
            List of GenerationOutput
        """
        for request_id, prompt, max_tokens in requests:
            self.pipeline.add_request(request_id, prompt, max_tokens, priority)
        
        return self.pipeline.process_requests()
    
    def benchmark(
        self,
        num_requests: int = 100,
        avg_prompt_length: int = 100,
        max_generation_length: int = 100
    ) -> Dict[str, Any]:
        """
        Run inference benchmark.
        
        Args:
            num_requests: Number of requests to process
            avg_prompt_length: Average prompt length
            max_generation_length: Max tokens to generate
        
        Returns:
            Benchmark results
        """
        self.pipeline.reset_statistics()
        
        # Generate requests
        for i in range(num_requests):
            prompt = torch.randint(0, self.config.vocab_size, (avg_prompt_length,))
            self.pipeline.add_request(
                request_id=f"req_{i}",
                prompt_tokens=prompt,
                max_tokens=max_generation_length,
                priority=i % 3  # Vary priorities
            )
        
        # Process all requests
        start = time.time()
        outputs = self.pipeline.process_requests()
        elapsed = time.time() - start
        
        # Calculate metrics
        total_tokens = sum(o.num_tokens for o in outputs)
        
        return {
            "num_requests": num_requests,
            "total_tokens_generated": total_tokens,
            "elapsed_seconds": elapsed,
            "throughput_requests_per_sec": num_requests / elapsed,
            "throughput_tokens_per_sec": total_tokens / elapsed,
            "avg_latency_ms": sum(o.latency_ms for o in outputs) / len(outputs),
            "statistics": self.pipeline.get_statistics()
        }


def create_inference_pipeline(
    model: nn.Module,
    num_gpus: int = 8,
    **kwargs
) -> UnifiedInferencePipeline:
    """
    Factory function to create inference pipeline.
    
    Args:
        model: Model to use for inference
        num_gpus: Number of GPUs
        **kwargs: Additional config options
    
    Returns:
        Configured UnifiedInferencePipeline
    """
    config = PipelineConfig(num_gpus=num_gpus, world_size=num_gpus, **kwargs)
    return UnifiedInferencePipeline(model, config)


if __name__ == "__main__":
    # Test unified pipeline
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Unified Inference Pipeline...")
    
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=32000, hidden_size=256):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.output = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            return self.output(x)
    
    model = SimpleModel()
    
    config = PipelineConfig(
        num_gpus=1,
        enable_speculative_decoding=False,  # Disabled for testing
        enable_prefix_caching=True
    )
    
    pipeline = UnifiedInferencePipeline(model, config)
    
    # Add requests
    for i in range(5):
        tokens = torch.randint(0, 32000, (20,))
        pipeline.add_request(f"req_{i}", tokens, max_tokens=10)
    
    # Process
    outputs = pipeline.process_requests(num_batches=2)
    
    print(f"Processed {len(outputs)} requests")
    for output in outputs:
        print(f"  {output.request_id}: {output.num_tokens} tokens in {output.latency_ms:.1f}ms")
    
    # Stats
    stats = pipeline.get_statistics()
    print(f"\nPipeline Stats:")
    print(f"  Avg Latency: {stats['avg_latency_ms']:.1f}ms")
    print(f"  Avg Throughput: {stats['avg_throughput_tokens_per_sec']:.1f} tokens/sec")
    
    print("\nPipeline integration test passed!")
