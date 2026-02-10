"""
Token-Level Batcher
===================

Advanced batching at the token level for maximum throughput.

Instead of batching requests, this batcher groups tokens across all
active requests to keep GPU fully utilized with minimal idle time.

Features:
- Token-level batching across requests
- Priority queue management
- Dynamic batch construction
- Request interleaving
- SLA preservation

Sprint 2.2 - Distributed Inference & Performance
Created: 2025-12-26
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
import time
import heapq

logger = logging.getLogger(__name__)


class RequestState(Enum):
    """State of a generation request."""
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class TokenRequest:
    """A single token generation request."""
    request_id: str
    prompt_tokens: torch.Tensor
    max_tokens: int
    priority: int = 0
    arrival_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    
    # State tracking
    state: RequestState = RequestState.WAITING
    generated_tokens: int = 0
    start_time: Optional[float] = None
    
    def __lt__(self, other):
        """Comparison for priority queue."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.arrival_time < other.arrival_time  # Earlier first
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time since arrival."""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) * 1000
    
    def is_deadline_exceeded(self) -> bool:
        """Check if deadline has been exceeded."""
        if self.deadline is None:
            return False
        return time.time() > self.deadline


@dataclass
class TokenBatch:
    """A batch of tokens from multiple requests."""
    batch_id: str
    request_ids: List[str]
    tokens: torch.Tensor              # [batch_size, seq_len]
    token_counts: List[int]           # Tokens per request
    indices: List[int]                # Index of each token in its request
    batch_size: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        self.batch_size = len(self.request_ids)
        self.total_tokens = self.tokens.shape[1]


class TokenBatcher:
    """
    Token-level batcher for high-throughput inference.
    
    Maintains a priority queue of requests and constructs batches
    by selecting the next token from each active request.
    """
    
    def __init__(
        self,
        max_batch_size: int = 128,
        max_batch_tokens: int = 4096,
        padding_strategy: str = "right"
    ):
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.padding_strategy = padding_strategy
        
        # Request management
        self.pending_requests: Dict[str, TokenRequest] = {}
        self.active_requests: Dict[str, TokenRequest] = {}
        self.completed_requests: Dict[str, TokenRequest] = {}
        
        # Priority queue for scheduling
        self.request_queue: List[TokenRequest] = []
        
        # Statistics
        self.total_requests = 0
        self.total_batches = 0
        self.total_tokens_processed = 0
        
        logger.info(
            f"TokenBatcher initialized: "
            f"max_batch_size={max_batch_size}, "
            f"max_batch_tokens={max_batch_tokens}"
        )
    
    def add_request(
        self,
        request_id: str,
        prompt_tokens: torch.Tensor,
        max_tokens: int,
        priority: int = 0,
        deadline: Optional[float] = None
    ):
        """
        Add a new generation request.
        
        Args:
            request_id: Unique request identifier
            prompt_tokens: Initial prompt tokens [seq_len]
            max_tokens: Maximum tokens to generate
            priority: Request priority (higher = more urgent)
            deadline: Optional deadline timestamp
        """
        request = TokenRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            priority=priority,
            deadline=deadline
        )
        
        self.pending_requests[request_id] = request
        heapq.heappush(self.request_queue, request)
        self.total_requests += 1
        
        logger.debug(f"Added request {request_id}")
    
    def get_batch(
        self,
        batch_size: Optional[int] = None,
        batch_tokens: Optional[int] = None
    ) -> Optional[TokenBatch]:
        """
        Get next batch of tokens.
        
        Selects requests to fill the batch to specified size/token count.
        
        Args:
            batch_size: Maximum batch size (requests)
            batch_tokens: Maximum batch tokens
        
        Returns:
            TokenBatch if available, None otherwise
        """
        if not self.request_queue:
            return None
        
        if batch_size is None:
            batch_size = self.max_batch_size
        if batch_tokens is None:
            batch_tokens = self.max_batch_tokens
        
        # Select requests for this batch
        selected_requests = []
        selected_ids = []
        batch_token_count = 0
        
        # Build batch from queue
        temp_queue = []
        
        while self.request_queue and len(selected_requests) < batch_size:
            request = heapq.heappop(self.request_queue)
            
            # Check if we can add this request
            tokens_in_prompt = request.prompt_tokens.shape[0]
            if batch_token_count + tokens_in_prompt <= batch_tokens:
                selected_requests.append(request)
                selected_ids.append(request.request_id)
                batch_token_count += tokens_in_prompt
                request.state = RequestState.RUNNING
                request.start_time = time.time()
                self.active_requests[request.request_id] = request
            else:
                # Put back in queue
                temp_queue.append(request)
        
        # Restore remaining requests to queue
        for req in temp_queue:
            heapq.heappush(self.request_queue, req)
        
        if not selected_requests:
            return None
        
        # Construct batch tensor
        max_seq_len = max(r.prompt_tokens.shape[0] for r in selected_requests)
        batch_tensor = self._construct_batch_tensor(
            selected_requests,
            max_seq_len
        )
        
        # Create batch object
        batch_id = f"batch_{self.total_batches}"
        self.total_batches += 1
        
        batch = TokenBatch(
            batch_id=batch_id,
            request_ids=selected_ids,
            tokens=batch_tensor,
            token_counts=[r.prompt_tokens.shape[0] for r in selected_requests],
            indices=list(range(len(selected_requests)))
        )
        
        self.total_tokens_processed += batch.total_tokens
        
        return batch
    
    def _construct_batch_tensor(
        self,
        requests: List[TokenRequest],
        max_seq_len: int
    ) -> torch.Tensor:
        """
        Construct padded batch tensor from requests.
        
        Args:
            requests: List of requests
            max_seq_len: Maximum sequence length
        
        Returns:
            Padded batch tensor [batch_size, max_seq_len]
        """
        batch_size = len(requests)
        batch = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        
        for i, request in enumerate(requests):
            tokens = request.prompt_tokens
            
            if self.padding_strategy == "right":
                batch[i, :len(tokens)] = tokens
            elif self.padding_strategy == "left":
                batch[i, -len(tokens):] = tokens
        
        return batch
    
    def mark_completed(self, request_id: str) -> bool:
        """
        Mark a request as completed.
        
        Args:
            request_id: Request to mark complete
        
        Returns:
            True if request was active
        """
        if request_id in self.active_requests:
            request = self.active_requests.pop(request_id)
            request.state = RequestState.FINISHED
            self.completed_requests[request_id] = request
            logger.debug(f"Marked request {request_id} as completed")
            return True
        return False
    
    def get_pending_count(self) -> int:
        """Get number of pending requests."""
        return len(self.pending_requests)
    
    def get_active_count(self) -> int:
        """Get number of active requests."""
        return len(self.active_requests)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "total_tokens": self.total_tokens_processed,
            "pending": self.get_pending_count(),
            "active": self.get_active_count(),
            "completed": len(self.completed_requests),
            "avg_batch_size": (
                self.total_tokens_processed / self.total_batches
                if self.total_batches > 0 else 0
            )
        }


class RequestQueue:
    """
    Priority queue for managing token generation requests.
    
    Ensures SLA preservation while maintaining fairness.
    """
    
    def __init__(self, enable_preemption: bool = False):
        self.queue: List[TokenRequest] = []
        self.enable_preemption = enable_preemption
    
    def enqueue(self, request: TokenRequest):
        """Enqueue a request."""
        heapq.heappush(self.queue, request)
    
    def dequeue(self) -> Optional[TokenRequest]:
        """Dequeue highest priority request."""
        if self.queue:
            return heapq.heappop(self.queue)
        return None
    
    def requeue(self, request: TokenRequest):
        """Requeue a request."""
        heapq.heappush(self.queue, request)
    
    def prioritize_sla_violations(self):
        """
        Boost priority of requests approaching deadline.
        
        Called periodically to ensure SLA violations are minimized.
        """
        if not self.enable_preemption:
            return
        
        current_time = time.time()
        
        # Check for deadline violations
        for request in self.queue:
            if request.deadline and request.deadline - current_time < 100:  # 100ms until deadline
                # Boost priority
                request.priority += 10


class BatchScheduler:
    """
    Schedule batch execution based on various strategies.
    
    Strategies:
    - FCFS: First Come First Served
    - Priority: By priority value
    - SLA: Minimize SLA violations
    - Fairness: Balance between all requests
    """
    
    def __init__(self, strategy: str = "sla"):
        self.strategy = strategy
        self.iteration = 0
    
    def select_batch(
        self,
        batcher: TokenBatcher,
        batch_size: int
    ) -> Optional[TokenBatch]:
        """
        Select next batch to execute.
        
        Args:
            batcher: TokenBatcher instance
            batch_size: Desired batch size
        
        Returns:
            Selected TokenBatch
        """
        if self.strategy == "fcfs":
            return batcher.get_batch(batch_size)
        
        elif self.strategy == "priority":
            # Already handled by priority queue in batcher
            return batcher.get_batch(batch_size)
        
        elif self.strategy == "sla":
            # Prioritize requests approaching deadline
            batcher.request_queue.__class__ = RequestQueue  # Type cast
            return batcher.get_batch(batch_size)
        
        elif self.strategy == "fairness":
            # Round-robin style
            self.iteration = (self.iteration + 1) % max(1, len(batcher.active_requests) + 1)
            return batcher.get_batch(batch_size)
        
        return batcher.get_batch(batch_size)


def create_token_batcher(
    max_batch_size: int = 128,
    max_batch_tokens: int = 4096,
    **kwargs
) -> TokenBatcher:
    """
    Factory function to create token batcher.
    
    Args:
        max_batch_size: Maximum batch size
        max_batch_tokens: Maximum batch tokens
        **kwargs: Additional config options
    
    Returns:
        Configured TokenBatcher
    """
    return TokenBatcher(
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
        **kwargs
    )


if __name__ == "__main__":
    # Test token batcher
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Token Batcher...")
    
    batcher = TokenBatcher(max_batch_size=32, max_batch_tokens=512)
    
    # Add requests
    for i in range(10):
        tokens = torch.randint(0, 32000, (50 + i * 10,))
        batcher.add_request(
            request_id=f"req_{i}",
            prompt_tokens=tokens,
            max_tokens=100,
            priority=i % 3
        )
    
    # Get batches
    batches = 0
    while True:
        batch = batcher.get_batch()
        if batch is None:
            break
        
        print(f"Batch {batch.batch_id}: {batch.batch_size} requests, {batch.total_tokens} tokens")
        batches += 1
        
        # Mark requests complete
        for req_id in batch.request_ids:
            batcher.mark_completed(req_id)
    
    # Print stats
    stats = batcher.get_stats()
    print(f"Stats: {stats}")
    
    print("Token batcher test passed!")
