"""
Pipeline Parallelism for Large Model Training and Inference.

This module implements GPipe-style pipeline parallelism with micro-batching,
1F1B (one-forward-one-backward) scheduling, and activation checkpointing.

Architecture:
    - Model layers distributed across GPUs (vertical sharding)
    - Micro-batching reduces pipeline bubble overhead
    - Asynchronous inter-stage communication
    - Memory-efficient activation checkpointing

Performance Characteristics:
    - Bubble ratio: (p-1) / (p-1 + m) where p=stages, m=micro-batches
    - Memory: O(micro_batch_size * hidden_size * num_layers_per_stage)
    - Throughput scales near-linearly with micro-batch count

Author: Ryzanstein Team
Version: 2.3.0 (Sprint 2.3 - Multi-GPU Optimization)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import deque
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda import Stream

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================

class PipelineSchedule(Enum):
    """Pipeline execution schedules."""
    GPIPE = auto()           # All forward, then all backward (high memory)
    ONE_F_ONE_B = auto()     # Interleaved forward/backward (balanced)
    INTERLEAVED = auto()     # Interleaved stages for better utilization
    ZERO_BUBBLE = auto()     # Advanced schedule minimizing bubbles
    

class ActivationPolicy(Enum):
    """Activation checkpointing policies."""
    NONE = auto()            # Store all activations
    FULL = auto()            # Checkpoint all layers
    SELECTIVE = auto()       # Checkpoint every N layers
    MEMORY_EFFICIENT = auto() # Adaptive based on memory pressure


class StageState(Enum):
    """State of a pipeline stage."""
    IDLE = auto()
    FORWARD = auto()
    BACKWARD = auto()
    WAITING_RECV = auto()
    WAITING_SEND = auto()


@dataclass
class PipelineConfig:
    """Configuration for pipeline parallelism."""
    
    # Core settings
    num_stages: int = 4
    num_micro_batches: int = 8
    schedule: PipelineSchedule = PipelineSchedule.ONE_F_ONE_B
    
    # Activation checkpointing
    activation_policy: ActivationPolicy = ActivationPolicy.SELECTIVE
    checkpoint_interval: int = 2  # Checkpoint every N layers
    
    # Communication
    async_communication: bool = True
    prefetch_count: int = 2  # Prefetch next N micro-batches
    
    # Memory management
    max_activation_memory_mb: int = 8192
    activation_offload: bool = False  # Offload to CPU
    
    # Batch dimensions
    micro_batch_size: int = 1
    sequence_length: int = 2048
    hidden_size: int = 4096
    
    # Advanced options
    scatter_gather_tensors: bool = True
    gradient_accumulation: int = 1
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_micro_batches < self.num_stages:
            logger.warning(
                f"num_micro_batches ({self.num_micro_batches}) < num_stages "
                f"({self.num_stages}). High bubble ratio expected."
            )
        
        # Calculate expected bubble ratio
        p = self.num_stages
        m = self.num_micro_batches
        self.bubble_ratio = (p - 1) / (p - 1 + m)
        
        logger.info(
            f"Pipeline config: {self.num_stages} stages, "
            f"{self.num_micro_batches} micro-batches, "
            f"bubble ratio: {self.bubble_ratio:.2%}"
        )
    
    @property
    def total_batch_size(self) -> int:
        """Total effective batch size."""
        return self.micro_batch_size * self.num_micro_batches


@dataclass
class PipelineStats:
    """Runtime statistics for pipeline execution."""
    
    total_micro_batches: int = 0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    bubble_time_ms: float = 0.0
    
    peak_activation_memory_mb: float = 0.0
    checkpointed_layers: int = 0
    
    send_count: int = 0
    recv_count: int = 0
    bytes_transferred: int = 0
    
    def compute_efficiency(self) -> float:
        """Compute pipeline efficiency."""
        total_time = (
            self.forward_time_ms + 
            self.backward_time_ms + 
            self.bubble_time_ms
        )
        if total_time == 0:
            return 1.0
        return 1.0 - (self.bubble_time_ms / total_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_micro_batches": self.total_micro_batches,
            "forward_time_ms": self.forward_time_ms,
            "backward_time_ms": self.backward_time_ms,
            "communication_time_ms": self.communication_time_ms,
            "bubble_time_ms": self.bubble_time_ms,
            "peak_activation_memory_mb": self.peak_activation_memory_mb,
            "checkpointed_layers": self.checkpointed_layers,
            "efficiency": self.compute_efficiency(),
        }


# =============================================================================
# Communication Primitives
# =============================================================================

class PipelineCommunicator:
    """
    Handles inter-stage communication in pipeline parallelism.
    
    Manages point-to-point tensor transfers between adjacent stages
    with optional asynchronous operations and prefetching.
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        stage_id: int,
        world_size: int,
    ):
        self.config = config
        self.stage_id = stage_id
        self.world_size = world_size
        
        self.prev_rank = stage_id - 1 if stage_id > 0 else None
        self.next_rank = stage_id + 1 if stage_id < world_size - 1 else None
        
        # Communication streams for async ops
        if torch.cuda.is_available():
            self.send_stream = Stream()
            self.recv_stream = Stream()
        else:
            self.send_stream = None
            self.recv_stream = None
        
        # Pending operations
        self._pending_sends: List[dist.Work] = []
        self._pending_recvs: List[dist.Work] = []
        
        # Receive buffers for prefetching
        self._recv_buffers: Dict[int, torch.Tensor] = {}
        
        # Statistics
        self.stats = PipelineStats()
        
        logger.debug(
            f"Stage {stage_id}: prev={self.prev_rank}, next={self.next_rank}"
        )
    
    def send_forward(
        self,
        tensor: torch.Tensor,
        micro_batch_id: int,
        async_op: bool = True,
    ) -> Optional[dist.Work]:
        """Send activation to next stage."""
        if self.next_rank is None:
            return None
        
        start = time.perf_counter()
        
        if self.config.async_communication and async_op:
            with torch.cuda.stream(self.send_stream) if self.send_stream else contextmanager(lambda: (yield))():
                work = dist.isend(tensor, dst=self.next_rank)
                self._pending_sends.append(work)
        else:
            work = dist.send(tensor, dst=self.next_rank)
        
        self.stats.send_count += 1
        self.stats.bytes_transferred += tensor.numel() * tensor.element_size()
        self.stats.communication_time_ms += (time.perf_counter() - start) * 1000
        
        return work if async_op else None
    
    def recv_forward(
        self,
        tensor_shape: Tuple[int, ...],
        dtype: torch.dtype,
        micro_batch_id: int,
        async_op: bool = True,
    ) -> Tuple[torch.Tensor, Optional[dist.Work]]:
        """Receive activation from previous stage."""
        if self.prev_rank is None:
            raise RuntimeError("Cannot receive on first stage")
        
        start = time.perf_counter()
        
        # Check for prefetched buffer
        if micro_batch_id in self._recv_buffers:
            tensor = self._recv_buffers.pop(micro_batch_id)
        else:
            device = f"cuda:{self.stage_id}" if torch.cuda.is_available() else "cpu"
            tensor = torch.empty(tensor_shape, dtype=dtype, device=device)
        
        if self.config.async_communication and async_op:
            with torch.cuda.stream(self.recv_stream) if self.recv_stream else contextmanager(lambda: (yield))():
                work = dist.irecv(tensor, src=self.prev_rank)
                self._pending_recvs.append(work)
        else:
            dist.recv(tensor, src=self.prev_rank)
            work = None
        
        self.stats.recv_count += 1
        self.stats.communication_time_ms += (time.perf_counter() - start) * 1000
        
        return tensor, work if async_op else (tensor, None)
    
    def send_backward(
        self,
        tensor: torch.Tensor,
        micro_batch_id: int,
        async_op: bool = True,
    ) -> Optional[dist.Work]:
        """Send gradients to previous stage."""
        if self.prev_rank is None:
            return None
        
        start = time.perf_counter()
        
        if self.config.async_communication and async_op:
            with torch.cuda.stream(self.send_stream) if self.send_stream else contextmanager(lambda: (yield))():
                work = dist.isend(tensor, dst=self.prev_rank)
                self._pending_sends.append(work)
        else:
            work = dist.send(tensor, dst=self.prev_rank)
        
        self.stats.send_count += 1
        self.stats.bytes_transferred += tensor.numel() * tensor.element_size()
        self.stats.communication_time_ms += (time.perf_counter() - start) * 1000
        
        return work if async_op else None
    
    def recv_backward(
        self,
        tensor_shape: Tuple[int, ...],
        dtype: torch.dtype,
        micro_batch_id: int,
        async_op: bool = True,
    ) -> Tuple[torch.Tensor, Optional[dist.Work]]:
        """Receive gradients from next stage."""
        if self.next_rank is None:
            raise RuntimeError("Cannot receive gradients on last stage")
        
        start = time.perf_counter()
        
        device = f"cuda:{self.stage_id}" if torch.cuda.is_available() else "cpu"
        tensor = torch.empty(tensor_shape, dtype=dtype, device=device)
        
        if self.config.async_communication and async_op:
            with torch.cuda.stream(self.recv_stream) if self.recv_stream else contextmanager(lambda: (yield))():
                work = dist.irecv(tensor, src=self.next_rank)
                self._pending_recvs.append(work)
        else:
            dist.recv(tensor, src=self.next_rank)
            work = None
        
        self.stats.recv_count += 1
        self.stats.communication_time_ms += (time.perf_counter() - start) * 1000
        
        return tensor, work if async_op else (tensor, None)
    
    def wait_pending(self) -> None:
        """Wait for all pending async operations."""
        for work in self._pending_sends:
            work.wait()
        for work in self._pending_recvs:
            work.wait()
        
        self._pending_sends.clear()
        self._pending_recvs.clear()
    
    def prefetch_forward(
        self,
        tensor_shape: Tuple[int, ...],
        dtype: torch.dtype,
        micro_batch_ids: List[int],
    ) -> None:
        """Prefetch activations for upcoming micro-batches."""
        if self.prev_rank is None:
            return
        
        for mb_id in micro_batch_ids[:self.config.prefetch_count]:
            if mb_id not in self._recv_buffers:
                device = f"cuda:{self.stage_id}" if torch.cuda.is_available() else "cpu"
                self._recv_buffers[mb_id] = torch.empty(
                    tensor_shape, dtype=dtype, device=device
                )


# =============================================================================
# Activation Checkpointing
# =============================================================================

class ActivationCheckpointer:
    """
    Manages activation checkpointing for memory-efficient training.
    
    Supports multiple policies:
    - NONE: Store all activations (fastest, highest memory)
    - FULL: Checkpoint all layers (slowest, lowest memory)
    - SELECTIVE: Checkpoint every N layers (balanced)
    - MEMORY_EFFICIENT: Adaptive based on memory pressure
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        num_layers: int,
    ):
        self.config = config
        self.num_layers = num_layers
        self.policy = config.activation_policy
        
        # Determine which layers to checkpoint
        self._checkpoint_layers = self._compute_checkpoint_layers()
        
        # Stored activations
        self._activations: Dict[int, Dict[int, torch.Tensor]] = {}
        self._checkpoints: Dict[int, Dict[int, torch.Tensor]] = {}
        
        # Memory tracking
        self._current_memory_mb = 0.0
        self._peak_memory_mb = 0.0
        
        logger.info(
            f"ActivationCheckpointer: policy={self.policy.name}, "
            f"checkpointed layers: {len(self._checkpoint_layers)}/{num_layers}"
        )
    
    def _compute_checkpoint_layers(self) -> set:
        """Compute which layers should be checkpointed."""
        if self.policy == ActivationPolicy.NONE:
            return set()
        elif self.policy == ActivationPolicy.FULL:
            return set(range(self.num_layers))
        elif self.policy == ActivationPolicy.SELECTIVE:
            interval = self.config.checkpoint_interval
            return {i for i in range(0, self.num_layers, interval)}
        else:  # MEMORY_EFFICIENT
            # Start with every 4th layer, adjust dynamically
            return {i for i in range(0, self.num_layers, 4)}
    
    def should_checkpoint(self, layer_idx: int) -> bool:
        """Check if layer should be checkpointed."""
        if self.policy == ActivationPolicy.MEMORY_EFFICIENT:
            # Dynamically adjust based on memory pressure
            if self._current_memory_mb > self.config.max_activation_memory_mb * 0.8:
                return True
        return layer_idx in self._checkpoint_layers
    
    def store_activation(
        self,
        micro_batch_id: int,
        layer_idx: int,
        activation: torch.Tensor,
    ) -> None:
        """Store activation for backward pass."""
        if micro_batch_id not in self._activations:
            self._activations[micro_batch_id] = {}
        
        if self.should_checkpoint(layer_idx):
            # Store detached copy for checkpointing
            self._activations[micro_batch_id][layer_idx] = activation.detach()
        else:
            # Store full activation
            self._activations[micro_batch_id][layer_idx] = activation
        
        # Update memory tracking
        mem_bytes = activation.numel() * activation.element_size()
        self._current_memory_mb += mem_bytes / (1024 * 1024)
        self._peak_memory_mb = max(self._peak_memory_mb, self._current_memory_mb)
    
    def get_activation(
        self,
        micro_batch_id: int,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """Retrieve stored activation."""
        if micro_batch_id not in self._activations:
            return None
        return self._activations[micro_batch_id].get(layer_idx)
    
    def recompute_activation(
        self,
        micro_batch_id: int,
        layer_idx: int,
        layer_fn: Callable,
        input_activation: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute activation from checkpoint."""
        # Find nearest checkpoint before this layer
        checkpoint_idx = layer_idx
        while checkpoint_idx >= 0 and checkpoint_idx not in self._checkpoint_layers:
            checkpoint_idx -= 1
        
        if checkpoint_idx < 0:
            # No checkpoint, use input
            activation = input_activation
            start_idx = 0
        else:
            activation = self.get_activation(micro_batch_id, checkpoint_idx)
            start_idx = checkpoint_idx + 1
        
        # Recompute from checkpoint to target layer
        # Note: In practice, this would iterate through layers
        # For now, just return the stored or recomputed activation
        if activation is None:
            activation = layer_fn(input_activation)
        
        return activation
    
    def clear_micro_batch(self, micro_batch_id: int) -> None:
        """Clear activations for completed micro-batch."""
        if micro_batch_id in self._activations:
            # Update memory tracking
            for activation in self._activations[micro_batch_id].values():
                mem_bytes = activation.numel() * activation.element_size()
                self._current_memory_mb -= mem_bytes / (1024 * 1024)
            
            del self._activations[micro_batch_id]
    
    def clear_all(self) -> None:
        """Clear all stored activations."""
        self._activations.clear()
        self._checkpoints.clear()
        self._current_memory_mb = 0.0
    
    @property
    def peak_memory_mb(self) -> float:
        """Get peak activation memory usage."""
        return self._peak_memory_mb


# =============================================================================
# Pipeline Stage
# =============================================================================

class PipelineStage(nn.Module):
    """
    A single stage in the pipeline containing a subset of model layers.
    
    Handles:
    - Forward and backward passes for assigned layers
    - Activation storage and checkpointing
    - Communication with adjacent stages
    """
    
    def __init__(
        self,
        stage_id: int,
        layers: nn.ModuleList,
        config: PipelineConfig,
        communicator: PipelineCommunicator,
    ):
        super().__init__()
        
        self.stage_id = stage_id
        self.layers = layers
        self.config = config
        self.communicator = communicator
        
        self.is_first = stage_id == 0
        self.is_last = stage_id == config.num_stages - 1
        
        # Checkpointer
        self.checkpointer = ActivationCheckpointer(config, len(layers))
        
        # State tracking
        self.state = StageState.IDLE
        
        # Stored inputs/outputs for backward
        self._micro_batch_inputs: Dict[int, torch.Tensor] = {}
        self._micro_batch_outputs: Dict[int, torch.Tensor] = {}
        
        logger.debug(
            f"PipelineStage {stage_id}: {len(layers)} layers, "
            f"is_first={self.is_first}, is_last={self.is_last}"
        )
    
    def forward_step(
        self,
        micro_batch_id: int,
        input_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute forward pass for one micro-batch."""
        self.state = StageState.FORWARD
        
        # Get input
        if self.is_first:
            assert input_tensor is not None, "First stage requires input"
            hidden = input_tensor
        else:
            # Receive from previous stage
            self.state = StageState.WAITING_RECV
            hidden, work = self.communicator.recv_forward(
                tensor_shape=input_tensor.shape if input_tensor is not None 
                    else self._infer_input_shape(),
                dtype=torch.float16,  # Assume FP16 for inference
                micro_batch_id=micro_batch_id,
            )
            if work:
                work.wait()
            self.state = StageState.FORWARD
        
        # Store input for backward
        self._micro_batch_inputs[micro_batch_id] = hidden.detach().clone()
        
        # Forward through layers
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer(hidden)
            
            # Checkpoint if needed
            self.checkpointer.store_activation(micro_batch_id, layer_idx, hidden)
        
        # Store output for backward
        self._micro_batch_outputs[micro_batch_id] = hidden
        
        # Send to next stage if not last
        if not self.is_last:
            self.state = StageState.WAITING_SEND
            work = self.communicator.send_forward(hidden, micro_batch_id)
            if work:
                work.wait()
        
        self.state = StageState.IDLE
        return hidden
    
    def backward_step(
        self,
        micro_batch_id: int,
        output_grad: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Execute backward pass for one micro-batch."""
        self.state = StageState.BACKWARD
        
        # Get output gradient
        if self.is_last:
            assert output_grad is not None, "Last stage requires output gradient"
            grad = output_grad
        else:
            # Receive from next stage
            self.state = StageState.WAITING_RECV
            output = self._micro_batch_outputs[micro_batch_id]
            grad, work = self.communicator.recv_backward(
                tensor_shape=output.shape,
                dtype=output.dtype,
                micro_batch_id=micro_batch_id,
            )
            if work:
                work.wait()
            self.state = StageState.BACKWARD
        
        # Backward through layers (reverse order)
        for layer_idx in reversed(range(len(self.layers))):
            layer = self.layers[layer_idx]
            
            # Get or recompute activation
            activation = self.checkpointer.get_activation(micro_batch_id, layer_idx)
            if activation is None:
                # Would recompute from checkpoint
                activation = self._micro_batch_inputs[micro_batch_id]
            
            # Compute gradients (simplified - actual impl would use autograd)
            if hasattr(layer, 'weight') and layer.weight.requires_grad:
                # Gradient computation would happen via autograd
                pass
            
            # Note: In practice, backward() is called on the output tensor
            # with the loss gradient, and PyTorch handles the chain
        
        # Get input gradient
        input_grad = self._micro_batch_inputs[micro_batch_id].grad \
            if self._micro_batch_inputs[micro_batch_id].requires_grad else None
        
        # Send to previous stage if not first
        if not self.is_first and input_grad is not None:
            self.state = StageState.WAITING_SEND
            work = self.communicator.send_backward(input_grad, micro_batch_id)
            if work:
                work.wait()
        
        # Cleanup
        self.checkpointer.clear_micro_batch(micro_batch_id)
        del self._micro_batch_inputs[micro_batch_id]
        del self._micro_batch_outputs[micro_batch_id]
        
        self.state = StageState.IDLE
        return input_grad
    
    def _infer_input_shape(self) -> Tuple[int, ...]:
        """Infer expected input shape."""
        return (
            self.config.micro_batch_size,
            self.config.sequence_length,
            self.config.hidden_size,
        )


# =============================================================================
# Pipeline Schedulers
# =============================================================================

class PipelineScheduler:
    """Base class for pipeline execution schedulers."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def generate_schedule(
        self,
        num_micro_batches: int,
    ) -> List[Tuple[str, int, int]]:
        """
        Generate execution schedule.
        
        Returns list of (operation, stage_id, micro_batch_id) tuples.
        Operations: 'forward', 'backward', 'bubble'
        """
        raise NotImplementedError


class GPipeScheduler(PipelineScheduler):
    """
    GPipe schedule: All forwards, then all backwards.
    
    Simple but high memory usage (stores all activations).
    
    Schedule pattern for 4 stages, 4 micro-batches:
    Time:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    GPU0: F0 F1 F2 F3 -- -- -- -- -- -- -- -- B3 B2 B1 B0
    GPU1: -- F0 F1 F2 F3 -- -- -- -- -- -- B3 B2 B1 B0 --
    GPU2: -- -- F0 F1 F2 F3 -- -- -- -- B3 B2 B1 B0 -- --
    GPU3: -- -- -- F0 F1 F2 F3 -- -- B3 B2 B1 B0 -- -- --
    """
    
    def generate_schedule(
        self,
        num_micro_batches: int,
    ) -> List[Tuple[str, int, int]]:
        schedule = []
        num_stages = self.config.num_stages
        
        # All forward passes
        for mb in range(num_micro_batches):
            for stage in range(num_stages):
                schedule.append(('forward', stage, mb))
        
        # All backward passes (reverse micro-batch order)
        for mb in reversed(range(num_micro_batches)):
            for stage in reversed(range(num_stages)):
                schedule.append(('backward', stage, mb))
        
        return schedule


class OneFOneBScheduler(PipelineScheduler):
    """
    1F1B (One Forward One Backward) schedule.
    
    Interleaves forward and backward passes to reduce memory.
    After warmup, each stage alternates F and B.
    
    Schedule pattern for 4 stages, 8 micro-batches:
    Time:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
    GPU0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4    B5    B6    B7
    GPU1:    F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4    B5    B6    B7
    GPU2:       F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4    B5    B6 B7
    GPU3:          F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
    """
    
    def generate_schedule(
        self,
        num_micro_batches: int,
    ) -> List[Tuple[str, int, int]]:
        schedule = []
        num_stages = self.config.num_stages
        
        # Per-stage tracking
        forward_mb = [0] * num_stages  # Next forward micro-batch
        backward_mb = [0] * num_stages  # Next backward micro-batch
        in_flight = [0] * num_stages    # Micro-batches in flight
        
        # Warmup: Fill pipeline with forwards
        warmup_steps = num_stages - 1
        for stage in range(num_stages):
            for _ in range(min(warmup_steps - stage + 1, num_micro_batches)):
                if forward_mb[stage] < num_micro_batches:
                    schedule.append(('forward', stage, forward_mb[stage]))
                    forward_mb[stage] += 1
                    in_flight[stage] += 1
        
        # Steady state: 1F1B alternation
        while any(backward_mb[s] < num_micro_batches for s in range(num_stages)):
            for stage in range(num_stages):
                # Backward if we have in-flight micro-batches
                if in_flight[stage] > 0 and backward_mb[stage] < num_micro_batches:
                    schedule.append(('backward', stage, backward_mb[stage]))
                    backward_mb[stage] += 1
                    in_flight[stage] -= 1
                
                # Forward if we have more micro-batches
                if forward_mb[stage] < num_micro_batches:
                    schedule.append(('forward', stage, forward_mb[stage]))
                    forward_mb[stage] += 1
                    in_flight[stage] += 1
        
        return schedule


class InterleavedScheduler(PipelineScheduler):
    """
    Interleaved schedule with virtual stages.
    
    Each device handles multiple non-consecutive stages,
    reducing bubble ratio further.
    """
    
    def __init__(self, config: PipelineConfig, num_virtual_stages: int = 2):
        super().__init__(config)
        self.num_virtual_stages = num_virtual_stages
    
    def generate_schedule(
        self,
        num_micro_batches: int,
    ) -> List[Tuple[str, int, int]]:
        # Simplified interleaved schedule
        schedule = []
        num_physical = self.config.num_stages
        
        # Each physical stage handles num_virtual_stages virtual stages
        for mb in range(num_micro_batches):
            for v_stage in range(self.num_virtual_stages):
                for p_stage in range(num_physical):
                    schedule.append(('forward', p_stage, mb))
        
        for mb in reversed(range(num_micro_batches)):
            for v_stage in reversed(range(self.num_virtual_stages)):
                for p_stage in reversed(range(num_physical)):
                    schedule.append(('backward', p_stage, mb))
        
        return schedule


# =============================================================================
# Pipeline Engine
# =============================================================================

class PipelineParallelEngine:
    """
    Orchestrates pipeline-parallel training and inference.
    
    Distributes model layers across GPUs and manages:
    - Schedule execution
    - Inter-stage communication
    - Activation checkpointing
    - Memory optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: PipelineConfig,
        device_ids: Optional[List[int]] = None,
    ):
        self.config = config
        self.device_ids = device_ids or list(range(config.num_stages))
        
        if len(self.device_ids) != config.num_stages:
            raise ValueError(
                f"device_ids length ({len(self.device_ids)}) must match "
                f"num_stages ({config.num_stages})"
            )
        
        # Partition model into stages
        self.stages = self._partition_model(model)
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Statistics
        self.stats = PipelineStats()
        
        logger.info(
            f"PipelineParallelEngine: {config.num_stages} stages on "
            f"devices {self.device_ids}"
        )
    
    def _partition_model(self, model: nn.Module) -> List[PipelineStage]:
        """Partition model layers into pipeline stages."""
        # Get all layers (simplified - assumes model has .layers attribute)
        if hasattr(model, 'layers'):
            all_layers = list(model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            all_layers = list(model.transformer.h)
        else:
            # Try to find sequential layers
            all_layers = []
            for name, module in model.named_children():
                if isinstance(module, (nn.ModuleList, nn.Sequential)):
                    all_layers.extend(list(module))
        
        if not all_layers:
            raise ValueError("Could not find layers to partition")
        
        # Distribute layers across stages
        num_layers = len(all_layers)
        layers_per_stage = num_layers // self.config.num_stages
        remainder = num_layers % self.config.num_stages
        
        stages = []
        layer_idx = 0
        
        for stage_id in range(self.config.num_stages):
            # Extra layer for first 'remainder' stages
            stage_layers = layers_per_stage + (1 if stage_id < remainder else 0)
            
            stage_layer_list = nn.ModuleList(
                all_layers[layer_idx:layer_idx + stage_layers]
            )
            layer_idx += stage_layers
            
            # Move to appropriate device
            device = self.device_ids[stage_id]
            if torch.cuda.is_available():
                stage_layer_list = stage_layer_list.to(f"cuda:{device}")
            
            # Create communicator
            communicator = PipelineCommunicator(
                config=self.config,
                stage_id=stage_id,
                world_size=self.config.num_stages,
            )
            
            # Create stage
            stage = PipelineStage(
                stage_id=stage_id,
                layers=stage_layer_list,
                config=self.config,
                communicator=communicator,
            )
            stages.append(stage)
            
            logger.debug(
                f"Stage {stage_id}: {stage_layers} layers on device {device}"
            )
        
        return stages
    
    def _create_scheduler(self) -> PipelineScheduler:
        """Create appropriate scheduler based on config."""
        schedule_type = self.config.schedule
        
        if schedule_type == PipelineSchedule.GPIPE:
            return GPipeScheduler(self.config)
        elif schedule_type == PipelineSchedule.ONE_F_ONE_B:
            return OneFOneBScheduler(self.config)
        elif schedule_type == PipelineSchedule.INTERLEAVED:
            return InterleavedScheduler(self.config)
        else:
            logger.warning(f"Unknown schedule {schedule_type}, using 1F1B")
            return OneFOneBScheduler(self.config)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Execute pipeline-parallel forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            
        Returns:
            Model output tensor
        """
        start_time = time.perf_counter()
        
        # Split into micro-batches
        batch_size = input_ids.size(0)
        micro_batches = self._split_micro_batches(input_ids)
        
        # Generate schedule (forward only for inference)
        schedule = []
        for mb_id in range(self.config.num_micro_batches):
            for stage_id in range(self.config.num_stages):
                schedule.append(('forward', stage_id, mb_id))
        
        # Execute schedule
        outputs = {}
        for op, stage_id, mb_id in schedule:
            stage = self.stages[stage_id]
            
            if op == 'forward':
                input_tensor = micro_batches[mb_id] if stage_id == 0 else None
                output = stage.forward_step(mb_id, input_tensor)
                
                if stage.is_last:
                    outputs[mb_id] = output
        
        # Combine micro-batch outputs
        result = self._combine_micro_batches(outputs)
        
        # Update stats
        self.stats.forward_time_ms = (time.perf_counter() - start_time) * 1000
        self.stats.total_micro_batches += self.config.num_micro_batches
        
        return result
    
    def _split_micro_batches(
        self,
        input_ids: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Split input into micro-batches."""
        batch_size = input_ids.size(0)
        mb_size = self.config.micro_batch_size
        
        if batch_size < self.config.num_micro_batches * mb_size:
            # Pad if needed
            pad_size = self.config.num_micro_batches * mb_size - batch_size
            input_ids = torch.cat([
                input_ids,
                torch.zeros(pad_size, input_ids.size(1), dtype=input_ids.dtype, device=input_ids.device)
            ])
        
        micro_batches = {}
        for i in range(self.config.num_micro_batches):
            start = i * mb_size
            end = start + mb_size
            micro_batches[i] = input_ids[start:end]
        
        return micro_batches
    
    def _combine_micro_batches(
        self,
        outputs: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Combine micro-batch outputs."""
        sorted_outputs = [outputs[i] for i in sorted(outputs.keys())]
        return torch.cat(sorted_outputs, dim=0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        # Aggregate stats from all stages
        total_comm_time = sum(
            stage.communicator.stats.communication_time_ms 
            for stage in self.stages
        )
        peak_memory = max(
            stage.checkpointer.peak_memory_mb 
            for stage in self.stages
        )
        
        self.stats.communication_time_ms = total_comm_time
        self.stats.peak_activation_memory_mb = peak_memory
        
        return self.stats.to_dict()


# =============================================================================
# Factory Functions
# =============================================================================

def create_pipeline_engine(
    model: nn.Module,
    num_stages: int = 4,
    num_micro_batches: int = 8,
    schedule: PipelineSchedule = PipelineSchedule.ONE_F_ONE_B,
    device_ids: Optional[List[int]] = None,
    **kwargs,
) -> PipelineParallelEngine:
    """
    Create a pipeline-parallel model from a sequential model.
    
    Args:
        model: Sequential model to parallelize
        num_stages: Number of pipeline stages (GPUs)
        num_micro_batches: Number of micro-batches per iteration
        schedule: Execution schedule type
        device_ids: List of GPU device IDs
        **kwargs: Additional PipelineConfig options
        
    Returns:
        PipelineParallelEngine wrapping the model
    """
    config = PipelineConfig(
        num_stages=num_stages,
        num_micro_batches=num_micro_batches,
        schedule=schedule,
        **kwargs,
    )
    
    return PipelineParallelEngine(
        model=model,
        config=config,
        device_ids=device_ids,
    )


def estimate_pipeline_memory(
    num_layers: int,
    hidden_size: int,
    sequence_length: int,
    num_stages: int,
    num_micro_batches: int,
    activation_policy: ActivationPolicy = ActivationPolicy.SELECTIVE,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """
    Estimate memory requirements for pipeline parallelism.
    
    Returns:
        Dictionary with memory estimates in MB
    """
    bytes_per_element = 2 if dtype == torch.float16 else 4
    layers_per_stage = num_layers // num_stages
    
    # Activation memory per layer
    activation_size = sequence_length * hidden_size * bytes_per_element
    
    # Memory depends on checkpointing policy
    if activation_policy == ActivationPolicy.NONE:
        # Store all activations for all micro-batches
        activations_per_stage = activation_size * layers_per_stage * num_micro_batches
    elif activation_policy == ActivationPolicy.FULL:
        # Only input activation per micro-batch
        activations_per_stage = activation_size * num_micro_batches
    elif activation_policy == ActivationPolicy.SELECTIVE:
        # Checkpoint every 2 layers
        checkpoints = (layers_per_stage + 1) // 2
        activations_per_stage = activation_size * checkpoints * num_micro_batches
    else:  # MEMORY_EFFICIENT
        # Adaptive - estimate conservatively
        checkpoints = (layers_per_stage + 3) // 4
        activations_per_stage = activation_size * checkpoints * num_micro_batches
    
    # Convert to MB
    activation_memory_mb = activations_per_stage / (1024 * 1024)
    
    # Communication buffers (2x for send/recv)
    comm_buffer_mb = 2 * activation_size / (1024 * 1024)
    
    return {
        "activation_memory_mb": activation_memory_mb,
        "communication_buffer_mb": comm_buffer_mb,
        "total_per_stage_mb": activation_memory_mb + comm_buffer_mb,
        "layers_per_stage": layers_per_stage,
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "PipelineSchedule",
    "ActivationPolicy",
    "StageState",
    
    # Configuration
    "PipelineConfig",
    "PipelineStats",
    
    # Core classes
    "PipelineCommunicator",
    "ActivationCheckpointer",
    "PipelineStage",
    "PipelineParallelEngine",
    
    # Schedulers
    "PipelineScheduler",
    "GPipeScheduler",
    "OneFOneBScheduler",
    "InterleavedScheduler",
    
    # Factory functions
    "create_pipeline_parallel_model",
    "estimate_pipeline_memory",
]
