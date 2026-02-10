"""
Tensor Parallelism for Multi-GPU Model Sharding
=================================================

Implements Megatron-style tensor parallelism for distributing large models
across multiple GPUs. Splits attention heads and FFN layers horizontally
for parallel computation with all-reduce synchronization.

Key Features:
- Column/Row parallel linear layers (Megatron-LM style)
- Attention head distribution across GPUs
- FFN expert parallelism (for MoE models)
- Efficient all-reduce collective operations
- Gradient checkpointing integration
- Memory-efficient activation recomputation
- Async communication overlap with computation

Performance Targets:
- Near-linear scaling up to 8 GPUs (>90% efficiency)
- <5% communication overhead with overlap
- <100ms latency for 7B model on 2 GPUs

Sprint 2.3: Multi-GPU Optimization
Created: 2025-01-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from abc import ABC, abstractmethod
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


class TensorParallelMode(Enum):
    """Tensor parallel splitting strategies."""
    COLUMN = "column"           # Split weights along columns (output dim)
    ROW = "row"                 # Split weights along rows (input dim)
    ATTENTION_HEAD = "head"     # Split by attention heads
    EXPERT = "expert"           # Split by MoE experts
    EMBEDDING = "embedding"     # Split embedding table


@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallelism."""
    # Parallelism settings
    world_size: int = 1                     # Number of GPUs for tensor parallelism
    rank: int = 0                           # Current GPU rank in the tensor parallel group
    sequence_parallel: bool = False         # Enable sequence parallelism
    async_communication: bool = True        # Async all-reduce with compute overlap
    gradient_accumulation_steps: int = 1    # Number of gradient accumulation steps
    
    # Model dimensions
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8            # For GQA models
    intermediate_size: int = 11008
    num_layers: int = 32
    vocab_size: int = 32000
    
    # Communication
    process_group: Optional[Any] = None
    backend: str = "nccl"
    
    # Optimization
    gradient_checkpointing: bool = True
    use_fused_kernels: bool = True
    reduce_scatter_coalesce: bool = True    # Coalesce small tensors
    coalesce_threshold_bytes: int = 2 ** 20 # 1 MB threshold


@dataclass  
class TensorParallelStats:
    """Statistics for tensor parallel execution."""
    total_compute_time_ms: float = 0.0
    total_comm_time_ms: float = 0.0
    all_reduce_count: int = 0
    all_gather_count: int = 0
    reduce_scatter_count: int = 0
    bytes_communicated: int = 0
    overlap_efficiency: float = 0.0


class ProcessGroupManager:
    """
    Manage process groups for tensor parallelism.
    
    Handles creation and lifecycle of NCCL process groups
    for different parallelism dimensions.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.world_size: int = 1
        self.rank: int = 0
        self.tensor_parallel_size: int = 1
        self.pipeline_parallel_size: int = 1
        self.data_parallel_size: int = 1
        
        # Process groups
        self.tensor_parallel_group: Optional[Any] = None
        self.pipeline_parallel_group: Optional[Any] = None
        self.data_parallel_group: Optional[Any] = None
        self.model_parallel_group: Optional[Any] = None
        
        self._initialized = True
        
    def initialize(
        self,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        backend: str = "nccl"
    ) -> None:
        """Initialize process groups for parallelism."""
        if not dist.is_initialized():
            logger.warning("Distributed not initialized, using single GPU")
            return
            
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = self.world_size // (
            tensor_parallel_size * pipeline_parallel_size
        )
        
        # Create tensor parallel groups (GPUs in same TP group)
        num_tp_groups = self.world_size // tensor_parallel_size
        for i in range(num_tp_groups):
            ranks = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.tensor_parallel_group = group
                
        # Create pipeline parallel groups (GPUs in same PP group)
        for i in range(self.world_size // pipeline_parallel_size):
            ranks = [i + j * (self.world_size // pipeline_parallel_size) 
                    for j in range(pipeline_parallel_size)]
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.pipeline_parallel_group = group
                
        logger.info(
            f"Initialized process groups: TP={tensor_parallel_size}, "
            f"PP={pipeline_parallel_size}, DP={self.data_parallel_size}"
        )
        
    def get_tensor_parallel_rank(self) -> int:
        """Get rank within tensor parallel group."""
        return self.rank % self.tensor_parallel_size
    
    def get_pipeline_parallel_rank(self) -> int:
        """Get rank within pipeline parallel group."""
        return (self.rank // self.tensor_parallel_size) % self.pipeline_parallel_size
    
    def get_data_parallel_rank(self) -> int:
        """Get rank within data parallel group."""
        return self.rank // (self.tensor_parallel_size * self.pipeline_parallel_size)


class AllReduceHandle:
    """
    Handle for async all-reduce operations.
    
    Enables overlap of communication with computation.
    """
    
    def __init__(self, tensor: torch.Tensor, group: Any, async_op: bool = True):
        self.tensor = tensor
        self.group = group
        self.async_op = async_op
        self.handle: Optional[Any] = None
        self._completed = not async_op
        
        # Only perform distributed operations if initialized
        if dist.is_initialized():
            if async_op:
                self.handle = dist.all_reduce(tensor, group=group, async_op=True)
            else:
                dist.all_reduce(tensor, group=group)
        else:
            # In non-distributed mode, mark as completed immediately
            self._completed = True
            
    def wait(self) -> torch.Tensor:
        """Wait for async operation to complete."""
        if not self._completed and self.handle is not None:
            self.handle.wait()
            self._completed = True
        return self.tensor
    
    def is_completed(self) -> bool:
        """Check if operation is completed."""
        if self._completed:
            return True
        if self.handle is not None:
            return self.handle.is_completed()
        return True


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column-parallel weight distribution.
    
    Splits weight matrix along output dimension (columns).
    Each GPU computes a portion of the output, then all-gather
    to reconstruct the full output.
    
    Y = X @ A = X @ [A_1, A_2, ..., A_n]
    
    Where A_i is a column partition on GPU i.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: TensorParallelConfig,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Optional[Callable] = None,
        async_comm: bool = True,
        rank: Optional[int] = None
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.gather_output = gather_output
        self.async_comm = async_comm
        self.rank = rank if rank is not None else config.rank
        
        # Partition output features across GPUs
        self.tp_size = config.world_size
        assert out_features % self.tp_size == 0, \
            f"Output features {out_features} not divisible by TP size {self.tp_size}"
        
        self.out_features_per_partition = out_features // self.tp_size
        
        # Create partitioned weight
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_partition)
            )
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        self._init_weights(init_method)
        
        # Stats tracking
        self.num_forward = 0
        self.comm_time_total = 0.0
        
    def _init_weights(self, init_method: Optional[Callable]) -> None:
        """Initialize weights with scaling for tensor parallelism."""
        if init_method is not None:
            init_method(self.weight)
        else:
            # Xavier initialization scaled for TP
            std = math.sqrt(2.0 / (self.in_features + self.out_features))
            nn.init.normal_(self.weight, mean=0.0, std=std)
            
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with column-parallel computation.
        
        Args:
            input: Input tensor [batch, seq, in_features]
            
        Returns:
            Output tensor [batch, seq, out_features] if gather_output
            else [batch, seq, out_features_per_partition]
        """
        # Linear on partition: [B, S, in] @ [in, out/tp] -> [B, S, out/tp]
        output_parallel = F.linear(input, self.weight, self.bias)
        
        if self.gather_output:
            # All-gather to get full output
            output = self._all_gather(output_parallel)
            return output
        else:
            return output_parallel
            
    def _all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather tensor partitions across tensor parallel group."""
        if self.tp_size == 1:
            return tensor
            
        if not dist.is_initialized():
            return tensor
            
        world_size = self.tp_size
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        
        group = self.config.process_group
        dist.all_gather(tensor_list, tensor, group=group)
        
        # Concatenate along last dimension
        output = torch.cat(tensor_list, dim=-1)
        return output


class RowParallelLinear(nn.Module):
    """
    Linear layer with row-parallel weight distribution.
    
    Splits weight matrix along input dimension (rows).
    Each GPU receives partitioned input and computes partial output,
    then all-reduce to sum the results.
    
    Y = X @ A = [X_1, X_2, ...] @ [A_1; A_2; ...] = sum(X_i @ A_i)
    
    Where A_i is a row partition and X_i is input partition on GPU i.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: TensorParallelConfig,
        bias: bool = True,
        input_is_parallel: bool = False,
        reduce_output: bool = True,
        init_method: Optional[Callable] = None,
        async_comm: bool = True,
        rank: Optional[int] = None
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.input_is_parallel = input_is_parallel
        self.reduce_output = reduce_output
        self.async_comm = async_comm
        self.rank = rank if rank is not None else config.rank
        
        # Partition input features across GPUs
        self.tp_size = config.world_size
        assert in_features % self.tp_size == 0, \
            f"Input features {in_features} not divisible by TP size {self.tp_size}"
        
        self.in_features_per_partition = in_features // self.tp_size
        
        # Create partitioned weight
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        
        # Bias only on rank 0 (or replicated)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self._init_weights(init_method)
        
        # Async all-reduce handle
        self._all_reduce_handle: Optional[AllReduceHandle] = None
        
    def _init_weights(self, init_method: Optional[Callable]) -> None:
        """Initialize weights with scaling for tensor parallelism."""
        if init_method is not None:
            init_method(self.weight)
        else:
            # Xavier initialization scaled for TP
            std = math.sqrt(2.0 / (self.in_features + self.out_features))
            std = std / math.sqrt(self.tp_size)  # Scale for reduction
            nn.init.normal_(self.weight, mean=0.0, std=std)
            
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, input: torch.Tensor) -> Union[torch.Tensor, AllReduceHandle]:
        """
        Forward pass with row-parallel computation.
        
        Args:
            input: Input tensor [batch, seq, in_features] or 
                   [batch, seq, in_features_per_partition] if input_is_parallel
                   
        Returns:
            Output tensor [batch, seq, out_features]
        """
        if self.input_is_parallel:
            input_parallel = input
        else:
            # Scatter input across GPUs
            input_parallel = self._scatter_input(input)
            
        # Linear on partition: [B, S, in/tp] @ [in/tp, out] -> [B, S, out]
        output_parallel = F.linear(input_parallel, self.weight)
        
        if self.reduce_output:
            # All-reduce to sum partitions
            if self.async_comm:
                self._all_reduce_handle = AllReduceHandle(
                    output_parallel, 
                    self.config.process_group,
                    async_op=True
                )
                output = output_parallel  # Will be updated in-place
            else:
                output = self._all_reduce(output_parallel)
                
            # Add bias after reduction
            if self.bias is not None:
                output = output + self.bias
                
            return output
        else:
            return output_parallel
            
    def _scatter_input(self, input: torch.Tensor) -> torch.Tensor:
        """Scatter input tensor to partitions."""
        if self.tp_size == 1:
            return input
            
        # Get partition for this rank
        pg_manager = ProcessGroupManager()
        tp_rank = pg_manager.get_tensor_parallel_rank()
        
        start_idx = tp_rank * self.in_features_per_partition
        end_idx = start_idx + self.in_features_per_partition
        
        return input[..., start_idx:end_idx]
        
    def _all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Synchronous all-reduce across tensor parallel group."""
        if self.tp_size == 1:
            return tensor
            
        if not dist.is_initialized():
            return tensor
            
        dist.all_reduce(tensor, group=self.config.process_group)
        return tensor
        
    def wait_for_async(self) -> None:
        """Wait for async all-reduce to complete."""
        if self._all_reduce_handle is not None:
            self._all_reduce_handle.wait()
            self._all_reduce_handle = None


class ParallelAttention(nn.Module):
    """
    Tensor-parallel multi-head attention.
    
    Distributes attention heads across GPUs. Each GPU computes
    attention for a subset of heads, then results are gathered.
    
    Supports:
    - Multi-head attention (MHA)
    - Grouped-query attention (GQA)
    - Multi-query attention (MQA)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        config: TensorParallelConfig,
        rank: int = 0,
        num_kv_heads: Optional[int] = None,
        layer_idx: int = 0
    ):
        super().__init__()
        
        self.config = config
        self.rank = rank
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.tp_size = config.world_size
        
        # Validate head distribution
        assert self.num_heads % self.tp_size == 0, \
            f"Num heads {self.num_heads} not divisible by TP size {self.tp_size}"
        
        self.num_heads_per_partition = self.num_heads // self.tp_size
        self.num_kv_heads_per_partition = max(1, self.num_kv_heads // self.tp_size)
        
        # Q, K, V projections (column parallel)
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            config,
            bias=False,
            gather_output=False  # Keep partitioned for attention
        )
        
        # K, V may share heads (GQA)
        kv_hidden = self.num_kv_heads * self.head_dim
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            kv_hidden,
            config,
            bias=False,
            gather_output=False
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            kv_hidden,
            config,
            bias=False,
            gather_output=False
        )
        
        # Output projection (row parallel)
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            config,
            bias=False,
            input_is_parallel=True,
            reduce_output=True
        )
        
        # Rotary embeddings (computed per partition)
        self._init_rotary_embeddings()
        
    def _init_rotary_embeddings(self, max_seq_len: int = 8192) -> None:
        """Initialize rotary position embeddings."""
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        
    def _apply_rotary(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply rotary position embeddings.
        
        Args:
            x: Input tensor of shape [batch, num_heads, seq_len, head_dim]
            position_ids: Optional position indices [batch, seq_len]
            
        Returns:
            Tensor with rotary embeddings applied, same shape as input
        """
        # x is [B, H, S, D] after transpose in forward()
        seq_len = x.shape[2]  # Sequence length is at dimension 2 after transpose
        
        if position_ids is not None:
            cos = self.cos_cached[position_ids]  # [B, S, D]
            sin = self.sin_cached[position_ids]
            # Expand for heads: [B, S, D] -> [B, 1, S, D]
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            cos = self.cos_cached[:seq_len]  # [S, D]
            sin = self.sin_cached[:seq_len]
            # Expand for batch and heads: [S, D] -> [1, 1, S, D]
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            
        # Rotate half
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass with tensor-parallel attention.
        
        Args:
            hidden_states: [batch, seq, hidden_size]
            attention_mask: [batch, 1, seq, seq] or None
            position_ids: [batch, seq] or None
            past_key_value: Cached K, V tensors
            use_cache: Whether to return updated cache
            
        Returns:
            output: [batch, seq, hidden_size]
            past_key_value: Updated cache if use_cache=True
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project Q, K, V (column parallel)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for attention: [B, S, H/tp, D] -> [B, H/tp, S, D]
        query = query.view(
            batch_size, seq_length, self.num_heads_per_partition, self.head_dim
        ).transpose(1, 2)
        
        key = key.view(
            batch_size, seq_length, self.num_kv_heads_per_partition, self.head_dim
        ).transpose(1, 2)
        
        value = value.view(
            batch_size, seq_length, self.num_kv_heads_per_partition, self.head_dim
        ).transpose(1, 2)
        
        # Apply rotary embeddings
        query = self._apply_rotary(query, position_ids)
        key = self._apply_rotary(key, position_ids)
        
        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
            
        if use_cache:
            present_key_value = (key, value)
        else:
            present_key_value = None
            
        # Repeat K, V for GQA
        if self.num_kv_heads_per_partition < self.num_heads_per_partition:
            n_rep = self.num_heads_per_partition // self.num_kv_heads_per_partition
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)
            
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # Attention output
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape: [B, H/tp, S, D] -> [B, S, H*D/tp]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, -1)
        
        # Output projection (row parallel with all-reduce)
        output = self.o_proj(attn_output)
        
        if use_cache:
            return output, present_key_value
        else:
            return output


class ParallelMLP(nn.Module):
    """
    Tensor-parallel MLP (Feed-Forward Network).
    
    Uses column parallel for up-projection and row parallel for
    down-projection to minimize communication.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        config: TensorParallelConfig,
        rank: int = 0,
        layer_idx: int = 0
    ):
        super().__init__()
        
        self.config = config
        self.rank = rank
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Gate and up projection (column parallel)
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            config,
            bias=False,
            gather_output=False
        )
        
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            config,
            bias=False,
            gather_output=False
        )
        
        # Down projection (row parallel)
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            config,
            bias=False,
            input_is_parallel=True,
            reduce_output=True
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with tensor-parallel MLP.
        
        Uses SwiGLU activation: down(silu(gate(x)) * up(x))
        """
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        
        # SwiGLU activation
        intermediate = F.silu(gate) * up
        
        # Down projection with all-reduce
        output = self.down_proj(intermediate)
        
        return output


class ParallelTransformerLayer(nn.Module):
    """
    Full tensor-parallel transformer layer.
    
    Combines parallel attention and MLP with proper
    residual connections and layer normalization.
    """
    
    def __init__(
        self,
        config: TensorParallelConfig,
        layer_idx: int = 0
    ):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        
        # Layer normalization (replicated on all GPUs)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        
        # Tensor-parallel attention
        self.self_attn = ParallelAttention(config, layer_idx)
        
        # Tensor-parallel MLP
        self.mlp = ParallelMLP(config, layer_idx)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for tensor-parallel transformer layer.
        
        Follows pre-norm architecture:
        x = x + attention(norm(x))
        x = x + mlp(norm(x))
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        hidden_states = residual + attn_output
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class TensorParallelModel(nn.Module):
    """
    Tensor-parallel model wrapper.
    
    Can either wrap an existing model or create a full transformer model.
    
    Handles:
    - Model sharding across GPUs
    - Embedding parallelism
    - Layer distribution
    - Output projection
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[TensorParallelConfig] = None,
        rank: int = 0,
        num_layers: Optional[int] = None
    ):
        super().__init__()
        
        # Handle config - either from model or provided
        if config is None:
            config = TensorParallelConfig()
        self.config = config
        self.rank = rank
        
        # If model is provided, wrap it; otherwise create internal layers
        if model is not None:
            # Wrapping mode: store reference to wrapped model
            self.wrapped_model = model
            self.num_layers = 0
            self.embed_tokens = None
            self.layers = nn.ModuleList()
            self.norm = None
            self.lm_head = None
        else:
            # Full transformer mode: create internal components
            self.wrapped_model = None
            self.num_layers = num_layers or config.num_layers
            
            # Parallel embedding (vocabulary split)
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                config
            )
            
            # Transformer layers
            self.layers = nn.ModuleList([
                ParallelTransformerLayer(config, layer_idx=i)
                for i in range(self.num_layers)
            ])
            
            # Final layer norm (replicated)
            self.norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
            
            # LM head (column parallel)
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                config,
                bias=False,
                gather_output=True  # Need full vocab for sampling
            )
        
        # Stats
        self.stats = TensorParallelStats()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through tensor-parallel model.
        
        Args:
            input_ids: [batch, seq]
            attention_mask: [batch, seq] or None
            position_ids: [batch, seq] or None  
            past_key_values: List of (K, V) tuples per layer
            use_cache: Whether to return updated KV cache
            
        Returns:
            logits: [batch, seq, vocab_size]
            present_key_values: Updated cache if use_cache=True
        """
        # If we're wrapping an existing model, delegate to it
        if self.wrapped_model is not None:
            return self.wrapped_model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache
            )
        
        batch_size, seq_length = input_ids.shape
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare position IDs
        if position_ids is None:
            if past_key_values is not None and len(past_key_values) > 0:
                past_length = past_key_values[0][0].shape[2]
            else:
                past_length = 0
            position_ids = torch.arange(
                past_length, past_length + seq_length, 
                device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
            
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(
                attention_mask, seq_length, past_key_values
            )
            
        # Process through layers
        present_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache
            )
            
            if use_cache:
                present_key_values.append(present_kv)
                
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        return logits, present_key_values
        
    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        seq_length: int,
        past_key_values: Optional[List] = None
    ) -> torch.Tensor:
        """Prepare causal attention mask."""
        if past_key_values is not None and len(past_key_values) > 0:
            past_length = past_key_values[0][0].shape[2]
        else:
            past_length = 0
            
        total_length = past_length + seq_length
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.full((seq_length, total_length), float("-inf"), device=attention_mask.device),
            diagonal=past_length + 1
        )
        
        # Combine with padding mask
        if attention_mask.dim() == 2:
            # [batch, seq] -> [batch, 1, 1, seq]
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * float("-inf")
            
        combined_mask = causal_mask[None, None, :, :] + attention_mask
        
        return combined_mask


class ParallelEmbedding(nn.Module):
    """
    Tensor-parallel embedding layer.
    
    Splits vocabulary across GPUs for large vocabulary models.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        config: TensorParallelConfig,
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.config = config
        self.tp_size = config.world_size
        
        # Partition vocabulary
        assert num_embeddings % self.tp_size == 0, \
            f"Vocab size {num_embeddings} not divisible by TP size {self.tp_size}"
            
        self.vocab_per_partition = num_embeddings // self.tp_size
        
        # Get partition range for this rank
        pg_manager = ProcessGroupManager()
        tp_rank = pg_manager.get_tensor_parallel_rank()
        
        self.vocab_start_idx = tp_rank * self.vocab_per_partition
        self.vocab_end_idx = self.vocab_start_idx + self.vocab_per_partition
        
        # Create partitioned embedding
        self.weight = nn.Parameter(
            torch.empty(self.vocab_per_partition, embedding_dim)
        )
        
        # Initialize
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        
        # Handle padding index
        if padding_idx is not None and self.vocab_start_idx <= padding_idx < self.vocab_end_idx:
            local_padding_idx = padding_idx - self.vocab_start_idx
            with torch.no_grad():
                self.weight[local_padding_idx].fill_(0)
                
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with parallel embedding.
        
        Each GPU looks up tokens in its partition, zeros for out-of-range,
        then all-reduce to combine results.
        """
        # Create mask for tokens in this partition
        mask = (input_ids >= self.vocab_start_idx) & (input_ids < self.vocab_end_idx)
        
        # Compute local indices
        local_indices = torch.where(
            mask,
            input_ids - self.vocab_start_idx,
            torch.zeros_like(input_ids)
        )
        
        # Embedding lookup
        embeddings = F.embedding(local_indices, self.weight)
        
        # Zero out embeddings for tokens not in this partition
        embeddings = embeddings * mask.unsqueeze(-1).float()
        
        # All-reduce to combine partitions
        if self.tp_size > 1 and dist.is_initialized():
            dist.all_reduce(embeddings, group=self.config.process_group)
            
        return embeddings


def create_tensor_parallel_model(
    base_model: nn.Module,
    config: TensorParallelConfig
) -> TensorParallelModel:
    """
    Create a tensor-parallel version of a model.
    
    Factory function to convert existing model to tensor-parallel.
    """
    logger.info(f"Creating tensor-parallel model with TP size {config.world_size}")
    
    tp_model = TensorParallelModel(config)
    
    # Copy weights if base_model provided
    if base_model is not None:
        _copy_weights_to_tp_model(base_model, tp_model, config)
        
    return tp_model


def _copy_weights_to_tp_model(
    source: nn.Module,
    target: TensorParallelModel,
    config: TensorParallelConfig
) -> None:
    """Copy and shard weights from source model to tensor-parallel model."""
    pg_manager = ProcessGroupManager()
    tp_rank = pg_manager.get_tensor_parallel_rank()
    tp_size = config.world_size
    
    source_state = source.state_dict()
    target_state = target.state_dict()
    
    for name, param in target.named_parameters():
        # Find corresponding source parameter
        source_name = _map_tp_name_to_source(name)
        
        if source_name in source_state:
            source_param = source_state[source_name]
            
            # Shard if necessary
            if "column_parallel" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name:
                # Column parallel: shard output dimension
                chunk_size = source_param.shape[0] // tp_size
                start_idx = tp_rank * chunk_size
                end_idx = start_idx + chunk_size
                param.data.copy_(source_param[start_idx:end_idx])
                
            elif "row_parallel" in name or "down_proj" in name or "o_proj" in name:
                # Row parallel: shard input dimension
                chunk_size = source_param.shape[1] // tp_size
                start_idx = tp_rank * chunk_size
                end_idx = start_idx + chunk_size
                param.data.copy_(source_param[:, start_idx:end_idx])
                
            else:
                # Replicate
                param.data.copy_(source_param)
                
    logger.info(f"Copied weights to tensor-parallel model (rank {tp_rank})")


def _map_tp_name_to_source(tp_name: str) -> str:
    """Map tensor-parallel parameter name to source model name."""
    # Simple mapping - extend as needed for specific models
    return tp_name.replace("_parallel", "")
