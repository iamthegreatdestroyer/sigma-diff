"""
BitNet Distributed Model Wrapper

Implements ParallelModelWrapper for BitNet transformer models.
Replaces standard layers with tensor-parallel versions for distributed inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import logging

from .architecture import ParallelModelWrapper, DistributedConfig, CommunicationHandler
from .tensor_parallel import (
    RowParallelLinear, ColumnParallelLinear,
    ParallelAttention, ParallelMLP
)
from .communication import NCCLCommunicator
from .orchestrator import MultiGPUOrchestrator

logger = logging.getLogger(__name__)


class BitNetParallelModelWrapper(ParallelModelWrapper):
    """Tensor-parallel wrapper for BitNet transformer models.

    Replaces attention and MLP layers with parallel versions.
    Handles weight distribution and gradient synchronization.
    """

    def __init__(self,
                 model: nn.Module,
                 config: DistributedConfig,
                 comm_handler: Optional[CommunicationHandler] = None):
        """Initialize BitNet parallel model wrapper.

        Args:
            model: Original BitNet model to parallelize
            config: Distributed configuration
            comm_handler: Communication handler (auto-created if None)
        """
        if comm_handler is None:
            comm_handler = NCCLCommunicator()

        super().__init__(model, config, comm_handler)

        self.orchestrator = MultiGPUOrchestrator(
            rank=config.rank,
            world_size=config.world_size,
            backend=config.backend,
            device=config.device
        )

        # Track original layers for weight distribution
        self.original_layers = {}
        self.parallel_layers = {}

        # Initialize distributed environment
        self.orchestrator.initialize(
            master_addr=config.master_addr,
            master_port=config.master_port
        )

        logger.info(f"BitNetParallelModelWrapper initialized: rank={config.rank}, "
                   f"world_size={config.world_size}")

    def apply_tensor_parallelism(self) -> None:
        """Replace model layers with tensor-parallel versions.

        Walks through the model and replaces:
        - Linear layers in attention with ParallelAttention
        - Linear layers in MLP with ParallelMLP
        - Embedding layers with sharded versions (future)
        """
        logger.info("Applying tensor parallelism to BitNet model...")

        # Replace transformer layers
        if hasattr(self.model, 'layers') and self.model.layers:
            for i, layer in enumerate(self.model.layers):
                self._parallelize_transformer_layer(layer, i)

        logger.info(f"✓ Tensor parallelism applied to {len(self.parallel_layers)} layers")

    def _parallelize_transformer_layer(self, layer, layer_idx: int) -> None:
        """Replace layers in a single transformer layer.

        Args:
            layer: Transformer layer to parallelize
            layer_idx: Layer index for logging
        """
        # Store original layers for weight mapping
        self.original_layers[f'layer_{layer_idx}_attention'] = layer.attention
        self.original_layers[f'layer_{layer_idx}_mlp'] = layer.mlp

        # Replace attention with parallel version
        if hasattr(layer.attention, 'q_proj') and hasattr(layer.attention, 'k_proj'):
            # Get attention configuration from original layer
            hidden_size = layer.attention.hidden_size
            num_heads = layer.attention.num_heads
            num_kv_heads = getattr(layer.attention, 'num_kv_heads', num_heads)

            # Create parallel attention
            parallel_attention = ParallelAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                comm_handler=self.comm_handler
            )

            layer.attention = parallel_attention
            self.parallel_layers[f'layer_{layer_idx}_attention'] = parallel_attention

        # Replace MLP with parallel version
        if hasattr(layer.mlp, 'gate_proj') and hasattr(layer.mlp, 'up_proj'):
            # Get MLP configuration
            hidden_size = layer.mlp.hidden_size
            intermediate_size = layer.mlp.intermediate_size

            # Create parallel MLP
            parallel_mlp = ParallelMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                comm_handler=self.comm_handler
            )

            layer.mlp = parallel_mlp
            self.parallel_layers[f'layer_{layer_idx}_mlp'] = parallel_mlp

        logger.debug(f"✓ Parallelized layer {layer_idx}")

    def distribute_weights(self) -> None:
        """Distribute model weights across GPUs according to parallelism strategy.

        Maps original weights to parallel layer weights using sharding patterns.
        """
        logger.info("Distributing weights across GPUs...")

        # Use weight distributor for sharding
        from .model_loader import WeightDistributor
        distributor = WeightDistributor(
            rank=self.config.rank,
            world_size=self.config.world_size,
            tp_size=self.config.tensor_parallel_size
        )

        # Distribute attention weights
        for layer_name, parallel_layer in self.parallel_layers.items():
            if 'attention' in layer_name:
                self._distribute_attention_weights(
                    parallel_layer, self.original_layers[layer_name], distributor
                )
            elif 'mlp' in layer_name:
                self._distribute_mlp_weights(
                    parallel_layer, self.original_layers[layer_name], distributor
                )

        logger.info("✓ Weight distribution completed")

    def _distribute_attention_weights(self,
                                    parallel_attn: ParallelAttention,
                                    original_attn,
                                    distributor: 'WeightDistributor') -> None:
        """Distribute attention layer weights.

        Args:
            parallel_attn: Parallel attention layer
            original_attn: Original attention layer
            distributor: Weight distributor
        """
        # Distribute Q, K, V projections (column-parallel)
        if hasattr(original_attn, 'q_proj'):
            qw, qb = distributor.shard_linear_layer_column_wise(
                original_attn.q_proj.weight, original_attn.q_proj.bias
            )
            parallel_attn.q_proj.weight.data.copy_(qw)
            if qb is not None:
                parallel_attn.q_proj.bias.data.copy_(qb)

        if hasattr(original_attn, 'k_proj'):
            kw, kb = distributor.shard_linear_layer_column_wise(
                original_attn.k_proj.weight, original_attn.k_proj.bias
            )
            parallel_attn.k_proj.weight.data.copy_(kw)
            if kb is not None:
                parallel_attn.k_proj.bias.data.copy_(kb)

        if hasattr(original_attn, 'v_proj'):
            vw, vb = distributor.shard_linear_layer_column_wise(
                original_attn.v_proj.weight, original_attn.v_proj.bias
            )
            parallel_attn.v_proj.weight.data.copy_(vw)
            if vb is not None:
                parallel_attn.v_proj.bias.data.copy_(vb)

        # Distribute output projection (row-parallel)
        if hasattr(original_attn, 'o_proj'):
            ow, ob = distributor.shard_linear_layer_row_wise(
                original_attn.o_proj.weight, original_attn.o_proj.bias
            )
            parallel_attn.o_proj.weight.data.copy_(ow)
            if ob is not None:
                parallel_attn.o_proj.bias.data.copy_(ob)

    def _distribute_mlp_weights(self,
                               parallel_mlp: ParallelMLP,
                               original_mlp,
                               distributor: 'WeightDistributor') -> None:
        """Distribute MLP layer weights.

        Args:
            parallel_mlp: Parallel MLP layer
            original_mlp: Original MLP layer
            distributor: Weight distributor
        """
        # Distribute gate and up projections (column-parallel)
        if hasattr(original_mlp, 'gate_proj'):
            gw, gb = distributor.shard_linear_layer_column_wise(
                original_mlp.gate_proj.weight, original_mlp.gate_proj.bias
            )
            parallel_mlp.gate_proj.weight.data.copy_(gw)
            if gb is not None:
                parallel_mlp.gate_proj.bias.data.copy_(gb)

        if hasattr(original_mlp, 'up_proj'):
            uw, ub = distributor.shard_linear_layer_column_wise(
                original_mlp.up_proj.weight, original_mlp.up_proj.bias
            )
            parallel_mlp.up_proj.weight.data.copy_(uw)
            if ub is not None:
                parallel_mlp.up_proj.bias.data.copy_(ub)

        # Distribute down projection (row-parallel)
        if hasattr(original_mlp, 'down_proj'):
            dw, db = distributor.shard_linear_layer_row_wise(
                original_mlp.down_proj.weight, original_mlp.down_proj.bias
            )
            parallel_mlp.down_proj.weight.data.copy_(dw)
            if db is not None:
                parallel_mlp.down_proj.bias.data.copy_(db)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward pass with distributed orchestration.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            **kwargs: Additional model-specific arguments

        Returns:
            Model output (logits, embeddings, etc.)
        """
        # For now, delegate to original model forward
        # In full implementation, this would handle distributed input sharding
        return self.model.forward(input_ids, attention_mask, **kwargs)

    def synchronize_gradients(self) -> None:
        """Synchronize gradients across ranks after backward pass.

        Performs all-reduce on gradients of parallel layers.
        """
        if not self.training:
            return

        logger.debug("Synchronizing gradients across ranks...")

        # Synchronize gradients for all parallel layers
        for parallel_layer in self.parallel_layers.values():
            for param in parallel_layer.parameters():
                if param.grad is not None:
                    self.comm_handler.all_reduce(param.grad, op="sum")

        logger.debug("✓ Gradient synchronization completed")

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics for this rank.

        Returns:
            Dictionary with memory stats
        """
        device = self.orchestrator.get_device()
        if device.type == 'cuda':
            return {
                'allocated_gb': torch.cuda.memory_allocated(device) / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved(device) / (1024**3),
                'max_allocated_gb': torch.cuda.max_memory_allocated(device) / (1024**3),
            }
        return {}

    def cleanup(self) -> None:
        """Cleanup distributed resources."""
        self.orchestrator.cleanup()
        logger.info("BitNetParallelModelWrapper cleaned up")
