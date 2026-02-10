"""
Cross-Modal Fusion Layer
=========================

Production-ready cross-modal fusion for combining vision and language representations.
Implements multiple fusion strategies for multi-modal inference.

Features:
- Cross-attention fusion (vision attends to text, text attends to vision)
- Early, mid, and late fusion strategies
- Gated fusion with learnable mixing coefficients
- Efficient implementation with Flash Attention support

Sprint 2.1 - Multi-Modal Inference
Created: 2025-12-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Supported fusion strategies."""
    EARLY = "early"           # Concatenate before encoding
    CROSS_ATTENTION = "cross_attention"  # Cross-attention between modalities
    GATED = "gated"          # Gated fusion with learnable weights
    LATE = "late"            # Fuse at final layer only
    PERCEIVER = "perceiver"  # Perceiver-style latent fusion


@dataclass
class FusionConfig:
    """Configuration for fusion layer."""
    fusion_strategy: FusionStrategy = FusionStrategy.CROSS_ATTENTION
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_fusion_layers: int = 6
    intermediate_size: int = 4096
    dropout: float = 0.1
    attention_dropout: float = 0.0
    use_flash_attention: bool = True
    vision_hidden_size: int = 1024
    text_hidden_size: int = 4096
    num_latent_tokens: int = 64  # For Perceiver-style fusion
    residual_connection: bool = True
    layer_norm_eps: float = 1e-6


@dataclass
class FusionInput:
    """Input for fusion layer."""
    vision_features: torch.Tensor  # [B, vision_seq, vision_dim]
    text_features: torch.Tensor    # [B, text_seq, text_dim]
    vision_attention_mask: Optional[torch.Tensor] = None
    text_attention_mask: Optional[torch.Tensor] = None


@dataclass
class FusionOutput:
    """Output from fusion layer."""
    fused_features: torch.Tensor      # [B, seq, hidden_size]
    vision_updated: Optional[torch.Tensor] = None  # [B, vision_seq, hidden_size]
    text_updated: Optional[torch.Tensor] = None    # [B, text_seq, hidden_size]
    attention_weights: Optional[Dict[str, torch.Tensor]] = None


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: one modality attends to another.
    
    Supports bidirectional attention (vision->text and text->vision).
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attention = use_flash_attention
        
        # Query from one modality
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        # Key, Value from other modality
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(
        self,
        query_states: torch.Tensor,    # States that generate queries
        key_value_states: torch.Tensor,  # States that generate keys/values
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Cross-modal attention forward pass.
        
        Args:
            query_states: [B, query_seq, hidden_size]
            key_value_states: [B, kv_seq, hidden_size]
            attention_mask: [B, 1, query_seq, kv_seq] or [B, kv_seq]
        
        Returns:
            output: [B, query_seq, hidden_size]
            attention_weights: Optional attention weights
        """
        batch_size, query_len, _ = query_states.shape
        kv_len = key_value_states.shape[1]
        
        # Project query, key, value
        q = self.q_proj(query_states)
        k = self.k_proj(key_value_states)
        v = self.v_proj(key_value_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Process attention mask
        if attention_mask is not None and attention_mask.dim() == 2:
            # [B, kv_seq] -> [B, 1, 1, kv_seq]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * torch.finfo(q.dtype).min
        
        # Try Flash Attention
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0
            )
            attention_weights = None
        else:
            # Standard attention computation
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            attn_output = torch.matmul(attention_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, query_len, -1)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attention_weights if output_attentions else None


class FusionMLP(nn.Module):
    """Feed-forward network for fusion layer."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class CrossAttentionFusionBlock(nn.Module):
    """
    Single cross-attention fusion block.
    
    Implements bidirectional cross-attention:
    1. Vision attends to text (vision_to_text)
    2. Text attends to vision (text_to_vision)
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        
        # Vision -> Text cross-attention
        self.vision_to_text_attn = CrossModalAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout,
            use_flash_attention=config.use_flash_attention
        )
        
        # Text -> Vision cross-attention
        self.text_to_vision_attn = CrossModalAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout,
            use_flash_attention=config.use_flash_attention
        )
        
        # MLPs for each modality
        self.vision_mlp = FusionMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout
        )
        self.text_mlp = FusionMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout
        )
        
        # Layer norms
        self.vision_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.vision_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.text_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.text_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.residual = config.residual_connection
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass for bidirectional cross-attention fusion.
        
        Args:
            vision_features: [B, vision_seq, hidden_size]
            text_features: [B, text_seq, hidden_size]
            vision_mask: [B, vision_seq]
            text_mask: [B, text_seq]
        
        Returns:
            updated_vision: [B, vision_seq, hidden_size]
            updated_text: [B, text_seq, hidden_size]
            attention_weights: Optional dict with attention weights
        """
        attention_weights = {}
        
        # Vision attends to text
        normed_vision = self.vision_norm1(vision_features)
        vision_attn_out, v2t_weights = self.vision_to_text_attn(
            query_states=normed_vision,
            key_value_states=text_features,
            attention_mask=text_mask,
            output_attentions=output_attentions
        )
        if self.residual:
            vision_features = vision_features + vision_attn_out
        else:
            vision_features = vision_attn_out
        
        # Text attends to vision
        normed_text = self.text_norm1(text_features)
        text_attn_out, t2v_weights = self.text_to_vision_attn(
            query_states=normed_text,
            key_value_states=vision_features,
            attention_mask=vision_mask,
            output_attentions=output_attentions
        )
        if self.residual:
            text_features = text_features + text_attn_out
        else:
            text_features = text_attn_out
        
        # MLP for vision
        vision_features = vision_features + self.vision_mlp(
            self.vision_norm2(vision_features)
        )
        
        # MLP for text
        text_features = text_features + self.text_mlp(
            self.text_norm2(text_features)
        )
        
        if output_attentions:
            attention_weights['vision_to_text'] = v2t_weights
            attention_weights['text_to_vision'] = t2v_weights
        
        return vision_features, text_features, attention_weights if output_attentions else None


class GatedFusion(nn.Module):
    """
    Gated fusion with learnable mixing coefficients.
    
    Learns to adaptively weight vision and text contributions.
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        
        # Project to common hidden size
        self.vision_proj = nn.Linear(config.vision_hidden_size, config.hidden_size)
        self.text_proj = nn.Linear(config.text_hidden_size, config.hidden_size)
        
        # Gating mechanism
        self.gate_vision = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.gate_text = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        vision_features: torch.Tensor,  # [B, vision_seq, vision_dim]
        text_features: torch.Tensor,    # [B, text_seq, text_dim]
    ) -> torch.Tensor:
        """
        Gated fusion of vision and text features.
        
        For simplicity, uses pooled representations (mean pooling).
        
        Returns:
            fused: [B, hidden_size]
        """
        # Project to common space
        vision_proj = self.vision_proj(vision_features)  # [B, vision_seq, hidden_size]
        text_proj = self.text_proj(text_features)        # [B, text_seq, hidden_size]
        
        # Pool sequences (mean pooling)
        vision_pooled = vision_proj.mean(dim=1)  # [B, hidden_size]
        text_pooled = text_proj.mean(dim=1)      # [B, hidden_size]
        
        # Concatenate for gating
        combined = torch.cat([vision_pooled, text_pooled], dim=-1)  # [B, hidden_size * 2]
        
        # Compute gates
        gate_v = torch.sigmoid(self.gate_vision(combined))  # [B, hidden_size]
        gate_t = torch.sigmoid(self.gate_text(combined))    # [B, hidden_size]
        
        # Apply gates
        gated_vision = gate_v * vision_pooled
        gated_text = gate_t * text_pooled
        
        # Combine and project
        fused = torch.cat([gated_vision, gated_text], dim=-1)
        fused = self.output_proj(fused)
        fused = self.layer_norm(fused)
        
        return fused


class PerceiverFusion(nn.Module):
    """
    Perceiver-style fusion using learnable latent tokens.
    
    Latent tokens attend to both vision and text, creating a
    compressed multi-modal representation.
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        
        # Learnable latent tokens
        self.latent_tokens = nn.Parameter(
            torch.randn(1, config.num_latent_tokens, config.hidden_size)
        )
        
        # Project inputs to hidden size
        self.vision_proj = nn.Linear(config.vision_hidden_size, config.hidden_size)
        self.text_proj = nn.Linear(config.text_hidden_size, config.hidden_size)
        
        # Cross-attention: latents attend to inputs
        self.cross_attn_vision = CrossModalAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout,
            use_flash_attention=config.use_flash_attention
        )
        self.cross_attn_text = CrossModalAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout,
            use_flash_attention=config.use_flash_attention
        )
        
        # Self-attention on latents
        self.self_attn = CrossModalAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout,
            use_flash_attention=config.use_flash_attention
        )
        
        # MLP
        self.mlp = FusionMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm4 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perceiver-style fusion.
        
        Returns:
            latent_features: [B, num_latents, hidden_size]
        """
        batch_size = vision_features.shape[0]
        
        # Project inputs
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        
        # Expand latent tokens for batch
        latents = self.latent_tokens.expand(batch_size, -1, -1)
        
        # Latents attend to vision
        latents = latents + self.cross_attn_vision(
            query_states=self.norm1(latents),
            key_value_states=vision_proj,
            attention_mask=vision_mask
        )[0]
        
        # Latents attend to text
        latents = latents + self.cross_attn_text(
            query_states=self.norm2(latents),
            key_value_states=text_proj,
            attention_mask=text_mask
        )[0]
        
        # Self-attention on latents
        latents = latents + self.self_attn(
            query_states=self.norm3(latents),
            key_value_states=latents
        )[0]
        
        # MLP
        latents = latents + self.mlp(self.norm4(latents))
        
        return latents


class CrossModalFusionLayer(nn.Module):
    """
    Main fusion layer supporting multiple fusion strategies.
    
    Combines vision and language representations using configurable
    fusion mechanisms.
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Project inputs to common hidden size
        self.vision_proj = nn.Linear(config.vision_hidden_size, config.hidden_size)
        self.text_proj = nn.Linear(config.text_hidden_size, config.hidden_size)
        
        # Initialize fusion mechanism based on strategy
        if config.fusion_strategy == FusionStrategy.CROSS_ATTENTION:
            self.fusion_blocks = nn.ModuleList([
                CrossAttentionFusionBlock(config)
                for _ in range(config.num_fusion_layers)
            ])
        elif config.fusion_strategy == FusionStrategy.GATED:
            self.fusion = GatedFusion(config)
        elif config.fusion_strategy == FusionStrategy.PERCEIVER:
            self.fusion = PerceiverFusion(config)
        elif config.fusion_strategy == FusionStrategy.EARLY:
            # Simple concatenation + projection
            self.fusion_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        elif config.fusion_strategy == FusionStrategy.LATE:
            # Late fusion uses separate processing then combines
            self.late_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        logger.info(f"CrossModalFusionLayer initialized with strategy: {config.fusion_strategy.value}")
    
    def forward(
        self,
        fusion_input: FusionInput,
        output_attentions: bool = False
    ) -> FusionOutput:
        """
        Forward pass through fusion layer.
        
        Args:
            fusion_input: FusionInput containing vision and text features
            output_attentions: Whether to return attention weights
        
        Returns:
            FusionOutput with fused features
        """
        vision_features = fusion_input.vision_features
        text_features = fusion_input.text_features
        vision_mask = fusion_input.vision_attention_mask
        text_mask = fusion_input.text_attention_mask
        
        # Project to common hidden size
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        
        attention_weights = {}
        
        if self.config.fusion_strategy == FusionStrategy.CROSS_ATTENTION:
            # Multi-layer cross-attention fusion
            for i, block in enumerate(self.fusion_blocks):
                vision_proj, text_proj, attn = block(
                    vision_proj, text_proj,
                    vision_mask, text_mask,
                    output_attentions
                )
                if attn:
                    attention_weights[f'layer_{i}'] = attn
            
            # Concatenate for final output
            fused = torch.cat([vision_proj.mean(dim=1), text_proj.mean(dim=1)], dim=-1)
            fused = self.final_norm(
                nn.functional.linear(fused, 
                    torch.eye(self.config.hidden_size, device=fused.device)[:, :fused.shape[-1]]
                    if fused.shape[-1] != self.config.hidden_size
                    else torch.eye(self.config.hidden_size, device=fused.device)
                )
            )
            
            # Create proper fused output
            fused = (vision_proj.mean(dim=1) + text_proj.mean(dim=1)) / 2
            fused = self.final_norm(fused)
            
            return FusionOutput(
                fused_features=fused,
                vision_updated=vision_proj,
                text_updated=text_proj,
                attention_weights=attention_weights if output_attentions else None
            )
        
        elif self.config.fusion_strategy == FusionStrategy.GATED:
            fused = self.fusion(vision_features, text_features)
            return FusionOutput(fused_features=fused)
        
        elif self.config.fusion_strategy == FusionStrategy.PERCEIVER:
            latents = self.fusion(vision_features, text_features, vision_mask, text_mask)
            fused = latents.mean(dim=1)  # Pool latents
            fused = self.final_norm(fused)
            return FusionOutput(fused_features=fused)
        
        elif self.config.fusion_strategy == FusionStrategy.EARLY:
            # Pool and concatenate
            vision_pooled = vision_proj.mean(dim=1)
            text_pooled = text_proj.mean(dim=1)
            fused = torch.cat([vision_pooled, text_pooled], dim=-1)
            fused = self.fusion_proj(fused)
            fused = self.final_norm(fused)
            return FusionOutput(fused_features=fused)
        
        elif self.config.fusion_strategy == FusionStrategy.LATE:
            # Pool and combine
            vision_pooled = vision_proj.mean(dim=1)
            text_pooled = text_proj.mean(dim=1)
            fused = torch.cat([vision_pooled, text_pooled], dim=-1)
            fused = self.late_proj(fused)
            fused = self.final_norm(fused)
            return FusionOutput(fused_features=fused)
        
        else:
            raise ValueError(f"Unknown fusion strategy: {self.config.fusion_strategy}")
    
    def fuse(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convenience method for fusion.
        
        Returns:
            fused_features: [B, hidden_size]
        """
        fusion_input = FusionInput(
            vision_features=vision_features,
            text_features=text_features,
            vision_attention_mask=vision_mask,
            text_attention_mask=text_mask
        )
        output = self.forward(fusion_input)
        return output.fused_features


def create_fusion_layer(
    fusion_type: str = "cross_attention",
    vision_dim: int = 1024,
    text_dim: int = 4096,
    hidden_dim: int = 1024,
    **kwargs
) -> CrossModalFusionLayer:
    """
    Factory function to create fusion layers.
    
    Args:
        fusion_type: Type of fusion ("cross_attention", "gated", "perceiver", "early", "late")
        vision_dim: Vision feature dimension
        text_dim: Text feature dimension
        hidden_dim: Hidden dimension for fusion
        **kwargs: Additional configuration options
    
    Returns:
        Configured CrossModalFusionLayer
    """
    strategy_mapping = {
        "cross_attention": FusionStrategy.CROSS_ATTENTION,
        "gated": FusionStrategy.GATED,
        "perceiver": FusionStrategy.PERCEIVER,
        "early": FusionStrategy.EARLY,
        "late": FusionStrategy.LATE
    }
    
    config = FusionConfig(
        fusion_strategy=strategy_mapping.get(fusion_type, FusionStrategy.CROSS_ATTENTION),
        vision_hidden_size=vision_dim,
        text_hidden_size=text_dim,
        hidden_size=hidden_dim,
        **kwargs
    )
    
    return CrossModalFusionLayer(config)


if __name__ == "__main__":
    # Test fusion layer
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Cross-Modal Fusion Layer...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create fusion layer
    config = FusionConfig(
        fusion_strategy=FusionStrategy.CROSS_ATTENTION,
        hidden_size=512,
        vision_hidden_size=768,
        text_hidden_size=1024,
        num_fusion_layers=2,
        num_attention_heads=8
    )
    
    fusion = CrossModalFusionLayer(config).to(device)
    
    # Create dummy inputs
    batch_size = 4
    vision_seq = 196  # 14x14 patches
    text_seq = 32
    
    vision_features = torch.randn(batch_size, vision_seq, 768).to(device)
    text_features = torch.randn(batch_size, text_seq, 1024).to(device)
    
    # Forward pass
    output = fusion.fuse(vision_features, text_features)
    
    print(f"Fused features shape: {output.shape}")
    print("Fusion layer test passed!")
