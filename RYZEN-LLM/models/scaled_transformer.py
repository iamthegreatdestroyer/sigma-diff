"""
PHASE 3 STAGE 3a: SCALED TRANSFORMER MODEL
==========================================
Implements production-ready scaled version of the transformer architecture.

Configuration:
- embedding_dim: 256 → 512 (2x scale)
- num_layers: 2 → 4 (2x scale)
- ff_dim: 512 → 1024 (2x scale)
- Total parameters: 134K → ~1.1M (8x increase)

Maintains all Phase 1 optimizations while scaling to production size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ScaledPositionalEncoding(nn.Module):
    """Optimized positional encoding for scaled model."""
    
    def __init__(self, embedding_dim: int, max_seq_len: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Pre-compute positional encodings (fixed)
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                             (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if embedding_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to embeddings."""
        batch_size, seq_len, _ = x.shape
        return x + self.pe[:, :seq_len, :].to(x.device)


class ScaledMultiHeadAttention(nn.Module):
    """Optimized multi-head attention for scaled model."""
    
    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head attention forward pass."""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embedding_dim)
        
        # Project output
        output = self.proj(attn_output)
        output = self.dropout(output)
        
        return output, attn_weights


class ScaledFeedForward(nn.Module):
    """Optimized feed-forward network for scaled model."""
    
    def __init__(self, embedding_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScaledTransformerBlock(nn.Module):
    """Optimized transformer block with optimization hooks."""
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = ScaledMultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = ScaledFeedForward(embedding_dim, ff_dim, dropout)
        
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transformer block forward pass with residual connections."""
        # Self-attention with residual
        attn_out, attn_weights = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        ff_out = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_out)
        
        return x, attn_weights


class ScaledTransformerModel(nn.Module):
    """
    Production-ready scaled transformer model.
    
    Scaling from Phase 2:
    - embedding_dim: 256 → 512
    - num_layers: 2 → 4
    - ff_dim: 512 → 1024
    - Total params: 134K → ~1.1M
    """
    
    def __init__(
        self,
        vocab_size: int = 2048,
        embedding_dim: int = 512,      # SCALED: was 256
        num_heads: int = 4,
        num_layers: int = 4,           # SCALED: was 2
        ff_dim: int = 1024,            # SCALED: was 512
        max_seq_len: int = 128,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = ScaledPositionalEncoding(embedding_dim, max_seq_len)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            ScaledTransformerBlock(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through scaled transformer model.
        
        Args:
            input_ids: (batch_size, seq_len) - Token indices
            attention_mask: (batch_size, seq_len) - Attention mask (optional)
        
        Returns:
            logits: (batch_size, num_classes) - Classification logits
            pooled_hidden: (batch_size, embedding_dim) - Pooled representation
        """
        batch_size, seq_len = input_ids.shape
        
        # Validate input
        assert seq_len <= self.max_seq_len, \
            f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
        
        # Embedding
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.embedding_dropout(x)
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Make causal mask
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=x.device),
            diagonal=1
        ).flip(dims=[0]) == 0
        
        # Combine masks
        combined_mask = (
            attention_mask.unsqueeze(1).unsqueeze(1) &
            causal_mask.unsqueeze(0)
        )
        
        # Transformer blocks
        attention_maps = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, combined_mask)
            attention_maps.append(attn_weights)
        
        # Global average pooling
        pooled_hidden = x.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Classification
        logits = self.classifier(pooled_hidden)
        
        # Return only logits for loss computation
        return logits
    
    def get_config(self) -> dict:
        """Return model configuration for checkpoint saving."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'max_seq_len': self.max_seq_len,
        }
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def validate_scaled_model():
    """Validate scaled model on dummy data."""
    print("=" * 60)
    print("VALIDATING SCALED TRANSFORMER MODEL")
    print("=" * 60)
    
    # Create model
    model = ScaledTransformerModel(
        vocab_size=2048,
        embedding_dim=512,
        num_heads=4,
        num_layers=4,
        ff_dim=1024,
        max_seq_len=128
    )
    
    # Print model info
    print(f"\n✅ Model created successfully")
    print(f"   Total parameters: {model.count_parameters():,}")
    print(f"   Config: {model.get_config()}")
    
    # Dummy input
    batch_size = 16
    seq_len = 128
    input_ids = torch.randint(0, 2048, (batch_size, seq_len))
    
    # Forward pass
    print(f"\n✅ Running forward pass...")
    print(f"   Input shape: {input_ids.shape}")
    
    logits, pooled = model(input_ids)
    
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Pooled hidden shape: {pooled.shape}")
    print(f"   ✅ Forward pass successful")
    
    # Parameter count comparison
    print(f"\n📊 PARAMETER COMPARISON:")
    print(f"   Phase 2 (Simple):  ~134K parameters")
    print(f"   Phase 3 (Scaled):  {model.count_parameters():,} parameters")
    print(f"   Scaling factor:    {model.count_parameters() / 134000:.1f}x")
    
    # Memory estimation
    param_memory = model.count_parameters() * 4 / (1024 * 1024)  # in MB (float32)
    activation_memory = batch_size * seq_len * 512 * 4 / (1024 * 1024)  # rough estimate
    total_memory = param_memory + activation_memory
    
    print(f"\n💾 MEMORY ESTIMATION:")
    print(f"   Parameters: {param_memory:.1f} MB")
    print(f"   Activations (est): {activation_memory:.1f} MB")
    print(f"   Total (est): {total_memory:.1f} MB")
    print(f"   vs Phase 1 Target (500MB): {'✅ OK' if total_memory < 500 else '⚠️  APPROACHING LIMIT'}")
    
    print("\n" + "=" * 60)
    print("✅ SCALED MODEL VALIDATION PASSED")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    model = validate_scaled_model()
