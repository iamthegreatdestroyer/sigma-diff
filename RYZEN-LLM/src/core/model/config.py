"""
BitNet Model Configuration
==========================
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
import json


class ActivationType(Enum):
    SILU = "silu"
    GELU = "gelu"
    RELU = "relu"


class NormType(Enum):
    RMSNORM = "rmsnorm"
    LAYERNORM = "layernorm"


@dataclass
class BitNetConfig:
    """
    Configuration for BitNet model architecture.
    
    Supports 1.58-bit ternary quantization (-1, 0, +1).
    """
    
    # Architecture
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32  # For GQA
    
    # Vocabulary
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    
    # Quantization
    bits_per_weight: float = 1.58
    use_ternary: bool = True
    quantization_group_size: int = 128
    
    # Normalization
    rms_norm_eps: float = 1e-6
    norm_type: NormType = NormType.RMSNORM
    
    # Attention
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_dropout: float = 0.0
    
    # FFN
    activation: ActivationType = ActivationType.SILU
    mlp_bias: bool = False
    
    # Special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    
    # Inference
    use_cache: bool = True
    tie_word_embeddings: bool = False
    
    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> "BitNetConfig":
        """Load config from pretrained model directory."""
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Map config keys to our format
        return cls(
            hidden_size=config_dict.get("hidden_size", 4096),
            intermediate_size=config_dict.get("intermediate_size", 11008),
            num_hidden_layers=config_dict.get("num_hidden_layers", 32),
            num_attention_heads=config_dict.get("num_attention_heads", 32),
            num_key_value_heads=config_dict.get("num_key_value_heads", 32),
            vocab_size=config_dict.get("vocab_size", 32000),
            max_position_embeddings=config_dict.get("max_position_embeddings", 4096),
            rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
            rope_theta=config_dict.get("rope_theta", 10000.0),
            bos_token_id=config_dict.get("bos_token_id", 1),
            eos_token_id=config_dict.get("eos_token_id", 2),
            pad_token_id=config_dict.get("pad_token_id", 0),
        )
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "bits_per_weight": self.bits_per_weight,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
        }
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_attention_heads
    
    @property
    def estimated_size_bytes(self) -> int:
        """Estimate model size in bytes."""
        # Rough estimate for ternary weights
        total_params = (
            self.vocab_size * self.hidden_size  # Embeddings
            + self.num_hidden_layers * (
                4 * self.hidden_size * self.hidden_size  # Attention
                + 3 * self.hidden_size * self.intermediate_size  # FFN
            )
        )
        # 1.58 bits per weight + overhead
        return int(total_params * self.bits_per_weight / 8 * 1.1)
