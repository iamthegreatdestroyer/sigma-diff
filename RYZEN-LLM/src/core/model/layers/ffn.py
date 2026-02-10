"""
BitNet Feed-Forward Network
============================
"""

import numpy as np
from typing import Optional

from ..quantization import QuantizedTensor, ternary_matmul


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation (Swish)."""
    return x * (1.0 / (1.0 + np.exp(-x)))


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class BitNetMLP:
    """
    BitNet MLP layer with gated activation.
    
    Uses SwiGLU: gate * silu(up) structure.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Activation function
        self.activation = silu if activation == "silu" else gelu
        
        # Weights (set by model loader)
        self.gate_proj: Optional[QuantizedTensor] = None
        self.up_proj: Optional[QuantizedTensor] = None
        self.down_proj: Optional[QuantizedTensor] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through MLP.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
        
        Returns:
            Output tensor (batch, seq_len, hidden_size)
        """
        # Gate projection
        gate = ternary_matmul(x, self.gate_proj)
        gate = self.activation(gate)
        
        # Up projection
        up = ternary_matmul(x, self.up_proj)
        
        # Gated activation
        hidden = gate * up
        
        # Down projection
        output = ternary_matmul(hidden, self.down_proj)
        
        return output
