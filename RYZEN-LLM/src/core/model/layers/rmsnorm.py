"""
RMS Normalization
=================
"""

import numpy as np


class RMSNorm:
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm, commonly used in LLaMA-style models.
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Weight parameter (set by model loader)
        self.weight: np.ndarray = np.ones(hidden_size, dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor (..., hidden_size)
        
        Returns:
            Normalized tensor
        """
        # Compute RMS
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        x_norm = x * np.reciprocal(np.sqrt(variance + self.eps))
        
        # Apply weight
        return x_norm * self.weight
