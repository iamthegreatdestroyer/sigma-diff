"""
Token Sampling Strategies
=========================
"""

import numpy as np
from typing import List, Optional


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply softmax with temperature."""
    scaled = logits / max(temperature, 1e-8)
    exp_logits = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep only top-k logits, set others to -inf."""
    if k <= 0:
        return logits
    
    indices_to_remove = logits < np.partition(logits, -k)[-k]
    logits[indices_to_remove] = -np.inf
    return logits


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: keep tokens with cumulative prob <= p."""
    if p >= 1.0:
        return logits
    
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    
    probs = softmax(sorted_logits)
    cumulative_probs = np.cumsum(probs)
    
    # Find cutoff
    cutoff_idx = np.searchsorted(cumulative_probs, p) + 1
    
    # Set logits below cutoff to -inf
    indices_to_remove = sorted_indices[cutoff_idx:]
    logits[indices_to_remove] = -np.inf
    
    return logits


def apply_repetition_penalty(
    logits: np.ndarray,
    generated_ids: List[int],
    penalty: float = 1.0,
) -> np.ndarray:
    """Apply repetition penalty to previously generated tokens."""
    if penalty == 1.0 or not generated_ids:
        return logits
    
    for token_id in set(generated_ids):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    
    return logits


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    generated_ids: Optional[List[int]] = None,
    repetition_penalty: float = 1.0,
    seed: Optional[int] = None,
) -> int:
    """
    Sample next token from logits.
    
    Args:
        logits: Raw logits from model (vocab_size,)
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        generated_ids: Previously generated tokens for repetition penalty
        repetition_penalty: Repetition penalty factor
        seed: Random seed for reproducibility
    
    Returns:
        Sampled token ID
    """
    if seed is not None:
        np.random.seed(seed)
    
    logits = logits.copy()
    
    # Apply repetition penalty
    if generated_ids:
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)
    
    # Temperature scaling
    if temperature == 0:
        # Greedy
        return int(np.argmax(logits))
    
    # Top-k filtering
    logits = top_k_filter(logits, top_k)
    
    # Top-p filtering
    logits = top_p_filter(logits, top_p)
    
    # Convert to probabilities
    probs = softmax(logits, temperature)
    
    # Sample
    return int(np.random.choice(len(probs), p=probs))
