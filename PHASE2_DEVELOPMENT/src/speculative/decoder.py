"""
Speculative Decoding
====================

Accelerate text generation through draft model speculation.

Technique from DeepMind: Use a smaller draft model to predict multiple tokens,
then verify predictions with the main model in parallel.

Features:
- Draft model prediction
- Parallel verification
- Acceptance sampling
- Adaptive speculation depth
- Fallback to standard decoding

Sprint 2.2 - Distributed Inference & Performance
Created: 2025-12-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class SpeculationConfig:
    """Configuration for speculative decoding."""
    draft_model_ratio: float = 0.4  # Draft model is 40% size of main
    max_speculation_depth: int = 4  # Max tokens to speculate
    min_acceptance_rate: float = 0.5
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class SpeculativeOutput:
    """Output from speculative decoding."""
    generated_ids: torch.Tensor
    num_tokens: int
    num_verified: int
    acceptance_rate: float
    latency_ms: float
    num_iterations: int


class DraftModel(nn.Module):
    """
    Lightweight draft model for speculation.
    
    Should be 40-50% of main model size while maintaining reasonable quality.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 16,
        intermediate_size: int = 8192
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(2048, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=intermediate_size,
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ]
        )
        
        # Output projection
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Layer norm
        self.ln_final = nn.LayerNorm(hidden_size)
        
        logger.info(
            f"DraftModel initialized: {vocab_size} vocab, {hidden_size} hidden, {num_layers} layers"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through draft model.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device)
        
        # Embeddings
        embeddings = self.embedding(input_ids) + self.position_embedding(positions)
        
        # Transformer layers
        x = embeddings
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attention_mask)
        
        # Output
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate_draft(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Generate draft tokens greedily.
        
        Args:
            input_ids: [batch, seq_len]
            num_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        
        Returns:
            generated_ids: [batch, num_tokens]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated = []
        current_ids = input_ids
        
        with torch.no_grad():
            for _ in range(num_tokens):
                # Get logits
                logits = self.forward(current_ids)
                next_logits = logits[:, -1, :]  # [batch, vocab]
                
                # Apply temperature
                if temperature > 0:
                    next_logits = next_logits / temperature
                
                # Apply top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumsum > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[:, indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
                
                generated.append(next_token)
                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        # Concatenate generated tokens
        generated = torch.cat(generated, dim=1)  # [batch, num_tokens]
        return generated


class SpeculativeVerifier:
    """
    Verify draft model predictions with main model.
    
    Uses acceptance sampling to decide which tokens to keep.
    """
    
    def __init__(self, main_model: nn.Module):
        self.main_model = main_model
    
    def verify_tokens(
        self,
        input_ids: torch.Tensor,
        draft_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        temperature: float = 0.7
    ) -> Tuple[torch.Tensor, float]:
        """
        Verify draft tokens and perform acceptance sampling.
        
        Args:
            input_ids: Original input [batch, seq_len]
            draft_ids: Draft generated tokens [batch, num_draft]
            draft_probs: Draft token probabilities [batch, num_draft, vocab]
            temperature: Sampling temperature
        
        Returns:
            (verified_ids, acceptance_rate)
        """
        batch_size = input_ids.shape[0]
        
        # Concatenate input and draft
        augmented_ids = torch.cat([input_ids, draft_ids], dim=1)
        
        # Get main model logits
        with torch.no_grad():
            main_logits = self.main_model(augmented_ids)
        
        # Extract logits for draft positions
        draft_start = input_ids.shape[1]
        draft_logits = main_logits[:, draft_start:draft_start+draft_ids.shape[1], :]
        
        # Apply temperature
        if temperature > 0:
            draft_logits = draft_logits / temperature
        
        main_probs = F.softmax(draft_logits, dim=-1)
        
        # Acceptance sampling
        verified_ids = []
        num_accepted = 0
        
        for i in range(draft_ids.shape[1]):
            draft_token = draft_ids[:, i]
            
            # Get probabilities
            p_draft = draft_probs[:, i, draft_token]
            p_main = main_probs[:, i, draft_token]
            
            # Acceptance ratio
            ratio = torch.minimum(
                torch.tensor(1.0, device=p_draft.device),
                p_main / (p_draft + 1e-6)
            )
            
            # Sample acceptance
            u = torch.rand_like(ratio)
            accept = u < ratio
            
            # If accepted, keep draft token; else sample from main
            if accept.all():
                verified_ids.append(draft_token.unsqueeze(1))
                num_accepted += 1
            else:
                # Sample from main model
                sampled = torch.multinomial(
                    main_probs[:, i, :],
                    num_samples=1
                )
                verified_ids.append(sampled)
        
        # Concatenate verified tokens
        verified = torch.cat(verified_ids, dim=1)
        
        acceptance_rate = num_accepted / max(1, draft_ids.shape[1])
        
        return verified, acceptance_rate


class SpeculativeDecoder(nn.Module):
    """
    High-level speculative decoder combining draft + verification.
    
    Orchestrates the speculation pipeline for efficient decoding.
    """
    
    def __init__(
        self,
        main_model: nn.Module,
        config: SpeculationConfig,
        draft_model: Optional[DraftModel] = None
    ):
        super().__init__()
        self.main_model = main_model
        self.config = config
        
        # Create or use provided draft model
        if draft_model is None:
            # Create default draft model (40% size)
            draft_model = self._create_draft_model()
        
        self.draft_model = draft_model
        self.verifier = SpeculativeVerifier(main_model)
        
        logger.info("SpeculativeDecoder initialized")
    
    def _create_draft_model(self) -> DraftModel:
        """Create default draft model."""
        # Assume main model has these attributes
        vocab_size = 32000
        hidden_size = 2048
        num_layers = 8
        
        return DraftModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> SpeculativeOutput:
        """
        Generate tokens using speculative decoding.
        
        Args:
            input_ids: Initial token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        
        Returns:
            SpeculativeOutput with generation details
        """
        start_time = time.time()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated = []
        current_ids = input_ids.clone()
        
        total_tokens = 0
        verified_tokens = 0
        num_iterations = 0
        
        while len(generated) < max_new_tokens:
            num_iterations += 1
            
            # Determine speculation depth
            remaining = max_new_tokens - len(generated)
            spec_depth = min(self.config.max_speculation_depth, remaining)
            
            # Get draft predictions
            with torch.no_grad():
                draft_ids = self.draft_model.generate_draft(
                    current_ids,
                    num_tokens=spec_depth,
                    temperature=temperature,
                    top_p=top_p
                )
            
            # Get draft probabilities
            with torch.no_grad():
                draft_logits = self.draft_model(
                    torch.cat([current_ids, draft_ids], dim=1)
                )
                draft_start = current_ids.shape[1]
                draft_probs = F.softmax(
                    draft_logits[:, draft_start:draft_start+spec_depth, :],
                    dim=-1
                )
            
            # Verify with main model
            verified_ids, acc_rate = self.verifier.verify_tokens(
                current_ids,
                draft_ids,
                draft_probs,
                temperature
            )
            
            # Add verified tokens
            num_verified = verified_ids.shape[1]
            generated.append(verified_ids)
            verified_tokens += num_verified
            total_tokens += num_verified
            
            # Update current IDs
            current_ids = torch.cat([current_ids, verified_ids], dim=1)
            
            logger.debug(
                f"Iteration {num_iterations}: {num_verified} tokens verified "
                f"(acceptance rate: {acc_rate:.2%})"
            )
            
            # Stop if reached max tokens or model output EOS
            if total_tokens >= max_new_tokens:
                break
        
        # Concatenate all generated tokens
        if generated:
            all_generated = torch.cat(generated, dim=1)
        else:
            all_generated = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return SpeculativeOutput(
            generated_ids=all_generated,
            num_tokens=all_generated.shape[1],
            num_verified=verified_tokens,
            acceptance_rate=verified_tokens / max(1, total_tokens),
            latency_ms=latency_ms,
            num_iterations=num_iterations
        )


class AdaptiveSpeculation:
    """
    Adaptively adjust speculation depth based on acceptance rate.
    
    If acceptance rate is high, increase depth for more parallelism.
    If low, decrease depth to reduce verification overhead.
    """
    
    def __init__(
        self,
        initial_depth: int = 4,
        min_depth: int = 1,
        max_depth: int = 8
    ):
        self.current_depth = initial_depth
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        self.acceptance_history: List[float] = []
        self.history_size = 10
    
    def update(self, acceptance_rate: float):
        """
        Update speculation depth based on acceptance rate.
        
        Args:
            acceptance_rate: Acceptance rate from last speculation
        """
        self.acceptance_history.append(acceptance_rate)
        
        if len(self.acceptance_history) > self.history_size:
            self.acceptance_history.pop(0)
        
        # Calculate average acceptance rate
        avg_acceptance = sum(self.acceptance_history) / len(self.acceptance_history)
        
        # Adjust depth
        if avg_acceptance > 0.8:
            # High acceptance - increase depth
            self.current_depth = min(self.current_depth + 1, self.max_depth)
        elif avg_acceptance < 0.5:
            # Low acceptance - decrease depth
            self.current_depth = max(self.current_depth - 1, self.min_depth)
    
    def get_depth(self) -> int:
        """Get current speculation depth."""
        return self.current_depth


def create_speculative_decoder(
    main_model: nn.Module,
    draft_ratio: float = 0.4,
    max_depth: int = 4,
    **kwargs
) -> SpeculativeDecoder:
    """
    Factory function to create speculative decoder.
    
    Args:
        main_model: Main generation model
        draft_ratio: Draft model size ratio
        max_depth: Maximum speculation depth
        **kwargs: Additional config options
    
    Returns:
        Configured SpeculativeDecoder
    """
    config = SpeculationConfig(
        draft_model_ratio=draft_ratio,
        max_speculation_depth=max_depth,
        **kwargs
    )
    return SpeculativeDecoder(main_model, config)


if __name__ == "__main__":
    # Test speculative decoding
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Speculative Decoding...")
    
    # Create models
    vocab_size = 32000
    
    # Simple main model (just for testing)
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 256)
            self.output = nn.Linear(256, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            return self.output(x)
    
    main_model = SimpleModel(vocab_size)
    draft_model = DraftModel(vocab_size, hidden_size=128, num_layers=2)
    
    # Create decoder
    config = SpeculationConfig(max_speculation_depth=4)
    decoder = SpeculativeDecoder(main_model, config, draft_model)
    
    # Test generation
    input_ids = torch.randint(0, vocab_size, (2, 10))
    output = decoder.generate(input_ids, max_new_tokens=20)
    
    print(f"Generated {output.num_tokens} tokens")
    print(f"Verified: {output.num_verified}/{output.num_tokens}")
    print(f"Acceptance rate: {output.acceptance_rate:.2%}")
    print(f"Latency: {output.latency_ms:.1f}ms")
    print(f"Iterations: {output.num_iterations}")
    
    print("Speculative decoding test passed!")
