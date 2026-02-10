"""
Mock BitNet Engine for Testing
[REF:API-MOCK] - Mock implementation when C++ bindings unavailable

This module provides a mock implementation of the BitNet engine
that can be used for testing the API server when the C++ bindings
are unavailable due to DLL loading issues.

Usage:
    The server.py will automatically use this mock when ryzen_llm_bindings
    fails to import.
"""

import random
import time
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class MockConfig:
    """Mock configuration matching BitNet config structure."""
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    vocab_size: int = 32000
    max_seq_length: int = 2048
    use_flash_attention: bool = True


class GenerationConfig:
    """Generation configuration for text generation."""
    
    def __init__(self):
        self.max_tokens: int = 100
        self.temperature: float = 0.7
        self.top_p: float = 1.0
        self.top_k: int = 50
        self.repetition_penalty: float = 1.1
        self.stop_sequences: List[str] = []


class MockBitNetEngine:
    """
    Mock BitNet engine that generates intelligent-looking responses.
    
    This is NOT a real LLM - it generates canned responses for testing
    the API pipeline. The real C++ engine will provide actual inference.
    """
    
    def __init__(self, config: MockConfig):
        self.config = config
        self._loaded = True
        self._model_name = "mock-bitnet-1.58b"
        
        # Response templates for different contexts
        self._greetings = [
            "Hello! I'm Ryzanstein, your personal AI assistant.",
            "Hi there! How can I help you today?",
            "Greetings! I'm ready to assist you.",
        ]
        
        self._capabilities = [
            "I can help with coding, research, analysis, and general questions.",
            "My capabilities include text generation, summarization, and Q&A.",
            "I'm designed for local inference on your AMD processor.",
        ]
        
        self._technical_responses = [
            "Based on my analysis, the solution involves optimizing the algorithm's time complexity.",
            "The key insight here is to leverage caching for better performance.",
            "I recommend a modular approach with clear separation of concerns.",
        ]
        
        self._mock_disclaimer = (
            "\n\n[Note: This response is from the MOCK engine. "
            "Real inference requires the C++ bindings to be properly loaded.]"
        )
    
    @property
    def is_loaded(self) -> bool:
        """Check if the engine is loaded."""
        return self._loaded
    
    def generate(self, input_tokens: List[int], config: GenerationConfig) -> List[int]:
        """
        Generate mock output tokens.
        
        This simulates token generation by returning a canned response
        converted to token IDs.
        
        Args:
            input_tokens: Input token IDs (used to seed response selection)
            config: Generation configuration
            
        Returns:
            List of output token IDs
        """
        # Use input tokens to deterministically select response type
        seed = sum(input_tokens) if input_tokens else 0
        random.seed(seed)
        
        # Select response based on pseudo-random seed
        response_pool = (
            self._greetings + 
            self._capabilities + 
            self._technical_responses
        )
        
        response = random.choice(response_pool)
        
        # Add disclaimer for clarity
        full_response = response + self._mock_disclaimer
        
        # Simple tokenization (hash-based)
        words = full_response.split()
        tokens = [hash(w) % self.config.vocab_size for w in words]
        
        # Respect max_tokens
        if config.max_tokens and len(tokens) > config.max_tokens:
            tokens = tokens[:config.max_tokens]
        
        return tokens
    
    def generate_text(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """
        Generate text from a prompt (convenience method).
        
        Args:
            prompt: Input prompt text
            config: Generation configuration
            
        Returns:
            Generated text response
        """
        if config is None:
            config = GenerationConfig()
        
        # Analyze prompt to provide contextual response
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'greet']):
            response = random.choice(self._greetings)
        elif any(word in prompt_lower for word in ['what can you', 'help', 'capabilities']):
            response = random.choice(self._capabilities)
        elif any(word in prompt_lower for word in ['code', 'program', 'algorithm', 'function']):
            response = random.choice(self._technical_responses)
        else:
            # Default response
            response = f"I understand you're asking about: '{prompt[:50]}...'. " + \
                       random.choice(self._capabilities)
        
        return response + self._mock_disclaimer
    
    def get_model_info(self) -> dict:
        """Get information about the mock model."""
        return {
            "name": self._model_name,
            "type": "mock",
            "parameters": "1.58B (simulated)",
            "quantization": "ternary (simulated)",
            "loaded": self._loaded,
            "warning": "This is a mock engine - not performing real inference"
        }


def create_bitnet_1_58b_config() -> MockConfig:
    """Create default BitNet 1.58B configuration."""
    return MockConfig(
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        vocab_size=32000,
        max_seq_length=2048,
        use_flash_attention=True
    )


def create_bitnet_3b_config() -> MockConfig:
    """Create BitNet 3B configuration."""
    return MockConfig(
        hidden_size=5120,
        num_layers=40,
        num_heads=40,
        vocab_size=32000,
        max_seq_length=4096,
        use_flash_attention=True
    )


# Alias for backward compatibility
BitNetEngine = MockBitNetEngine


# Module-level instance for testing
_default_config = create_bitnet_1_58b_config()
_default_engine = None


def get_default_engine() -> MockBitNetEngine:
    """Get or create the default engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = MockBitNetEngine(_default_config)
    return _default_engine


if __name__ == "__main__":
    # Quick test
    print("Testing Mock BitNet Engine...")
    
    config = create_bitnet_1_58b_config()
    engine = MockBitNetEngine(config)
    
    gen_config = GenerationConfig()
    gen_config.max_tokens = 50
    
    # Test token generation
    input_tokens = [100, 200, 300]  # Mock input
    output_tokens = engine.generate(input_tokens, gen_config)
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens[:10]}...")
    
    # Test text generation
    response = engine.generate_text("Hello, who are you?")
    print(f"\nPrompt: 'Hello, who are you?'")
    print(f"Response: {response}")
    
    print("\n✓ Mock engine test passed!")

