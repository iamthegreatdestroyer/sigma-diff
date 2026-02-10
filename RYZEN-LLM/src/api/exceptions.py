"""
Ryzanstein LLM Custom Exceptions
==========================
"""

from typing import Optional


class RyotError(Exception):
    """Base exception for all Ryzanstein LLM errors."""

    def __init__(self, message: str, error_code: str = "RYOT_ERROR", is_retryable: bool = False):
        super().__init__(message)
        self.error_code = error_code
        self.is_retryable = is_retryable


class ModelNotLoadedError(RyotError):
    def __init__(self, model_path: Optional[str] = None):
        message = f"Failed to load model from: {model_path}" if model_path else "No model loaded"
        super().__init__(message, "MODEL_NOT_LOADED", is_retryable=True)


class ContextTooLongError(RyotError):
    def __init__(self, input_length: int, max_length: int):
        super().__init__(f"Input length ({input_length}) exceeds context window ({max_length})", "CONTEXT_TOO_LONG")
        self.input_length = input_length
        self.max_length = max_length


class GenerationError(RyotError):
    def __init__(self, message: str, is_retryable: bool = True):
        super().__init__(message, "GENERATION_ERROR", is_retryable=is_retryable)


class CacheError(RyotError):
    def __init__(self, message: str, cache_id: Optional[str] = None):
        super().__init__(message, "CACHE_ERROR", is_retryable=True)
        self.cache_id = cache_id


class SigmaIntegrationError(RyotError):
    def __init__(self, message: str, operation: str = "unknown"):
        super().__init__(message, "SIGMA_INTEGRATION_ERROR", is_retryable=True)
        self.operation = operation


class RSUNotFoundError(RyotError):
    def __init__(self, semantic_hash: int):
        super().__init__(f"No RSU found for semantic hash: {semantic_hash:016x}", "RSU_NOT_FOUND")
        self.semantic_hash = semantic_hash


class AgentRequestError(RyotError):
    def __init__(self, agent_id: str, message: str):
        super().__init__(f"Agent {agent_id}: {message}", "AGENT_REQUEST_ERROR", is_retryable=True)
        self.agent_id = agent_id
