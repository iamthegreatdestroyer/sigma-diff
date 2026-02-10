"""
Ryzanstein API Integration for Continue.dev
Sprint 5: OpenAI-compatible API endpoint

Provides OpenAI-compatible interface to Ryzanstein LLM
for seamless integration with Continue.dev.
"""

import aiohttp
import asyncio
import json
import logging
from typing import Optional, Dict, List, AsyncIterator, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# API Data Models
# =============================================================================

class RoleType(Enum):
    """Message role types."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Chat message."""
    role: RoleType
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content
        }


@dataclass
class ChatCompletionRequest:
    """Chat completion request."""
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    stream: bool = True
    context_length: int = 4096
    request_id: Optional[str] = None

    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream": self.stream,
        }


@dataclass
class ChatCompletionResponse:
    """Chat completion response."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


# =============================================================================
# Ryzanstein API Client
# =============================================================================

class RyzansteinAPIClient:
    """
    OpenAI-compatible API client for Ryzanstein LLM.
    
    Provides standard OpenAI Chat Completions API interface
    for seamless integration with Continue.dev.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        model: str = "ryzanstein-7b",
        timeout: int = 30000,  # milliseconds
    ):
        """Initialize Ryzanstein API client."""
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key or "local-development"
        self.model = model
        self.timeout = aiohttp.ClientTimeout(seconds=timeout / 1000.0)
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._error_count = 0

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _ensure_session(self):
        """Ensure session is initialized."""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def create_chat_completion(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        stream: bool = True,
    ) -> ChatCompletionResponse:
        """
        Create a chat completion request (non-streaming).
        
        Args:
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream response
            
        Returns:
            ChatCompletionResponse
        """
        await self._ensure_session()
        
        request = ChatCompletionRequest(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            self._request_count += 1
            logger.debug(
                f"[Request #{self._request_count}] POST {self.api_url}/v1/chat/completions"
            )

            async with self.session.post(
                f"{self.api_url}/v1/chat/completions",
                json=request.to_dict(),
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self._error_count += 1
                    logger.error(
                        f"API error ({response.status}): {error_text}"
                    )
                    raise RuntimeError(
                        f"Ryzanstein API error: {response.status} - {error_text}"
                    )

                data = await response.json()
                logger.debug(f"Response received: {len(data.get('choices', []))} choices")
                return ChatCompletionResponse(**data)

        except asyncio.TimeoutError:
            self._error_count += 1
            logger.error("API request timeout")
            raise
        except Exception as e:
            self._error_count += 1
            logger.error(f"API request failed: {e}")
            raise

    async def create_chat_completion_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        """
        Create a streaming chat completion request.
        
        Args:
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            
        Yields:
            Text chunks from the streaming response
        """
        await self._ensure_session()
        
        request = ChatCompletionRequest(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,  # Always stream for this method
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            self._request_count += 1
            logger.debug(
                f"[Streaming Request #{self._request_count}] POST {self.api_url}/v1/chat/completions"
            )

            async with self.session.post(
                f"{self.api_url}/v1/chat/completions",
                json=request.to_dict(),
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self._error_count += 1
                    logger.error(f"API error ({response.status}): {error_text}")
                    raise RuntimeError(
                        f"Ryzanstein API error: {response.status} - {error_text}"
                    )

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            logger.debug("Stream completed")
                            break

                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse stream chunk: {e}")
                            continue

        except asyncio.TimeoutError:
            self._error_count += 1
            logger.error("Streaming request timeout")
            raise
        except Exception as e:
            self._error_count += 1
            logger.error(f"Streaming request failed: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if API is healthy and accessible.
        
        Returns:
            True if API is healthy, False otherwise
        """
        await self._ensure_session()
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }

            async with self.session.get(
                f"{self.api_url}/health",
                headers=headers,
            ) as response:
                return response.status == 200

        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Model information dictionary
        """
        await self._ensure_session()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        async with self.session.get(
            f"{self.api_url}/v1/models/{self.model}",
            headers=headers,
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.warning(f"Failed to get model info: {response.status}")
                return {}

    def get_stats(self) -> Dict[str, int]:
        """Get request statistics."""
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": (
                self._error_count / self._request_count
                if self._request_count > 0
                else 0
            ),
        }


# =============================================================================
# Continue.dev Integration
# =============================================================================

class RyzansteinProvider:
    """
    Ryzanstein provider for Continue.dev.
    
    Implements the Continue.dev provider interface for Ryzanstein LLM.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        model: str = "ryzanstein-7b",
    ):
        """Initialize Ryzanstein provider."""
        self.client = RyzansteinAPIClient(
            api_url=api_url,
            api_key=api_key,
            model=model,
        )
        self.logger = logger

    async def complete(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: Input prompt
            context: Optional context information
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        message = Message(role=RoleType.USER, content=prompt)
        
        # Add system message if provided
        messages = []
        if context and "system_prompt" in context:
            messages.append(
                Message(
                    role=RoleType.SYSTEM,
                    content=context["system_prompt"]
                )
            )
        messages.append(message)

        try:
            response = await self.client.create_chat_completion(
                messages=messages,
                **kwargs,
            )

            # Extract text from response
            if response.choices and len(response.choices) > 0:
                return response.choices[0]["message"]["content"]
            else:
                return ""

        except Exception as e:
            self.logger.error(f"Completion failed: {e}")
            raise

    async def complete_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming completion for the given prompt.
        
        Args:
            prompt: Input prompt
            context: Optional context information
            **kwargs: Additional parameters
            
        Yields:
            Text chunks
        """
        message = Message(role=RoleType.USER, content=prompt)
        
        # Add system message if provided
        messages = []
        if context and "system_prompt" in context:
            messages.append(
                Message(
                    role=RoleType.SYSTEM,
                    content=context["system_prompt"]
                )
            )
        messages.append(message)

        try:
            async for chunk in self.client.create_chat_completion_stream(
                messages=messages,
                **kwargs,
            ):
                yield chunk

        except Exception as e:
            self.logger.error(f"Streaming completion failed: {e}")
            raise


# =============================================================================
# Factory Functions
# =============================================================================

def create_ryzanstein_provider(
    api_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    model: str = "ryzanstein-7b",
) -> RyzansteinProvider:
    """
    Create a Ryzanstein provider for Continue.dev.
    
    Args:
        api_url: API URL
        api_key: API key
        model: Model name
        
    Returns:
        RyzansteinProvider instance
    """
    return RyzansteinProvider(
        api_url=api_url,
        api_key=api_key,
        model=model,
    )


async def test_api_connection(
    api_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
) -> bool:
    """
    Test connection to Ryzanstein API.
    
    Args:
        api_url: API URL
        api_key: API key
        
    Returns:
        True if connection successful
    """
    async with RyzansteinAPIClient(
        api_url=api_url,
        api_key=api_key,
    ) as client:
        return await client.health_check()


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        # Test connection
        connected = await test_api_connection()
        print(f"API Connection: {'✓' if connected else '✗'}")

        # Create provider
        provider = create_ryzanstein_provider()

        # Test completion
        try:
            response = await provider.complete(
                "Explain what a closure is in programming"
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(main())
