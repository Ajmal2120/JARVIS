"""
LLM Provider Interface — Abstract base class for all providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass


@dataclass
class ChatResponse:
    """Unified response from all LLM providers."""
    content: str
    role: str = "assistant"
    tool_calls: Optional[list[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[list[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """Send a chat completion request."""
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[list[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat completion response."""
        # Default implementation can be overridden
        response = await self.chat(
            messages=messages,
            model=model,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            **kwargs
        )
        yield {"type": "content", "content": response.content}
        yield {"type": "done", "usage": response.usage}