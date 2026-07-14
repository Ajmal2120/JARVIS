"""
Claude Provider — thin wrapper over the Anthropic SDK.

All Anthropic-specific code lives here and only here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import anthropic

from llm.provider import LLMProvider, ChatResponse

logger = logging.getLogger("jarvis.llm.claude")


class ClaudeProvider(LLMProvider):
    """Anthropic Claude implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        base_url: Optional[str] = None,
        max_retries: int = 5,
        timeout: float = 60.0,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._max_retries = max_retries
        self._timeout = timeout
        self._client: Optional[anthropic.AsyncAnthropic] = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Lazy-create the async Anthropic client."""
        if self._client is None:
            kwargs = dict(
                api_key=self._api_key,
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    @property
    def provider_name(self) -> str:
        return "claude"

    def _convert_messages(self, messages: list[Dict[str, Any]]) -> tuple[str, list[Dict[str, Any]]]:
        """Extract system prompt and convert to Anthropic message format."""
        system = None
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                system = msg.get("content", "")
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                # Already in Anthropic content blocks format (e.g. tool_result)
                converted.append({"role": role, "content": content})
            else:
                converted.append({"role": role, "content": content})
        return system, converted

    def _parse_response(self, response) -> ChatResponse:
        """Convert an Anthropic response to ChatResponse."""
        content_parts = []
        tool_calls = []
        for block in response.content:
            if hasattr(block, "text"):
                content_parts.append(block.text)
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return ChatResponse(
            content="".join(content_parts),
            tool_calls=tool_calls if tool_calls else None,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            } if response.usage else None,
        )

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
        client = self._get_client()
        system_str, anthropic_messages = self._convert_messages(messages)

        effective_system = system or system_str
        effective_model = model or self._model

        params: Dict[str, Any] = dict(
            model=effective_model,
            max_tokens=max_tokens or 1024,
            messages=anthropic_messages,
        )
        if effective_system:
            params["system"] = effective_system
        if tools:
            params["tools"] = tools

        response = await client.messages.create(**params)
        return self._parse_response(response)

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
        client = self._get_client()
        system_str, anthropic_messages = self._convert_messages(messages)

        effective_system = system or system_str
        effective_model = model or self._model

        params: Dict[str, Any] = dict(
            model=effective_model,
            max_tokens=max_tokens or 1024,
            messages=anthropic_messages,
        )
        if effective_system:
            params["system"] = effective_system
        if tools:
            params["tools"] = tools

        with client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield {"type": "content", "content": text}
            final = stream.get_final_message()
            yield {"type": "done", "usage": {"prompt_tokens": final.usage.input_tokens, "completion_tokens": final.usage.output_tokens} if final.usage else None}