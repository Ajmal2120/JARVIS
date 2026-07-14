"""
Gemini Provider — Google's Gemini implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import content_types
except ImportError:
    genai = None
    content_types = None

from llm.provider import LLMProvider, ChatResponse

logger = logging.getLogger("jarvis.llm.gemini")


class GeminiProvider(LLMProvider):
    """Google Gemini implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-exp",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        if not genai:
            raise ImportError(
                "google-generativeai is required for Gemini support. "
                "Install with: pip install google-generativeai"
            )

        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(self._model)

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _convert_tools(self, tools: Optional[list[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert Anthropic-style tool definitions to Gemini format."""
        if not tools:
            return None

        converted = []
        for tool in tools:
            # Gemini expects a simpler function declaration format
            converted.append({
                "function_declarations": [{
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool.get("input_schema", {})
                }]
            })
        return converted

    def _convert_messages(self, messages: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Convert to Gemini message format."""
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Gemini handles system messages differently - we'll prepend to first user message
                continue

            if isinstance(content, list):
                # Handle content blocks (e.g. tool results)
                for part in content:
                    if part.get("type") == "tool_result":
                        # Convert tool_result to a simple text message
                        converted.append({
                            "role": "user",
                            "parts": [{"text": f"Tool result for {part.get('tool_use_id', 'unknown')}:\n{part.get('content', '')}"}]
                        })
                    else:
                        # Handle image data
                        if part.get("type") == "image" and "data" in part:
                            converted.append({
                                "role": role,
                                "parts": [
                                    {"inline_data": {"mime_type": part.get("media_type", "image/png"), "data": part["data"]}},
                                    {"text": content.get("text", "") if isinstance(content, dict) else ""}
                                ]
                            })
            else:
                converted.append({
                    "role": role,
                    "parts": [{"text": content}]
                })
        return converted

    def _extract_tool_calls(self, response) -> List[Dict[str, Any]]:
        """Extract tool calls from Gemini response."""
        tool_calls = []

        # Check if response contains function calls
        if hasattr(response, "candidates"):
            for candidate in response.candidates:
                if hasattr(candidate, "function_calls"):
                    for call in candidate.function_calls:
                        tool_calls.append({
                            "id": call.name or str(hash(call)),
                            "name": call.name,
                            "input": call.args or {},
                        })

        return tool_calls

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
        # Handle system message
        user_messages = self._convert_messages(messages)
        if system:
            # Prepend system message to first user message
            if user_messages:
                user_messages[0]["parts"] = [{"text": system + "\n\n" + user_messages[0]["parts"][0]["text"]}]
            else:
                user_messages.append({"role": "user", "parts": [{"text": system}]})

        # Convert tools
        converted_tools = self._convert_tools(tools)

        # Prepare chat with tools
        if converted_tools:
            chat = self._client.start_chat(
                history=user_messages[:-1],  # All but the last message
                tools=converted_tools
            )
            response = await chat.send_message_async(user_messages[-1]["parts"][0]["text"])
        else:
            chat = self._model.start_chat(history=user_messages)
            response = await chat.send_message_async(user_messages[-1]["parts"][0]["text"])

        # Extract tool calls
        tool_calls = self._extract_tool_calls(response)

        # Get content
        content = response.text if hasattr(response, "text") else str(response)

        return ChatResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )

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
        # Note: Gemini streaming is not fully implemented yet
        # For now, we'll just yield the full response
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
        yield {"type": "done", "usage": None}