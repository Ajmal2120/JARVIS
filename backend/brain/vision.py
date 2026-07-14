"""
JARVIS Vision - Describe images using LLM
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

logger = logging.getLogger("jarvis.vision")


async def describe_image(image_base64: str, prompt: str = "Describe what's in this image in detail. Focus on text, UI elements, apps, and important visual content.") -> str:
    """Use the LLM to describe what's in an image."""
    from llm import get_provider

    try:
        provider = get_provider()

        message = await provider.chat(
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }],
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=None,
            tools=[],
        )

        description = message.text
        return description.strip() if description else "I couldn't analyze this image."

    except Exception as e:
        logger.error(f"Vision analysis failed: {e}")
        return f"I couldn't analyze this image: {e}"
