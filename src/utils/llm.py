"""
LLM client wrapper for agent reasoning and persona generation.
"""

import json
import logging
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat completion API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.max_tokens = max_tokens or LLM_MAX_TOKENS
        self.client = OpenAI(
            api_key=api_key or LLM_API_KEY,
            base_url=base_url or LLM_BASE_URL,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant's text."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> dict:
        """Chat and parse the response as JSON."""
        raw = self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response, attempting extraction")
            # Try to extract JSON from markdown code blocks
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            return json.loads(raw)
