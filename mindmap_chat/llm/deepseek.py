"""
DeepSeek API client implementation.
Uses DeepSeek for generation (answering) but Gemini for embeddings (cheap).
"""

import logging
import requests

from mindmap_chat.config import config

from .base import TRUNCATION_NOTICE, LLMClient, LLMResponseError
from .gemini import GeminiClient

logger = logging.getLogger(__name__)


class DeepSeekClient(LLMClient):
    """DeepSeek API client for generation, Gemini for embeddings."""

    def __init__(self):
        """Initialize DeepSeek client."""
        self.api_key = config.deepseek.api_key
        self.model_name = config.deepseek.model_name
        self.base_url = "https://api.deepseek.com/chat/completions"

        # Keep Gemini for cheap embeddings. Using the same embedding model as
        # GeminiClient keeps stored block embeddings comparable across clients.
        self._embedder = GeminiClient()

    def call(self, prompt: str, json_mode: bool = False) -> str:
        """
        Call DeepSeek API for text generation.

        Args:
            prompt: The prompt to send
            json_mode: If True, request JSON response

        Returns:
            The model's response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": config.deepseek.temperature,
            "max_tokens": config.deepseek.max_output_tokens,
            "response_format": {"type": "json_object"} if json_mode else None,
        }

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise Exception(f"DeepSeek API error: {e}") from e

        choices = data.get("choices") or []
        text = choices[0].get("message", {}).get("content") if choices else None
        if not text:
            raise LLMResponseError(f"DeepSeek returned no content: {data}")
        if choices[0].get("finish_reason") == "length":
            logger.warning(
                "DeepSeek response truncated at %d tokens", config.deepseek.max_output_tokens
            )
            if not json_mode:
                text += TRUNCATION_NOTICE
        return text

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding using Gemini (cheap, token-efficient).

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self._embedder.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts in a single Gemini request."""
        return self._embedder.embed_batch(texts)
