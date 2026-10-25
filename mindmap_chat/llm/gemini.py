"""
Gemini API client implementation (google-genai SDK).
"""

import logging
from google import genai
from google.genai import types

from mindmap_chat.config import config

from .base import TRUNCATION_NOTICE, LLMClient, LLMResponseError

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """Gemini API client."""

    def __init__(self):
        """Initialize Gemini client."""
        self.client = genai.Client(api_key=config.gemini.api_key)
        self.model_name = config.gemini.model_name

        # Embedding model for vector representations
        self.embedding_model = config.embeddings.model

    def call(self, prompt: str, json_mode: bool = False) -> str:
        """
        Call Gemini API.

        Args:
            prompt: The prompt to send
            json_mode: If True, request a JSON response

        Returns:
            The model's response

        Raises:
            LLMResponseError: if the response has no text (blocked, empty, etc.)
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=config.gemini.temperature,
                max_output_tokens=config.gemini.max_output_tokens,
                response_mime_type="application/json" if json_mode else None,
            ),
        )

        candidates = response.candidates or []
        finish_reason = candidates[0].finish_reason if candidates else None
        text = response.text
        if not text:
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else None
            raise LLMResponseError(
                f"Gemini returned no text (finish_reason={finish_reason}, "
                f"block_reason={block_reason})"
            )

        if finish_reason == types.FinishReason.MAX_TOKENS:
            logger.warning(
                "Gemini response truncated at %d tokens", config.gemini.max_output_tokens
            )
            if not json_mode:
                text += TRUNCATION_NOTICE
        return text

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding using Gemini's embedding model.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts in a single request."""
        if not texts:
            return []
        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=config.embeddings.embedding_dim
            ),
        )
        embeddings = result.embeddings or []
        if len(embeddings) != len(texts) or not all(e.values for e in embeddings):
            raise LLMResponseError(
                f"Gemini returned {len(embeddings)} embeddings for {len(texts)} texts"
            )
        return [list(e.values or []) for e in embeddings]
