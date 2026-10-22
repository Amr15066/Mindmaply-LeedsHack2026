"""
DeepSeek API client implementation.
Uses DeepSeek for generation (answering) but Gemini for embeddings (cheap).
"""

import requests

try:
    import google.genai as genai
except ImportError:
    import google.generativeai as genai

from .base import LLMClient
from mindmap_chat.config import config


class DeepSeekClient(LLMClient):
    """DeepSeek API client for generation, Gemini for embeddings."""

    def __init__(self):
        """Initialize DeepSeek client."""
        self.api_key = config.deepseek.api_key
        self.model_name = config.deepseek.model_name
        self.base_url = "https://api.deepseek.com/chat/completions"
        
        # Keep Gemini for cheap embeddings
        genai.configure(api_key=config.gemini.api_key)
        self.embedding_model = "text-embedding-004"

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
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"DeepSeek API error: {e}") from e

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding using Gemini (cheap, token-efficient).
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        result = genai.embed_content(
            model=self.embedding_model,
            content=text
        )
        return result["embedding"]
