"""
Gemini API client implementation.
"""

try:
    import google.generativeai as genai
except ImportError:
    # Fallback to new package if old one not available
    import google.genai as genai

from .base import LLMClient
from mindmap_chat.config import config


class GeminiClient(LLMClient):
    """Gemini API client."""

    def __init__(self):
        """Initialize Gemini client."""
        genai.configure(api_key=config.gemini.api_key)
        self.model = genai.GenerativeModel(config.gemini.model_name)
        
        # Embedding model for vector representations
        self.embedding_model = "gemini-embedding-001"

    def call(self, prompt: str, json_mode: bool = False) -> str:
        """
        Call Gemini API.
        
        Args:
            prompt: The prompt to send
            json_mode: If True, add instruction to return JSON
            
        Returns:
            The model's response
        """
        full_prompt = prompt
        if json_mode:
            full_prompt += "\n\nRESPOND ONLY WITH VALID JSON (no markdown, no extra text)."
        
        response = self.model.generate_content(
            full_prompt,
            generation_config={
                "temperature": config.gemini.temperature,
                "max_output_tokens": config.gemini.max_output_tokens,
            }
        )
        
        return response.text

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding using Gemini's embedding model.
        
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
