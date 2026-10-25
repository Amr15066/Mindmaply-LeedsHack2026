"""
Configuration and environment settings.
Centralized place for API keys, model names, thresholds.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from mindmap_chat folder
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


@dataclass
class GeminiConfig:
    """Gemini API configuration."""
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    model_name: str = "gemini-3-flash-preview"  # Fast and generous free tier
    temperature: float = 0.7
    max_output_tokens: int = 1024


@dataclass
class DeepSeekConfig:
    """DeepSeek API configuration (optional alternative generator)."""
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    max_output_tokens: int = 1024


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model: str = "gemini-embedding-001"  # Free, local, lightweight
    embedding_dim: int = 384


@dataclass
class DetectionThresholds:
    """Thresholds for intent detection."""
    continue_threshold: float = 0.90  # Same topic
    deepen_threshold: float = 0.70  # Deeper dive
    sibling_threshold: float = 0.75  # Related subtopic
    related_match_threshold: float = 0.60  # Looser match for related blocks
    tangent_threshold: float = 0.65  # Unrelated


@dataclass
class AppConfig:
    """Application-wide configuration."""
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    thresholds: DetectionThresholds = field(default_factory=DetectionThresholds)
    auto_summarize_after_n_messages: int = 6
    storage_path: str = "./data/conversation.json"
    context_window_size: int = 3  # Last N messages to include in context


# Global config instance
config = AppConfig()


def validate_config():
    """Validate that all required config is set."""
    if not config.gemini.api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    print("[OK] Configuration validated")
