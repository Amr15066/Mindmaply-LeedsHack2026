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

DEFAULT_STORAGE_PATH = str(Path(__file__).parent / "data" / "conversation.json")


def _env(name: str, default: str = ""):
    """Read an environment variable when the config object is created, not at import."""
    return field(default_factory=lambda: os.getenv(name, default))


@dataclass
class GeminiConfig:
    """Gemini API configuration."""
    api_key: str = _env("GEMINI_API_KEY")
    # Preview aliases can be retired without notice; override with GEMINI_MODEL.
    model_name: str = _env("GEMINI_MODEL", "gemini-3-flash-preview")
    temperature: float = 0.7
    max_output_tokens: int = 1024


@dataclass
class DeepSeekConfig:
    """DeepSeek API configuration (optional alternative generator)."""
    api_key: str = _env("DEEPSEEK_API_KEY")
    model_name: str = _env("DEEPSEEK_MODEL", "deepseek-chat")
    temperature: float = 0.7
    max_output_tokens: int = 1024


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model: str = "gemini-embedding-001"  # Hosted Gemini API model
    embedding_dim: int = 3072  # Model default; also supports 1536 / 768 (Matryoshka)


@dataclass
class DetectionThresholds:
    """Thresholds for intent detection."""
    # similarity >= continue_threshold          -> CONTINUE (no LLM call)
    # similarity >= deepen_threshold            -> DEEPEN   (no LLM call)
    # otherwise                                 -> ask the LLM classifier
    continue_threshold: float = 0.90  # Same topic
    deepen_threshold: float = 0.70  # Deeper dive
    # On TANGENT, the best block in any graph must reach this to be reused
    sibling_threshold: float = 0.75


@dataclass
class AppConfig:
    """Application-wide configuration."""
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    thresholds: DetectionThresholds = field(default_factory=DetectionThresholds)
    auto_summarize_after_n_messages: int = 6
    # Shared by the CLI and the web app; override with MINDMAP_STORAGE_PATH.
    storage_path: str = _env("MINDMAP_STORAGE_PATH", DEFAULT_STORAGE_PATH)
    context_window_size: int = 3  # Last N messages to include in context
    ancestor_context_depth: int = 3  # Parent blocks included in answer context
    ancestor_context_chars: int = 1500  # Character budget for those parents


# Global config instance
config = AppConfig()


def validate_config():
    """Validate that all required config is set."""
    if not config.gemini.api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    print("[OK] Configuration validated")
