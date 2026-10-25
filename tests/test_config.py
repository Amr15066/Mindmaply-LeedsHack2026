from dataclasses import fields
from pathlib import Path

import mindmap_chat
from mindmap_chat.config import AppConfig, DetectionThresholds


def test_environment_is_read_when_config_is_created(monkeypatch):
    """Regression (§2.12): os.getenv ran at import time, so later env changes were ignored."""
    monkeypatch.setenv("GEMINI_API_KEY", "key-from-env")
    monkeypatch.setenv("GEMINI_MODEL", "some-stable-model")
    monkeypatch.setenv("MINDMAP_STORAGE_PATH", "/tmp/elsewhere.json")
    cfg = AppConfig()
    assert cfg.gemini.api_key == "key-from-env"
    assert cfg.gemini.model_name == "some-stable-model"
    assert cfg.storage_path == "/tmp/elsewhere.json"


def test_defaults_without_environment(monkeypatch):
    for name in ("GEMINI_MODEL", "MINDMAP_STORAGE_PATH", "DEEPSEEK_MODEL"):
        monkeypatch.delenv(name, raising=False)
    cfg = AppConfig()
    assert cfg.gemini.model_name == "gemini-3-flash-preview"
    assert cfg.deepseek.model_name == "deepseek-chat"
    # One shared file for CLI and web, independent of the working directory
    package_dir = Path(mindmap_chat.__file__).parent
    assert Path(cfg.storage_path) == package_dir / "data" / "conversation.json"


def test_only_thresholds_that_are_used_exist():
    """Regression (§2.5): tangent/related_match thresholds were declared but had no effect."""
    assert {f.name for f in fields(DetectionThresholds)} == {
        "continue_threshold",
        "deepen_threshold",
        "sibling_threshold",
    }


def test_embedding_dim_matches_model_default():
    """Regression (§2.6): 384 was left over from a local-model design."""
    cfg = AppConfig()
    assert cfg.embeddings.model == "gemini-embedding-001"
    assert cfg.embeddings.embedding_dim == 3072
