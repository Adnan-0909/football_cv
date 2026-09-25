"""
Unit Tests for Configuration Management
=======================================
Verifies configuration loading, path resolution, and default fallbacks.
"""

from pathlib import Path
import pytest

from app.config import AppConfig, ModelConfig, VideoConfig, load_config, PROJECT_ROOT


def test_project_root_detection():
    """Verify project root is dynamically detected without hardcoding."""
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "main.py").exists()
    assert (PROJECT_ROOT / "app").exists()


def test_load_default_config():
    """Verify loading default config.yaml successfully."""
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.model.confidence_threshold > 0.0
    assert config.model.weights_path.is_absolute()
    assert config.video.input_path.is_absolute()
    assert config.video.output_path.is_absolute()


def test_directory_creation():
    """Verify standard directories are initialized."""
    config = load_config()
    config.ensure_directories()
    assert (config.project_root / "models").is_dir()
    assert (config.project_root / "input").is_dir()
    assert (config.project_root / "output").is_dir()


def test_custom_values():
    """Verify config structures accept custom overrides."""
    m_cfg = ModelConfig(confidence_threshold=0.6, device="cuda")
    assert m_cfg.confidence_threshold == 0.6
    assert m_cfg.device == "cuda"
