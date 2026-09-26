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


def test_team_classifier_config_defaults():
    """Stage 3 exposes every clustering / smoothing knob with sane defaults."""
    from app.config import TeamClassifierConfig

    tc = TeamClassifierConfig()
    assert tc.n_teams == 2
    assert 0.0 < tc.torso_crop_top < tc.torso_crop_bottom < 1.0
    assert 0.0 < tc.torso_crop_side < 0.5
    assert tc.min_dominant_ratio > 0.4  # dominant region must clearly dominate
    assert tc.min_consistent_frames >= 1
    assert tc.switch_frames >= tc.min_consistent_frames


def test_team_classifier_config_from_yaml(tmp_path: Path):
    """YAML overrides win; untouched keys keep their defaults."""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        "team_classifier:\n"
        "  torso_crop_side: 0.2\n"
        "  min_dominant_ratio: 0.6\n"
        "  unknown_radius_scale: 3.5\n"
        "  switch_frames: 12\n"
        "  random_seed: 42\n",
        encoding="utf-8",
    )
    loaded = load_config(cfg_file)
    tc = loaded.team_classifier
    assert tc.torso_crop_side == 0.2
    assert tc.min_dominant_ratio == 0.6
    assert tc.unknown_radius_scale == 3.5
    assert tc.switch_frames == 12
    assert tc.random_seed == 42
    # Untouched keys keep their defaults.
    assert tc.n_teams == 2
    assert tc.min_cluster_tracks == 4
    assert tc.feature_history == 32


def test_pitch_config_defaults():
    """Stage 4 defaults: standard pitch, manual calibration file, radar on."""
    from app.config import PitchConfig

    pitch = PitchConfig()
    assert pitch.length_meters == 105.0
    assert pitch.width_meters == 68.0
    assert pitch.min_points >= 4
    assert pitch.show_radar is True
    assert Path(pitch.calibration_path).name == "pitch.json"


def test_pitch_config_from_default_yaml():
    """config.yaml pins the pitch size and calibration path."""
    config = load_config()
    assert config.pitch.length_meters == 105.0
    assert config.pitch.width_meters == 68.0
    assert config.pitch.calibration_path.is_absolute()
    assert config.pitch.calibration_path.name == "pitch.json"


def test_pitch_config_from_yaml(tmp_path: Path):
    """YAML overrides win; the calibration path resolves against the root."""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        "pitch:\n"
        "  length_meters: 90.0\n"
        "  width_meters: 45.0\n"
        "  calibration_path: 'my_calib.json'\n"
        "  min_points: 6\n"
        "  show_radar: false\n",
        encoding="utf-8",
    )
    pitch = load_config(cfg_file).pitch
    assert pitch.length_meters == 90.0
    assert pitch.width_meters == 45.0
    assert pitch.min_points == 6
    assert pitch.show_radar is False
    assert pitch.calibration_path.is_absolute()
    assert pitch.calibration_path.name == "my_calib.json"
