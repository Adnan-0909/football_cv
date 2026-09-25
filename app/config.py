"""
Configuration Management Module
===============================

Handles loading, validating, and providing project-wide configuration.
All paths are resolved dynamically relative to the project root,
ensuring portability across different machines and operating systems.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union
import yaml

# Resolve the project root dynamically (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    """Settings for YOLO object detection."""
    weights_path: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "yolov8s.pt")
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5
    device: str = "cpu"
    player_class_id: int = 0
    ball_class_id: int = 32


@dataclass
class VideoConfig:
    """Settings for video reading, writing, and processing."""
    input_path: Path = field(default_factory=lambda: PROJECT_ROOT / "input" / "match_sample.mp4")
    output_path: Path = field(default_factory=lambda: PROJECT_ROOT / "output" / "tactical_analysis.mp4")
    max_frames: Optional[int] = None
    frame_stride: int = 1


@dataclass
class TrackerConfig:
    """Settings for player tracking across consecutive frames."""
    # Backend: "bytetrack" (Ultralytics built-in, needs lap) or
    # "bytetrack_lite" (dependency-free NumPy fallback).
    tracker_type: str = "bytetrack"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8
    # When True only detections of the player class are handed to the tracker:
    # other objects (e.g. the ball) stay detectable but never get a track ID.
    player_only: bool = True


@dataclass
class TeamClassifierConfig:
    """Settings for team identification using jersey color clustering."""
    n_teams: int = 2
    torso_crop_top: float = 0.2
    torso_crop_bottom: float = 0.6


@dataclass
class PitchConfig:
    """Settings for 2D tactical pitch dimensions and coordinate mapping."""
    length_meters: float = 105.0
    width_meters: float = 68.0


@dataclass
class LoggingConfig:
    """Settings for application logging."""
    level: str = "INFO"
    format: str = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class AppConfig:
    """Master application configuration aggregate."""
    project_root: Path = PROJECT_ROOT
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    team_classifier: TeamClassifierConfig = field(default_factory=TeamClassifierConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def ensure_directories(self) -> None:
        """Create standard project directories if they do not exist."""
        (self.project_root / "models").mkdir(parents=True, exist_ok=True)
        (self.project_root / "input").mkdir(parents=True, exist_ok=True)
        (self.project_root / "output").mkdir(parents=True, exist_ok=True)


def _resolve_path(raw_path: Union[str, Path], base_dir: Path = PROJECT_ROOT) -> Path:
    """Convert relative path strings to absolute Path objects based on base_dir."""
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """
    Load application configuration from a YAML file.
    If no path is provided or the file is missing, default settings are used.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        AppConfig: Fully initialized application configuration object.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = _resolve_path(config_path)

    raw_cfg: Dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    raw_cfg = loaded
        except Exception as e:
            print(f"[WARNING] Failed to parse config file at {config_path}: {e}. Using defaults.", file=sys.stderr)

    # Parse ModelConfig
    m_raw = raw_cfg.get("model", {})
    weights_raw = m_raw.get("weights_path", "models/yolov8s.pt")
    model_cfg = ModelConfig(
        weights_path=_resolve_path(weights_raw),
        confidence_threshold=float(m_raw.get("confidence_threshold", 0.35)),
        iou_threshold=float(m_raw.get("iou_threshold", 0.5)),
        device=str(m_raw.get("device", "cpu")),
        player_class_id=int(m_raw.get("player_class_id", 0)),
        ball_class_id=int(m_raw.get("ball_class_id", 32)),
    )

    # Parse VideoConfig
    v_raw = raw_cfg.get("video", {})
    input_raw = v_raw.get("input_path", "input/match_sample.mp4")
    output_raw = v_raw.get("output_path", "output/tactical_analysis.mp4")
    max_frames = v_raw.get("max_frames")
    video_cfg = VideoConfig(
        input_path=_resolve_path(input_raw),
        output_path=_resolve_path(output_raw),
        max_frames=int(max_frames) if max_frames is not None and max_frames != -1 else None,
        frame_stride=int(v_raw.get("frame_stride", 1)),
    )

    # Parse TrackerConfig
    t_raw = raw_cfg.get("tracker", {})
    tracker_cfg = TrackerConfig(
        tracker_type=str(t_raw.get("tracker_type", "bytetrack")),
        track_high_thresh=float(t_raw.get("track_high_thresh", 0.5)),
        track_low_thresh=float(t_raw.get("track_low_thresh", 0.1)),
        new_track_thresh=float(t_raw.get("new_track_thresh", 0.6)),
        track_buffer=int(t_raw.get("track_buffer", 30)),
        match_thresh=float(t_raw.get("match_thresh", 0.8)),
        player_only=bool(t_raw.get("player_only", True)),
    )

    # Parse TeamClassifierConfig
    tc_raw = raw_cfg.get("team_classifier", {})
    team_cfg = TeamClassifierConfig(
        n_teams=int(tc_raw.get("n_teams", 2)),
        torso_crop_top=float(tc_raw.get("torso_crop_top", 0.2)),
        torso_crop_bottom=float(tc_raw.get("torso_crop_bottom", 0.6)),
    )

    # Parse PitchConfig
    p_raw = raw_cfg.get("pitch", {})
    pitch_cfg = PitchConfig(
        length_meters=float(p_raw.get("length_meters", 105.0)),
        width_meters=float(p_raw.get("width_meters", 68.0)),
    )

    # Parse LoggingConfig
    l_raw = raw_cfg.get("logging", {})
    log_cfg = LoggingConfig(
        level=str(l_raw.get("level", "INFO")),
        format=str(l_raw.get("format", "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")),
        date_format=str(l_raw.get("date_format", "%Y-%m-%d %H:%M:%S")),
    )

    app_config = AppConfig(
        project_root=PROJECT_ROOT,
        model=model_cfg,
        video=video_cfg,
        tracker=tracker_cfg,
        team_classifier=team_cfg,
        pitch=pitch_cfg,
        logging=log_cfg,
    )

    # Automatically ensure standard working directories exist
    app_config.ensure_directories()
    return app_config


def setup_logging(cfg: Optional[LoggingConfig] = None) -> logging.Logger:
    """
    Configure application-wide logging with timestamps and log levels.

    Args:
        cfg: Optional LoggingConfig instance. Defaults to standard INFO config.

    Returns:
        logging.Logger: The configured root or main logger.
    """
    if cfg is None:
        cfg = LoggingConfig()

    level = getattr(logging, cfg.level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=cfg.format,
        datefmt=cfg.date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger("football_tracker")
    logger.setLevel(level)
    return logger
