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
    # Display/report quality gate: detections drawn and reported (HUD, ball
    # marker) need at least this confidence.
    confidence_threshold: float = 0.35
    # Inference resolution. 0 = auto (source frame width, multiple of 32,
    # min 640); > 0 forces that size. Ultralytics' implicit default of 640
    # downscales 1280-wide broadcast footage and measurably loses distant
    # players (person detections at conf >= 0.35 roughly double at native
    # width on the sample match).
    imgsz: int = 0
    # Tracker candidate floor: detections at/above this (but below
    # confidence_threshold) are still produced so ByteTrack's low-score
    # rescue stage can re-associate players through short occlusions.
    candidate_threshold: float = 0.10
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
    # Number of clusters: 2 teams (TEAM_A / TEAM_B), everything else UNKNOWN.
    n_teams: int = 2
    # Upper-body (torso) crop inside each player bounding box. Expressed as a
    # fraction of the box, so it adapts to players of any size on screen.
    torso_crop_top: float = 0.2
    torso_crop_bottom: float = 0.6
    # Columns trimmed on each side of the box (background / arms bleeding in).
    torso_crop_side: float = 0.15

    # --- dominant jersey colour extraction -------------------------------- #
    # Pixel clusters used to find the dominant colour region of the crop.
    n_dominant_colors: int = 3
    # Crops smaller than this many pixels are ignored (far-away players).
    min_crop_pixels: int = 40
    # Very large crops are subsampled to this many pixels (constant cost).
    max_crop_pixels: int = 512
    # The dominant region must cover at least this share of the crop, else the
    # observation is rejected (occlusion, advertisements / background).
    min_dominant_ratio: float = 0.45
    # Weight of brightness (V) in the 3-D colour feature. Lower = more robust
    # to shadows, higher = better white-vs-black separation.
    value_weight: float = 0.5
    # Lloyd iterations for the per-crop dominant colour.
    pixel_kmeans_iters: int = 6

    # --- two-team clustering --------------------------------------------- #
    # Frames between colour-model refits (keeps TEAM_A/TEAM_B meanings stable).
    recluster_interval: int = 15
    # Minimum pooled observations / distinct tracks before the first fit.
    min_cluster_samples: int = 40
    min_cluster_tracks: int = 4
    # Caps keeping one clip cheap: pool at most this many samples in total,
    # taking at most this many recent features per player (a player's vote is
    # the median of that same window).
    max_cluster_samples: int = 4000
    max_samples_per_track: int = 12
    # A player only votes in a fit once they have at least this many
    # observations: a brand-new track's first boxes sit on grass/background, so
    # an immature median would poison the very first models.
    min_vote_features: int = 6
    # k-means effort for the team clustering itself.
    kmeans_iters: int = 25
    kmeans_restarts: int = 3
    # Clusters closer than this multiple of their own spread are considered
    # one blob (similar kits) -> the model is refused and everyone stays
    # UNKNOWN instead of being split arbitrarily. 3.0 is the conservative
    # default; config.yaml lowers it for distant-camera footage where the
    # white-vs-navy gap is only ~2.2-2.8 robust spreads.
    min_separation_ratio: float = 3.0
    # Minimum absolute distance between the two centres in feature space. The
    # ratio gate above is scale-free, so a sub-split of ONE kit (both halves
    # tight => tiny radii) clears it easily; this floor refuses such
    # near-identical pairs while genuine kit pairs sit far above it.
    min_separation_distance: float = 0.10
    # Per-cluster radius is floored at this fraction of the separation so an
    # extremely tight cluster cannot reject everything as "too far".
    radius_floor_ratio: float = 0.05
    # Smallest share of votes any team cluster may hold. A split whose
    # minority is smaller than this found outliers rather than teams (lone
    # referee, grass-dominated players, background-flickered crops), so that
    # cluster is pruned and the fit retried; if no balanced split appears the
    # fit is refused instead of labelling everyone wrongly. Real footage shows
    # 2-4 outlier votes among 10-16 players (~15-25%), so 0.15 was too low to
    # prune them: 0.25 keeps a genuine small side (>= a quarter of votes) while
    # pruning the junk.
    min_minority_share: float = 0.25

    # --- assignment + temporal smoothing ---------------------------------- #
    # Rolling observation window per player (frames kept for the mean colour).
    feature_history: int = 32
    # Recent observations averaged when deciding the current team.
    mean_window: int = 10
    # Uncertainty gates (both relative to the clustered data, never fixed
    # colours): too far from the nearest team -> UNKNOWN (referee, keeper,
    # odd kit); almost equally close to both -> UNKNOWN (similar kits).
    unknown_radius_scale: float = 2.5
    unknown_margin_ratio: float = 0.75
    # Consecutive agreeing frames before a team is committed to a player...
    min_consistent_frames: int = 5
    # ...and consecutive contradicting frames before it may be switched.
    switch_frames: int = 8
    # Seed for the clustering (deterministic output for a given clip).
    random_seed: int = 0


@dataclass
class PitchConfig:
    """Settings for 2D tactical pitch dimensions and coordinate mapping (Stage 4).

    The coordinate system is meters with the origin at one corner of the
    pitch: x runs along the length (0 -> ``length_meters``), y along the width
    (0 -> ``width_meters``). See app/pitch.py for the full convention.
    """
    length_meters: float = 105.0
    width_meters: float = 68.0
    # Manual calibration file (image <-> pitch correspondences + homography).
    # Relative paths resolve against the project root. The pitch stage only
    # runs when this file exists - create it with: python main.py --calibrate
    calibration_path: Path = Path("calibration/pitch.json")
    # Minimum number of image/pitch point pairs accepted by findHomography().
    min_points: int = 4
    # Draw the top-down radar inset on the annotated video (needs calibration).
    show_radar: bool = True


@dataclass
class LoggingConfig:
    """Settings for application logging."""
    level: str = "INFO"
    format: str = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class TacticsConfig:
    """Stage 6/7 tactical analysis settings (formations + teammate graph)."""
    # Sliding window (frames) used to smooth formation labels over time.
    formation_window: int = 10
    # Below this smoothed confidence the formation is reported as UNKNOWN.
    formation_min_confidence: float = 0.6
    # Stage 7: teammates farther apart than this (metres) are never linked.
    max_connection_distance: float = 25.0
    # Stage 7: each player links only to its k nearest teammates (within the
    # distance cap), which keeps the graph sparse instead of fully connected.
    connection_k_neighbors: int = 3
    # Stage 7: an established edge survives up to this extra distance ratio
    # before dropping - temporal hysteresis so edges do not flicker.
    edge_hysteresis_ratio: float = 0.15


@dataclass
class AppConfig:
    """Master application configuration aggregate."""
    project_root: Path = PROJECT_ROOT
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    team_classifier: TeamClassifierConfig = field(default_factory=TeamClassifierConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    tactics: TacticsConfig = field(default_factory=TacticsConfig)
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
        imgsz=int(m_raw.get("imgsz", 0)),
        candidate_threshold=float(m_raw.get("candidate_threshold", 0.10)),
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
        torso_crop_side=float(tc_raw.get("torso_crop_side", 0.15)),
        n_dominant_colors=int(tc_raw.get("n_dominant_colors", 3)),
        min_crop_pixels=int(tc_raw.get("min_crop_pixels", 40)),
        max_crop_pixels=int(tc_raw.get("max_crop_pixels", 512)),
        min_dominant_ratio=float(tc_raw.get("min_dominant_ratio", 0.45)),
        value_weight=float(tc_raw.get("value_weight", 0.5)),
        pixel_kmeans_iters=int(tc_raw.get("pixel_kmeans_iters", 6)),
        recluster_interval=int(tc_raw.get("recluster_interval", 15)),
        min_cluster_samples=int(tc_raw.get("min_cluster_samples", 40)),
        min_cluster_tracks=int(tc_raw.get("min_cluster_tracks", 4)),
        max_cluster_samples=int(tc_raw.get("max_cluster_samples", 4000)),
        max_samples_per_track=int(tc_raw.get("max_samples_per_track", 12)),
        kmeans_iters=int(tc_raw.get("kmeans_iters", 25)),
        kmeans_restarts=int(tc_raw.get("kmeans_restarts", 3)),
        min_separation_ratio=float(tc_raw.get("min_separation_ratio", 3.0)),
        min_separation_distance=float(tc_raw.get("min_separation_distance", 0.10)),
        radius_floor_ratio=float(tc_raw.get("radius_floor_ratio", 0.05)),
        min_minority_share=float(tc_raw.get("min_minority_share", 0.25)),
        min_vote_features=int(tc_raw.get("min_vote_features", 6)),
        feature_history=int(tc_raw.get("feature_history", 32)),
        mean_window=int(tc_raw.get("mean_window", 10)),
        unknown_radius_scale=float(tc_raw.get("unknown_radius_scale", 2.5)),
        unknown_margin_ratio=float(tc_raw.get("unknown_margin_ratio", 0.75)),
        min_consistent_frames=int(tc_raw.get("min_consistent_frames", 5)),
        switch_frames=int(tc_raw.get("switch_frames", 8)),
        random_seed=int(tc_raw.get("random_seed", 0)),
    )

    # Parse PitchConfig
    p_raw = raw_cfg.get("pitch", {})
    pitch_cfg = PitchConfig(
        length_meters=float(p_raw.get("length_meters", 105.0)),
        width_meters=float(p_raw.get("width_meters", 68.0)),
        calibration_path=_resolve_path(
            str(p_raw.get("calibration_path", "calibration/pitch.json")), PROJECT_ROOT
        ),
        min_points=int(p_raw.get("min_points", 4)),
        show_radar=bool(p_raw.get("show_radar", True)),
    )

    # Parse TacticsConfig (Stage 6 formation smoothing, Stage 7 graph limits)
    ta_raw = raw_cfg.get("tactics", {})
    tactics_cfg = TacticsConfig(
        formation_window=int(ta_raw.get("formation_window", 10)),
        formation_min_confidence=float(ta_raw.get("formation_min_confidence", 0.6)),
        max_connection_distance=float(ta_raw.get("max_connection_distance", 25.0)),
        connection_k_neighbors=int(ta_raw.get("connection_k_neighbors", 3)),
        edge_hysteresis_ratio=float(ta_raw.get("edge_hysteresis_ratio", 0.15)),
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
        tactics=tactics_cfg,
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
