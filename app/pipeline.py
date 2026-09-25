"""
Pipeline Orchestration Module
==============================

Connects the individual stages into a runnable end-to-end flow:

    raw MP4 -> VideoReader -> FootballDetector -> PlayerTracker
            -> (optional stages) -> TacticalVisualizer -> VideoWriter

Stages that are still scaffolds (team classification, homography, formations,
passing lanes) are probed once at startup; if they raise ``NotImplementedError``
they are skipped with a single log line instead of crashing the run, so the
pipeline starts working automatically as they get implemented.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import cv2
import numpy as np

from app.config import AppConfig
from app.detector import FootballDetector, Detection
from app.tracker import PlayerTracker, TrackedObject
from app.track_log import TrackLog
from app.video import VideoReader, VideoWriter
from app.visualization import COLOR_BALL, COLOR_TEAM_A, COLOR_TEAM_B, TacticalVisualizer

# Fallback marker colour until team classification is implemented.
COLOR_UNKNOWN_PLAYER = (255, 255, 255)

# Progress is logged every N processed frames.
LOG_EVERY_N_FRAMES = 50


@dataclass
class PipelineStats:
    """Summary of a single pipeline run."""
    input_path: Path
    output_path: Path
    frames_processed: int = 0
    detections_total: int = 0
    tracks_total: int = 0
    unique_track_ids: Set[int] = field(default_factory=set)
    unique_player_ids: Set[int] = field(default_factory=set)
    players_total: int = 0
    players_last_frame: int = 0
    elapsed_seconds: float = 0.0
    # One record per tracked player per frame:
    # {frame, timestamp, player_id, bbox, center} - exportable with
    # TrackLog.save_csv() (see also the --track-csv flag in main.py).
    track_log: TrackLog = field(default_factory=TrackLog)

    @property
    def processing_fps(self) -> float:
        """Average processing speed in frames per second."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_processed / self.elapsed_seconds

    @property
    def mean_players_per_frame(self) -> float:
        """Average number of tracked players per processed frame."""
        if self.frames_processed == 0:
            return 0.0
        return self.players_total / self.frames_processed

    def summary(self) -> str:
        return (
            f"{self.frames_processed} frames | {self.detections_total} detections | "
            f"{len(self.unique_track_ids)} unique tracks ({len(self.unique_player_ids)} players) | "
            f"{self.processing_fps:.1f} fps | {self.elapsed_seconds:.1f}s | "
            f"-> {self.output_path}"
        )


class TacticalPipeline:
    """
    Runs detection, tracking, and annotation over a whole video.

    Components are injectable so tests (and future experiments) can substitute a
    stub detector or a custom tracker without touching production code.
    """

    def __init__(
        self,
        config: AppConfig,
        detector: Optional[FootballDetector] = None,
        tracker: Optional[PlayerTracker] = None,
        visualizer: Optional[TacticalVisualizer] = None,
    ) -> None:
        """
        Args:
            config: Full application configuration.
            detector: Optional detector override (defaults to ``FootballDetector``).
            tracker: Optional tracker override (defaults to ``PlayerTracker``).
            visualizer: Optional visualizer override.
        """
        self.config = config
        self.detector = detector if detector is not None else FootballDetector(config.model)
        self.tracker = tracker if tracker is not None else PlayerTracker(config.tracker)
        self.visualizer = visualizer if visualizer is not None else TacticalVisualizer()
        self.logger = logging.getLogger("football_tracker.pipeline")

        # Filled by _update_teams(): team_id per track_id, once the team
        # classifier exists. Empty for now.
        self.team_by_track: Dict[int, int] = {}
        self._team_stage_enabled = False
        self._team_classifier = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        input_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> PipelineStats:
        """
        Process a whole video and write the annotated result.

        Args:
            input_path: Raw input video. Defaults to ``config.video.input_path``.
            output_path: Annotated output video. Defaults to ``config.video.output_path``.

        Returns:
            PipelineStats: Counters describing the run.

        Raises:
            FileNotFoundError: If the input video does not exist.
            IOError: If the video cannot be opened / written.
        """
        input_path = Path(input_path) if input_path is not None else Path(self.config.video.input_path)
        output_path = Path(output_path) if output_path is not None else Path(self.config.video.output_path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input video not found: {input_path}. "
                "Put your raw footage there, or pass --input / set video.input_path in config.yaml."
            )

        stats = PipelineStats(input_path=input_path, output_path=output_path)
        self._probe_optional_stages()

        self.logger.info("Loading detection model (first run may download weights)...")
        started = time.perf_counter()
        self.detector.load_model()

        video_cfg = self.config.video
        stride = max(1, video_cfg.frame_stride)

        with VideoReader(
            input_path,
            frame_stride=stride,
            max_frames=video_cfg.max_frames,
        ) as reader:
            # Timestamps always refer to the source footage, so they stay
            # correct when frames are skipped with frame_stride.
            source_fps = reader.fps if reader.fps and reader.fps > 0 else 25.0
            with VideoWriter(output_path, fps=source_fps / stride, width=reader.width, height=reader.height) as writer:
                for frame_index, frame in reader.read_frames():
                    detections = self.detector.detect(frame)
                    tracks = self.tracker.update(self._tracker_inputs(detections), frame)

                    timestamp = frame_index / source_fps
                    for track in tracks:
                        if track.class_name == "player":
                            stats.track_log.add_track(frame_index, timestamp, track)

                    writer.write(self._annotate(frame, detections, tracks, frame_index))

                    stats.frames_processed += 1
                    stats.detections_total += len(detections)
                    stats.tracks_total += len(tracks)
                    player_ids = {t.track_id for t in tracks if t.class_name == "player"}
                    stats.unique_track_ids.update(t.track_id for t in tracks)
                    stats.unique_player_ids.update(player_ids)
                    stats.players_total += len(player_ids)
                    stats.players_last_frame = len(player_ids)

                    if stats.frames_processed % LOG_EVERY_N_FRAMES == 0:
                        self.logger.info(
                            "Processed %d frames (%d detections, %d active tracks)...",
                            stats.frames_processed,
                            stats.detections_total,
                            len(tracks),
                        )

        stats.elapsed_seconds = time.perf_counter() - started
        self.logger.info("Pipeline finished: %s", stats.summary())
        return stats

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    def _probe_optional_stages(self) -> None:
        """
        Check once which optional stages are implemented.

        Stage stubs raise ``NotImplementedError``; that is logged a single time
        and the stage is skipped for the rest of the run.
        """
        from app.team_classifier import TeamClassifier

        classifier = TeamClassifier(self.config.team_classifier)
        try:
            dummy = TrackedObject(track_id=-1, bbox=(0.0, 0.0, 1.0, 1.0), class_id=0, class_name="player")
            classifier.predict_team(np.zeros((2, 2, 3), dtype=np.uint8), dummy)
            self._team_stage_enabled = True
            self.logger.info("Team classification stage is available - enabled.")
        except NotImplementedError:
            self._team_stage_enabled = False
            self.logger.info(
                "Team classification not implemented yet (app/team_classifier.py) - "
                "players will be drawn in a neutral colour."
            )

    def _update_teams(self, frame: np.ndarray, tracks: List[TrackedObject]) -> None:
        """Assign a team to every tracked player (no-op until the stage exists)."""
        if not self._team_stage_enabled:
            return
        from app.team_classifier import TeamClassifier

        if self._team_classifier is None:
            self._team_classifier = TeamClassifier(self.config.team_classifier)
        for track in tracks:
            if track.class_name != "player":
                continue
            try:
                self.team_by_track[track.track_id] = self._team_classifier.predict_team(frame, track)
            except NotImplementedError:  # pragma: no cover - stage appears mid-run
                self._team_stage_enabled = False
                self.logger.info("Team classification became unavailable - disabling the stage.")
                return

    def _tracker_inputs(self, detections: List[Detection]) -> List[Detection]:
        """
        Detections handed to the tracker.

        With ``tracker.player_only`` (the default) non-player objects such as the
        ball are excluded: a ball is a handful of pixels large, so IoU-based IDs
        on it flicker and would pollute the player statistics. The ball is still
        detected (Stage 1 unchanged) and drawn from the raw detection.
        """
        if not self.config.tracker.player_only:
            return detections
        return [d for d in detections if d.class_name == "player"]

    def _annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        tracks: List[TrackedObject],
        frame_index: int,
    ) -> np.ndarray:
        """Draw boxes, IDs, trails, and a small HUD onto the frame."""
        self._update_teams(frame, tracks)

        players = 0
        ball_tracked = False
        for track in tracks:
            if track.class_name == "ball":
                ball_tracked = True
                self.visualizer.draw_ball_marker(frame, track.bbox, COLOR_BALL)
                continue

            players += 1
            color = self._color_for(track)
            self.visualizer.draw_player_marker(
                frame,
                track.bbox,
                track.track_id,
                color,
                confidence=track.confidence,
            )
            self._draw_trail(frame, track, color)

        # Keep the ball visible even when tracker.player_only excludes it.
        if not ball_tracked:
            for detection in detections:
                if detection.class_name == "ball":
                    self.visualizer.draw_ball_marker(frame, detection.bbox, COLOR_BALL)

        self._draw_hud(frame, frame_index, len(detections), players, len(tracks))
        return frame

    def _color_for(self, track: TrackedObject) -> tuple[int, int, int]:
        """Team colour for a player, or a neutral colour until teams are known."""
        team = self.team_by_track.get(track.track_id)
        if team is None:
            return COLOR_UNKNOWN_PLAYER
        return COLOR_TEAM_A if team % 2 == 0 else COLOR_TEAM_B

    @staticmethod
    def _draw_trail(frame: np.ndarray, track: TrackedObject, color: tuple[int, int, int]) -> None:
        """Draw the recent movement trail of a player."""
        points = track.trajectory[:-1]
        if len(points) < 2:
            return
        polyline = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [polyline], False, color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_hud(
        frame: np.ndarray,
        frame_index: int,
        detections: int,
        players: int,
        tracks: int,
    ) -> None:
        """Draw a compact status line in the top-left corner."""
        text = f"frame {frame_index:05d} | det {detections} | players {players} | tracks {tracks}"
        x, y = (10, 24)
        # Outline the text with uniform 1px black offsets: OpenCV widens the
        # glyph advance with stroke thickness, so a thick single black pass
        # would extend past the white text and look like ghost characters.
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cv2.putText(
                frame, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
            )
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
