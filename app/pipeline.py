"""
Pipeline Orchestration Module
==============================

Connects the individual stages into a runnable end-to-end flow:

    raw MP4 -> VideoReader -> FootballDetector -> PlayerTracker
            -> (optional stages) -> TacticalVisualizer -> VideoWriter

Stages that are still scaffolds (formations, passing lanes) are probed once at
startup; if they raise ``NotImplementedError`` they are skipped with a single
log line instead of crashing the run, so the pipeline starts working
automatically as they get implemented.

Stage 3 (team classification) lives in :mod:`app.team_classifier`: every
tracked player gets TEAM_A / TEAM_B / UNKNOWN, drawn in a per-team colour with
a legend, and exported per frame through :mod:`app.team_log`.

Stage 4 (pitch mapping) lives in :mod:`app.pitch`: once a manual calibration
file exists (``python main.py --calibrate``), every player's foot position is
projected onto a top-down pitch in meters, drawn as a radar minimap, and
exported per frame through :mod:`app.pitch_log`.
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
from app.pitch import PitchCoordinate, PitchTransformer, foot_position
from app.pitch_log import PitchLog
from app.team_classifier import TEAM_A, TEAM_B, TeamClassifier, team_label
from app.team_log import TeamLog
from app.tracker import PlayerTracker, TrackedObject
from app.track_log import TrackLog
from app.video import VideoReader, VideoWriter
from app.visualization import COLOR_BALL, COLOR_TEAM_A, COLOR_TEAM_B, COLOR_UNKNOWN, TacticalVisualizer

# Marker colour for players whose team is not (yet) known.
COLOR_UNKNOWN_PLAYER = COLOR_UNKNOWN

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
    # One record per tracked player per frame with the Stage 3 team label:
    # {frame, timestamp, player_id, team, center} - exportable with
    # TeamLog.save_csv() (see also the --team-csv flag in main.py).
    team_log: TeamLog = field(default_factory=TeamLog)
    # One record per tracked player per frame with its Stage 4 top-down pitch
    # position in meters - exportable with PitchLog.save_csv()
    # (see also the --pitch-csv flag in main.py).
    pitch_log: PitchLog = field(default_factory=PitchLog)

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
        self.visualizer = visualizer if visualizer is not None else TacticalVisualizer(
            pitch_size=(config.pitch.length_meters, config.pitch.width_meters)
        )
        self.logger = logging.getLogger("football_tracker.pipeline")

        # Filled by _update_teams(): team_id (TEAM_A/TEAM_B/None) per track_id,
        # persisted across frames so assignments never flicker.
        self.team_by_track: Dict[int, Optional[int]] = {}
        self._team_stage_enabled = False
        self._team_classifier = None

        # Stage 4: manual-calibration homography (enabled only when a
        # calibration file exists - see _probe_optional_stages). Pitch
        # positions are recomputed every frame, hence the plain dict.
        self.pitch_transformer = PitchTransformer(config.pitch)
        self.pitch_by_track: Dict[int, PitchCoordinate] = {}
        self.ball_pitch: Optional[PitchCoordinate] = None
        self._pitch_stage_enabled = False

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
                    # Stage 3: cluster jersey colours and assign teams first,
                    # so the annotation and the team log use the same labels.
                    self._update_teams(frame, tracks, frame_index)
                    # Stage 4: project every player's foot onto the top-down
                    # pitch (per-frame positions - players move).
                    self._update_pitch(tracks)
                    for track in tracks:
                        if track.class_name == "player":
                            stats.track_log.add_track(frame_index, timestamp, track)
                            stats.team_log.add_team(
                                frame_index,
                                timestamp,
                                track,
                                self.team_by_track.get(track.track_id),
                            )
                            coordinate = self.pitch_by_track.get(track.track_id)
                            if coordinate is not None:
                                stats.pitch_log.add_pitch(
                                    frame_index,
                                    timestamp,
                                    track,
                                    self.team_by_track.get(track.track_id),
                                    coordinate,
                                )

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
        if len(stats.team_log):
            counts = stats.team_log.team_counts
            self.logger.info(
                "Team assignments: TEAM_A=%d TEAM_B=%d UNKNOWN=%d (unique players)",
                counts["TEAM_A"], counts["TEAM_B"], counts["UNKNOWN"],
            )
        if len(stats.pitch_log):
            self.logger.info(
                "Pitch positions: %d records for %d players over %d frames",
                len(stats.pitch_log),
                len(stats.pitch_log.unique_ids),
                len(stats.pitch_log.frames),
            )
        self.logger.info("Pipeline finished: %s", stats.summary())
        return stats

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    def _probe_optional_stages(self) -> None:
        """
        Check once which optional stages are implemented.

        Stage stubs raise ``NotImplementedError``; that is logged a single time
        and the stage is skipped for the rest of the run. The probe instance is
        kept and reused for the whole run, so observations are never lost.
        """
        self._team_classifier = TeamClassifier(self.config.team_classifier)
        dummy = TrackedObject(track_id=-1, bbox=(0.0, 0.0, 1.0, 1.0), class_id=0, class_name="player")
        try:
            self._team_classifier.predict_team(np.zeros((2, 2, 3), dtype=np.uint8), dummy)
            self._team_stage_enabled = True
            self.logger.info("Team classification stage is available - enabled.")
        except NotImplementedError:
            self._team_stage_enabled = False
            self.logger.info(
                "Team classification not implemented yet (app/team_classifier.py) - "
                "players will be drawn in a neutral colour."
            )
        self._probe_pitch_stage()

    def _probe_pitch_stage(self) -> None:
        """
        Enable Stage 4 when a manual calibration file is present.

        No file -> the stage stays off with one log line (the rest of the
        pipeline is unaffected); a broken file -> the stage is skipped with a
        warning instead of aborting the run.
        """
        path = Path(self.config.pitch.calibration_path)
        if not path.exists():
            self._pitch_stage_enabled = False
            self.logger.info(
                "Pitch mapping not calibrated (no %s) - no radar/pitch output. "
                "Run 'python main.py --calibrate' to create it.",
                path,
            )
            return
        try:
            self.pitch_transformer.load_calibration(path)
        except Exception as exc:
            self._pitch_stage_enabled = False
            self.logger.warning(
                "Pitch calibration %s could not be loaded (%s) - stage disabled.",
                path,
                exc,
            )
            return
        self._pitch_stage_enabled = True
        self.logger.info(
            "Pitch mapping enabled: %d calibration points, mean error %.2f m "
            "(pitch %.0f x %.0f m)",
            self.pitch_transformer.point_count,
            self.pitch_transformer.mean_error_meters or 0.0,
            self.config.pitch.length_meters,
            self.config.pitch.width_meters,
        )

    def _update_pitch(self, tracks: List[TrackedObject]) -> None:
        """
        Project this frame's players onto the top-down pitch (Stage 4).

        Positions are recomputed every frame from each player's *foot*
        (bottom-centre of the bbox) - see :func:`app.pitch.foot_position`.
        ``self.pitch_by_track`` therefore always holds the current frame only.
        """
        if not self._pitch_stage_enabled:
            return

        self.pitch_by_track.clear()
        self.ball_pitch = None
        for track in tracks:
            if track.class_name not in ("player", "ball"):
                continue
            try:
                foot_x, foot_y = foot_position(track.bbox)
                coordinate = self.pitch_transformer.transform_point(foot_x, foot_y)
            except ValueError:
                continue  # degenerate bbox (NaN/inf) - skip this object only
            except RuntimeError:
                self._pitch_stage_enabled = False
                self.pitch_by_track.clear()
                self.logger.warning("Pitch projection failed - disabling the stage.")
                return
            if track.class_name == "ball":
                self.ball_pitch = coordinate
            else:
                self.pitch_by_track[track.track_id] = coordinate

    def _update_teams(
        self,
        frame: np.ndarray,
        tracks: List[TrackedObject],
        frame_index: int,
    ) -> None:
        """
        Assign a team to every tracked player of this frame (Stage 3).

        The jersey-colour model is refreshed and the temporally smoothed
        assignments are merged into ``self.team_by_track`` (keyed by track ID,
        so a player keeps its team across frames).
        """
        if not self._team_stage_enabled:
            return
        if self._team_classifier is None:
            self._team_classifier = TeamClassifier(self.config.team_classifier)

        players = [track for track in tracks if track.class_name == "player"]
        try:
            self.team_by_track.update(
                self._team_classifier.predict_teams(frame, players, frame_index)
            )
        except NotImplementedError:  # pragma: no cover - stage appears mid-run
            self._team_stage_enabled = False
            self.logger.info("Team classification became unavailable - disabling the stage.")

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
        """Draw boxes, IDs, trails, team legend, and a small HUD onto the frame."""
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

        # Stage 3: show which colour means which team, with live counts.
        if self._team_stage_enabled and players:
            self.visualizer.draw_team_legend(frame, self._team_counts(tracks))

        # Stage 4: top-down radar inset (extra panel - the original
        # video annotations above are left untouched).
        if (
            self._pitch_stage_enabled
            and self.config.pitch.show_radar
            and players
            and self.pitch_by_track
        ):
            radar_players = [
                (self.pitch_by_track[track.track_id], self._color_for(track), track.track_id)
                for track in tracks
                if track.class_name == "player" and track.track_id in self.pitch_by_track
            ]
            self.visualizer.draw_radar_minimap(frame, radar_players, self.ball_pitch)

        self._draw_hud(frame, frame_index, len(detections), players, len(tracks))
        return frame

    def _team_counts(self, tracks: List[TrackedObject]) -> Dict[str, int]:
        """Visible players per team label (TEAM_A / TEAM_B / UNKNOWN)."""
        counts = {"TEAM_A": 0, "TEAM_B": 0, "UNKNOWN": 0}
        for track in tracks:
            if track.class_name == "player":
                counts[team_label(self.team_by_track.get(track.track_id))] += 1
        return counts

    def _color_for(self, track: TrackedObject) -> tuple[int, int, int]:
        """Team colour for a player, or a neutral colour until teams are known."""
        team = self.team_by_track.get(track.track_id)
        if team == TEAM_A:
            return COLOR_TEAM_A
        if team == TEAM_B:
            return COLOR_TEAM_B
        return COLOR_UNKNOWN_PLAYER

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
