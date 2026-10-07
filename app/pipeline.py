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
from app.diagnostics import RunDiagnostics
from app.formation_analyzer import FormationAnalyzer
from app.pitch import PitchCoordinate, PitchTransformer, foot_position
from app.pitch_log import PitchLog
from app.pitch_renderer import (
    SEPARATOR_WIDTH,
    PitchRenderer,
    build_overlay_lines,
    pitch_panel_width,
    render_side_by_side,
)
from app.team_classifier import TEAM_A, TEAM_B, TeamClassifier, team_label, torso_roi
from app.team_graph import TeamGraphBuilder, TeamGraphResult
from app.team_log import TeamLog
from app.tracker import PlayerTracker, TrackedObject
from app.track_log import TrackLog
from app.video import VideoReader, VideoWriter
from app.visualization import COLOR_BALL, COLOR_TEAM_A, COLOR_TEAM_B, COLOR_UNKNOWN, TacticalVisualizer

# Marker colour for players whose team is not (yet) known.
COLOR_UNKNOWN_PLAYER = COLOR_UNKNOWN
# Debug overlay: badge colour for a labelled player whose model support fell
# below the weakest gate (red = committed label, weak/stale evidence).
COLOR_LOW_CONFIDENCE = (0, 0, 255)
# Debug overlay: badge background. Deliberately non-neutral (blue channel
# != red channel): no text anti-aliasing, grass, legend or team-marker blend
# can produce it, so tests can detect the badge by exact colour.
COLOR_DEBUG_BADGE = (50, 100, 150)

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
    # Stage 6: last processed frame's formation read-out per team
    # ({"TEAM_A": {...}, "TEAM_B": {...}}) - only filled in Stage 5 runs.
    formations: Dict[str, dict] = field(default_factory=dict)
    # Stage 7: last processed frame's teammate graph (edges + network
    # metrics) - only filled in Stage 5 runs.
    graph_result: Optional[TeamGraphResult] = None
    # Run diagnostics (see app.diagnostics): detection/tracking/team
    # statistics including team-label switches per track.
    diagnostics: Dict[str, object] = field(default_factory=dict)

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
        stage5: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Args:
            config: Full application configuration.
            detector: Optional detector override (defaults to ``FootballDetector``).
            tracker: Optional tracker override (defaults to ``PlayerTracker``).
            visualizer: Optional visualizer override.
            stage5: Emit the Stage 5 side-by-side output (annotated footage on
                the left, tactical pitch with Stage 6/7 overlays on the right).
                Off by default, so the plain pipeline output is unchanged.
            debug: Draw the per-player diagnostic overlay (team label +
                team confidence badge, jersey-ROI rectangle) on top of the
                normal annotation, so detection / tracking / classification
                failures can be told apart visually. Off by default.
        """
        self.config = config
        self.detector = detector if detector is not None else FootballDetector(config.model)
        self.tracker = tracker if tracker is not None else PlayerTracker(config.tracker)
        self.visualizer = visualizer if visualizer is not None else TacticalVisualizer(
            pitch_size=(config.pitch.length_meters, config.pitch.width_meters)
        )
        self.logger = logging.getLogger("football_tracker.pipeline")
        self.stage5 = bool(stage5)
        self.debug = bool(debug)
        # Per-frame detection/tracking/team statistics for this run
        # (rebuilt at the start of every run(); see app.diagnostics).
        self.diagnostics = RunDiagnostics(
            display_threshold=self.config.model.confidence_threshold
        )

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

        # Stage 5/6/7: created in _setup_stage5() once the source height is
        # known (the pitch panel is letterboxed to the video height).
        self.pitch_renderer: Optional[PitchRenderer] = None
        self.formation_analyzer: Optional[FormationAnalyzer] = None
        self.graph_builder: Optional[TeamGraphBuilder] = None
        # Side-by-side composite output (Stage 5 flag, off by default).
        self._stage5_enabled = False
        # Stage 6/7 analysis needs pitch positions, i.e. Stage 5 + calibration.
        self._tactics_enabled = False

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
        self.diagnostics = RunDiagnostics(
            display_threshold=self.config.model.confidence_threshold
        )
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

            # Stage 5: build the pitch panel at the source height and open the
            # writer with the *exact* composite width (the shared
            # pitch_panel_width() helper is what render_side_by_side uses too).
            self._setup_stage5(reader.height)
            output_width = reader.width
            if self._stage5_enabled and self.pitch_renderer is not None:
                output_width = (
                    reader.width
                    + SEPARATOR_WIDTH
                    + pitch_panel_width(reader.height, self.pitch_renderer)
                )

            with VideoWriter(
                output_path,
                fps=source_fps / stride,
                width=output_width,
                height=reader.height,
            ) as writer:
                if not writer.isOpened():  # open() already raises; belt and braces
                    raise RuntimeError(f"Video writer did not open: {output_path}")

                for frame_index, frame in reader.read_frames():
                    detections = self.detector.detect(frame)
                    tracks = self.tracker.update(self._tracker_inputs(detections), frame)

                    timestamp = frame_index / source_fps
                    # Stage 3: cluster jersey colours and assign teams first,
                    # so the annotation and the team log use the same labels.
                    self._update_teams(frame, tracks, frame_index)
                    # Bookkeeping for the end-of-run diagnostics report
                    # (attributes missing players / wrong labels to detection,
                    # tracking or classification - see app.diagnostics).
                    self.diagnostics.record(
                        frame_index, detections, tracks, self.team_by_track
                    )
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

                    # Stage 5: stamp pitch coordinates onto the tracks (the
                    # renderer reads track.pitch_coordinate).
                    positions: List[tuple] = []
                    if self._stage5_enabled:
                        positions = self._tactics_positions(tracks)

                    # Stage 6 (formation) + Stage 7 (teammate graph): one
                    # analysis pass per frame; results are drawn on the pitch.
                    formations: Optional[Dict[str, dict]] = None
                    graph_result: Optional[TeamGraphResult] = None
                    if self._tactics_enabled:
                        formations = self.formation_analyzer.update(positions)  # type: ignore[union-attr]
                        graph_result = self.graph_builder.update(positions)  # type: ignore[union-attr]
                        stats.formations = formations
                        stats.graph_result = graph_result

                    annotated = self._annotate(frame, detections, tracks, frame_index)
                    if self._stage5_enabled:
                        annotated = render_side_by_side(
                            annotated,
                            self.pitch_renderer,  # type: ignore[arg-type]
                            tracks,
                            self.team_by_track,
                            edges=graph_result.edges if graph_result is not None else (),
                            overlay_lines=build_overlay_lines(formations, graph_result),
                        )
                    writer.write(annotated)

                    stats.frames_processed += 1
                    # Display-grade detections only (>= confidence_threshold):
                    # the weaker candidates exist solely for the tracker and
                    # must not inflate reported counts (log/HUD stay
                    # comparable with runs that only emit display detections).
                    stats.detections_total += sum(
                        1
                        for d in detections
                        if d.confidence >= self.config.model.confidence_threshold
                    )
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
        for label in ("TEAM_A", "TEAM_B"):
            info = stats.formations.get(label)
            if info and info.get("players"):
                self.logger.info(
                    "Formation (last frame) %s: %s - confidence %.0f%%, "
                    "width %.1f m, depth %.1f m, compactness %.3f",
                    label,
                    info["formation"],
                    info["confidence"] * 100.0,
                    info["width"],
                    info["depth"],
                    info["compactness"],
                )
        if stats.graph_result is not None:
            for label in ("TEAM_A", "TEAM_B"):
                metric = stats.graph_result.metrics.get(label, {})
                if metric.get("players"):
                    self.logger.info(
                        "Teammate graph (last frame) %s: %d connections, "
                        "density %.2f, avg distance %.1f m, max distance %.1f m",
                        label,
                        metric["connections"],
                        metric["density"],
                        metric["avg_teammate_distance"],
                        metric["max_teammate_distance"],
                    )
        stats.diagnostics = self.diagnostics.summary()
        self.logger.info("Run diagnostics:\n%s", self.diagnostics.text())
        self.logger.info("Pipeline finished: %s", stats.summary())
        return stats

    # ------------------------------------------------------------------ #
    # Stage 5 / 6 / 7 helpers
    # ------------------------------------------------------------------ #

    def _setup_stage5(self, video_height: int) -> None:
        """
        Create the Stage 5/6/7 objects once the source height is known.

        The pitch panel is letterboxed to ``video_height``, and its width must
        be fixed *before* the ``VideoWriter`` opens (writer size and frame size
        have to match exactly - codecs round odd widths down). Rendering the
        panel directly at the final size also avoids a rescaling pass, so the
        pitch and its Stage 6/7 overlay stay crisp.

        Stage 6/7 analysis additionally needs pitch positions, so it is only
        enabled when Stage 5 was requested *and* a calibration exists.
        """
        self._stage5_enabled = self.stage5
        self._tactics_enabled = False
        if not self.stage5:
            return

        base = PitchRenderer(
            pitch_length_meters=self.config.pitch.length_meters,
            pitch_width_meters=self.config.pitch.width_meters,
        )
        panel_width = pitch_panel_width(video_height, base)
        self.pitch_renderer = PitchRenderer(
            pitch_length_meters=self.config.pitch.length_meters,
            pitch_width_meters=self.config.pitch.width_meters,
            rendered_size=(panel_width, video_height),
        )
        self.logger.info(
            "Stage 5 side-by-side enabled: pitch panel %d px wide at %d px "
            "height.",
            panel_width,
            video_height,
        )

        if not self._pitch_stage_enabled:
            self.logger.warning(
                "--stage5 requested but pitch mapping is not calibrated - the "
                "pitch panel will be empty and Stage 6/7 analysis is skipped. "
                "Run 'python main.py --calibrate' first."
            )
            return

        tactics = self.config.tactics
        self.formation_analyzer = FormationAnalyzer(
            pitch_length=self.config.pitch.length_meters,
            pitch_width=self.config.pitch.width_meters,
            window_size=tactics.formation_window,
            min_confidence=tactics.formation_min_confidence,
        )
        self.graph_builder = TeamGraphBuilder(
            max_connection_distance=tactics.max_connection_distance,
            k_neighbors=tactics.connection_k_neighbors,
            hysteresis_ratio=tactics.edge_hysteresis_ratio,
        )
        self._tactics_enabled = True
        self.logger.info(
            "Stage 6/7 analysis enabled: formation window %d frames "
            "(min confidence %.2f), max connection distance %.1f m "
            "(k=%d neighbours).",
            tactics.formation_window,
            tactics.formation_min_confidence,
            tactics.max_connection_distance,
            tactics.connection_k_neighbors,
        )

    def _tactics_positions(self, tracks: List[TrackedObject]) -> List[tuple]:
        """
        ``(track_id, team_label, PitchCoordinate)`` for every player that has a
        pitch position this frame - the shared input format of the Stage 6
        formation analyzer and the Stage 7 graph builder.

        Also stamps ``pitch_coordinate`` onto each track, which is how the
        Stage 5 renderer learns where to draw the player on the panel.
        """
        positions: List[tuple] = []
        for track in tracks:
            if track.class_name != "player":
                continue
            coordinate = self.pitch_by_track.get(track.track_id)
            if coordinate is None:
                continue
            track.pitch_coordinate = coordinate
            positions.append(
                (
                    track.track_id,
                    team_label(self.team_by_track.get(track.track_id)),
                    coordinate,
                )
            )
        return positions

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
            if self.debug:
                self._draw_debug_player(frame, track, color)

        # Keep the ball visible even when tracker.player_only excludes it -
        # at display quality only, so weak candidates cannot draw markers.
        if not ball_tracked:
            for detection in detections:
                if (
                    detection.class_name == "ball"
                    and detection.confidence >= self.config.model.confidence_threshold
                ):
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

        # HUD counts display-grade detections (candidates are tracker-only).
        displayable = sum(
            1
            for d in detections
            if d.confidence >= self.config.model.confidence_threshold
        )
        self._draw_hud(frame, frame_index, displayable, players, len(tracks))
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

    def _draw_debug_player(
        self,
        frame: np.ndarray,
        track: TrackedObject,
        color: tuple[int, int, int],
    ) -> None:
        """
        Debug overlay for one player (``--debug``).

        Draws two things the normal annotation omits:

        * the jersey ROI rectangle - exactly the pixels Stage 3 reads (shared
          geometry via :func:`app.team_classifier.torso_roi`), so a crop that
          lands on grass / a blurred torso is visible immediately;
        * a badge under the player with ``LABEL confidence`` from the team
          model. UNKNOWN stays white; a labelled player whose evidence
          dropped below the weakest gate (< 0.25) gets a red badge, which
          makes stale or wrong assignments stand out frame by frame.
        """
        left, top, right, bottom = torso_roi(track.bbox, self.config.team_classifier)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 1)

        if not self._team_stage_enabled or self._team_classifier is None:
            return
        label = team_label(self.team_by_track.get(track.track_id))
        conf = self._team_classifier.team_confidence(track.track_id)
        text = f"{label} {conf:.2f}"
        if label == "UNKNOWN":
            text_color = COLOR_UNKNOWN_PLAYER
        elif conf < 0.25:
            text_color = COLOR_LOW_CONFIDENCE
        else:
            text_color = color

        frame_h, frame_w = frame.shape[:2]
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        tx = min(max(x1, 2), max(2, frame_w - text_w - 4))
        # Below the foot ellipse (vertical radius ~ bbox width / 6); when the
        # player sits at the bottom edge, fall back above the ID label.
        ty = y2 + max(x2 - x1, 8) // 6 + text_h + 8
        if ty + 3 > frame_h:
            ty = max(y1 - 26, text_h + 6)
        cv2.rectangle(
            frame, (tx - 3, ty - text_h - 3), (tx + text_w + 3, ty + 3), COLOR_DEBUG_BADGE, -1
        )
        if text_color == COLOR_LOW_CONFIDENCE:
            cv2.rectangle(
                frame, (tx - 3, ty - text_h - 3), (tx + text_w + 3, ty + 3),
                COLOR_LOW_CONFIDENCE, 1,
            )
        cv2.putText(
            frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA
        )

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
