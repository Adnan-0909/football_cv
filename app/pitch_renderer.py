"""
Reusable pitch-rendering module for Stage 5.

Provides :class:`PitchRenderer` which can render a top-down football pitch
with all standard markings and player positions from the homography stage.

The pitch coordinate system matches the existing convention:
  - x: along the pitch length, 0 = left goal line, 105 = right goal line
  - y: along the pitch width, 0 = near sideline (camera side), 68 = far sideline
  - Origin (0, 0) is the bottom-left of the pitch in "ground" orientation.
Rendering orients y=0 at the bottom of the panel to match the camera view
(the camera sits on the near side, so near sideline appears at the bottom of
the video frame).

Stage 7 additions: :meth:`PitchRenderer.render` draws teammate connections
(edges from :mod:`app.team_graph`) *underneath* the player dots so nodes and
track IDs stay on top, and :func:`render_side_by_side` can stamp a Stage 6/7
text overlay onto the pitch panel. The graph *algorithm* lives in
:mod:`app.team_graph` - this module only draws what it is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from app.pitch import PitchCoordinate, draw_pitch
from app.team_classifier import team_label
from app.team_graph import GraphEdge, TeamGraphResult

# ---------------------------------------------------------------------------
# Colours (BGR, OpenCV)
# ---------------------------------------------------------------------------

COLOR_TEAM_A: Tuple[int, int, int] = (255, 60, 0)       # blueish
COLOR_TEAM_B: Tuple[int, int, int] = (0, 165, 255)      # orange
COLOR_UNKNOWN_PLAYER: Tuple[int, int, int] = (255, 255, 255)  # neutral white
COLOR_BALL: Tuple[int, int, int] = (0, 255, 255)        # yellow

# One overlay line: plain text, or text plus a BGR colour.
OverlayLine = Union[str, Tuple[str, Tuple[int, int, int]]]

# ---------------------------------------------------------------------------
# Pitch markings constants (drawn via draw_pitch())
# ---------------------------------------------------------------------------

_PENALTY_BOX_DEPTH = 16.5
_PENALTY_BOX_WIDTH = 40.32
_GOAL_AREA_DEPTH = 5.5
_GOAL_AREA_WIDTH = 18.32
_PENALTY_SPOT = 11.0
_CENTER_RADIUS = 9.15
_CORNER_RADIUS = 1.0

# Width of the 4px separator between the video and the pitch panel.
SEPARATOR_WIDTH = 4


@dataclass
class _PlayerRenderInfo:
    """Internal data carried from the caller to the renderer."""
    track_id: Optional[int]
    color: Tuple[int, int, int]
    coordinate: PitchCoordinate


class PitchRenderer:
    """
    Renders a top-down football pitch with player positions.

    The rendered pitch can be blended or placed side-by-side with the original
    video frame.  It uses the same homography calibration that Stage 4 employs,
    so player positions are already in pitch-metres.

    Parameters
    ----------
    pitch_length_meters:
        Pitch length in metres (default ``105.0``).
    pitch_width_meters:
        Pitch width in metres (default ``68.0``).
    rendered_size:
        (width, height) in pixels of the output pitch diagram.  The height
        determines the vertical resolution; the width scales proportionally.
    """

    def __init__(
        self,
        pitch_length_meters: float = 105.0,
        pitch_width_meters: float = 68.0,
        rendered_size: Tuple[int, int] = (700, 450),
    ) -> None:
        self.pitch_length = pitch_length_meters
        self.pitch_width = pitch_width_meters
        self.rendered_width, self.rendered_height = rendered_size

    # ------------------------------------------------------------------ #
    # Public rendering API
    # ------------------------------------------------------------------ #

    def render(
        self,
        players: Sequence[_PlayerRenderInfo],
        *,
        edges: Sequence[GraphEdge] = (),
        draw_ground: bool = True,
        draw_goals: bool = True,
    ) -> np.ndarray:
        """
        Render the pitch diagram with player positions.

        The pitch is oriented so that *y = 0* (near sideline) appears at the
        *bottom* of the panel — matching the camera view (the camera sits on the
        near side).  Player positions are taken directly from the homography
        Stage 4 coordinates.

        Parameters
        ----------
        players:
            Sequence of ``_PlayerRenderInfo``, each containing a track ID,
            a BGR colour (team colour), and a ``PitchCoordinate``.
        edges:
            Stage 7 teammate connections, drawn as team-coloured lines
            *before* the player dots so nodes and track IDs stay on top.
            Edges whose endpoints are not in ``players`` are skipped.
        draw_ground:
            If ``True`` (default) the pitch boundaries, halfway line, centre
            circle, penalty areas, goal areas and corner arcs are drawn.
        draw_goals:
            If ``True`` (default) the goal lines are drawn; otherwise the
            goal-area rectangles are still rendered for reference.

        Returns
        -------
        np.ndarray
            Rendered pitch image in BGR format, size ``(rendered_height,
            rendered_width, 3)`` with 0–255 values.
        """
        # ------------------------------------------------------------------
        # 1. Build the pitch diagram (draw_pitch puts (0,0) at top-left,
        #    +y downward).  We flip it so y=0 ends up at the bottom,
        #    matching the camera view.
        # ------------------------------------------------------------------
        radar = draw_pitch(
            self.pitch_length,
            self.pitch_width,
            (self.rendered_width, self.rendered_height),
            background=(45, 120, 45),  # grass green
            line_color=(255, 255, 255),  # white markings
            thickness=None,
        )
        # Flip vertically so near sideline (y=0) is at the bottom.
        radar = cv2.flip(radar, 0)

        # ------------------------------------------------------------------
        # 2. Map each player's pitch coordinate to pixel positions.
        # ------------------------------------------------------------------
        scale_x = self.rendered_width / float(self.pitch_length)
        scale_y = self.rendered_height / float(self.pitch_width)
        dot_r = max(3, self.rendered_height // 55)

        placed: List[Tuple[_PlayerRenderInfo, int, int]] = []
        pixel_by_id: Dict[int, Tuple[int, int, Tuple[int, int, int]]] = {}
        for info in players:
            # Clamp so wildly-bad projections still render inside the panel.
            px = min(max(int(round(info.coordinate.x * scale_x)), 0), self.rendered_width)
            # y=0 (near) -> bottom of panel: (pitch_width - y) * scale_y
            py = min(
                max(int(round((self.pitch_width - info.coordinate.y) * scale_y)), 0),
                self.rendered_height,
            )
            placed.append((info, px, py))
            if info.track_id is not None:
                pixel_by_id[info.track_id] = (px, py, info.color)

        # ------------------------------------------------------------------
        # 3. Stage 7: teammate connections, drawn under the dots.
        # ------------------------------------------------------------------
        for edge in edges:
            start = pixel_by_id.get(edge.id_a)
            end = pixel_by_id.get(edge.id_b)
            if start is None or end is None:
                continue  # endpoint off-pitch or not tracked this frame
            color = start[2]
            cv2.line(radar, start[:2], end[:2], color, 2, cv2.LINE_AA)

        # ------------------------------------------------------------------
        # 4. Player dots + track IDs (always on top of the edges).
        # ------------------------------------------------------------------
        for info, px, py in placed:
            color = info.color
            cv2.circle(radar, (px, py), dot_r, color, -1, cv2.LINE_AA)
            cv2.circle(radar, (px, py), dot_r, (0, 0, 0), 1, cv2.LINE_AA)

            if info.track_id is not None:
                cv2.putText(
                    radar,
                    str(info.track_id),
                    (px + dot_r + 1, py - dot_r // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return radar

    # ------------------------------------------------------------------
    # Helper: convert a tracked object into _PlayerRenderInfo
    # ------------------------------------------------------------------

    @staticmethod
    def _from_track(
        track: object,
        team_by_track: dict,
        color_team_a: Tuple[int, int, int] = COLOR_TEAM_A,
        color_team_b: Tuple[int, int, int] = COLOR_TEAM_B,
    ) -> Optional[_PlayerRenderInfo]:
        """
        Create a ``_PlayerRenderInfo`` from a ``TrackedObject`` (or any object
        with ``track_id``, ``class_name``, ``bbox``, and a team assignment).

        ``team_by_track`` holds team *ids* (``TEAM_A``/``TEAM_B`` from
        :mod:`app.team_classifier`, or ``None``) but string labels are accepted
        too.  Returns ``None`` if the track has no meaningful pitch coordinate.
        """
        if track is None:
            return None

        track_id = getattr(track, "track_id", None)
        class_name = getattr(track, "class_name", None)

        # Only render players (the homography stage already excludes the ball).
        if class_name != "player":
            return None

        raw_team = team_by_track.get(track_id)  # 0 / 1 / None (or a label)
        team = raw_team if isinstance(raw_team, str) else team_label(raw_team)
        if team == "TEAM_A":
            color = color_team_a
        elif team == "TEAM_B":
            color = color_team_b
        else:
            color = COLOR_UNKNOWN_PLAYER

        # The pipeline stamps track.pitch_coordinate from its per-frame
        # pitch_by_track dict before rendering (see
        # TacticalPipeline._tactics_positions).  Without it we cannot place
        # the player, so return None and let the caller skip gracefully.
        coordinate = getattr(track, "pitch_coordinate", None)
        if coordinate is None:
            return None

        return _PlayerRenderInfo(
            track_id=track_id,
            color=color,
            coordinate=coordinate,
        )


# ---------------------------------------------------------------------------
# Letterboxing / composite geometry
# ---------------------------------------------------------------------------

def pitch_panel_width(video_height: int, pitch_renderer: PitchRenderer) -> int:
    """
    Exact width of the pitch panel after letterboxing it to ``video_height``.

    Both :func:`render_side_by_side` and the pipeline (which opens the
    ``VideoWriter``) use this single expression, so the writer width and the
    frame width can never disagree.

    The width is forced even: video codecs round odd widths down on read-back
    (mp4v/MJPG store yuv420p), which would make the written file one pixel
    narrower than the frames handed to the writer.
    """
    width = int(
        pitch_renderer.rendered_width * video_height / pitch_renderer.rendered_height
    )
    if width % 2:
        width += 1
    return max(width, 2)


def build_overlay_lines(
    formations: Optional[Mapping[str, dict]] = None,
    graph: Optional[TeamGraphResult] = None,
) -> List[OverlayLine]:
    """
    Format the Stage 6 formation read-out and Stage 7 network metrics as
    overlay lines (one entry per line, optionally team-coloured).

    Args:
        formations: Result of :meth:`FormationAnalyzer.update`, or ``None``.
        graph: Result of :meth:`TeamGraphBuilder.update`, or ``None``.

    Returns:
        Lines ready to pass to :func:`render_side_by_side`.
    """
    lines: List[OverlayLine] = []
    colours = {"TEAM_A": COLOR_TEAM_A, "TEAM_B": COLOR_TEAM_B}

    for label in ("TEAM_A", "TEAM_B"):
        info = (formations or {}).get(label)
        if not info or not info.get("players"):
            continue
        lines.append(
            (
                f"{label} {info['formation']} {info['confidence']:.0%}"
                f"  w{info['width']:.0f}m d{info['depth']:.0f}m",
                colours.get(label, COLOR_UNKNOWN_PLAYER),
            )
        )

    if graph is not None:
        for label in ("TEAM_A", "TEAM_B"):
            metric = graph.metrics.get(label)
            if not metric or not metric.get("players"):
                continue
            lines.append(
                (
                    f"{label} net {metric['connections']} links"
                    f" dens {metric['density']:.2f}"
                    f" avg {metric['avg_teammate_distance']:.1f}m"
                    f" max {metric['max_teammate_distance']:.1f}m",
                    colours.get(label, COLOR_UNKNOWN_PLAYER),
                )
            )
    return lines


def draw_overlay(panel: np.ndarray, lines: Sequence[OverlayLine]) -> np.ndarray:
    """
    Stamp a small translucent text block into the top-left of a pitch panel.

    Args:
        panel: Pitch panel image (modified in place, also returned).
        lines: Lines from :func:`build_overlay_lines` (str or ``(str, color)``).

    Returns:
        The panel, for convenience.
    """
    if not lines:
        return panel

    height, width = panel.shape[:2]
    font = min(0.5, max(0.32, width / 2500.0))
    thickness = 1
    line_step = int(round(font * 24)) + 4
    pad = 6

    texts = [line if isinstance(line, str) else line[0] for line in lines]
    text_widths = [
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thickness)[0][0]
        for text in texts
    ]
    box_w = min(width - 2 * pad, max(text_widths) + 2 * pad)
    box_h = line_step * len(lines) + 2 * pad

    overlay = panel.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, panel)

    y = pad + line_step - int(line_step * 0.35)
    for line in lines:
        text, color = (line, (255, 255, 255)) if isinstance(line, str) else line
        # Uniform 1px black outline (see pipeline._draw_hud for the rationale).
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cv2.putText(
                panel,
                text,
                (pad + pad // 2 + dx, y + dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                font,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )
        cv2.putText(
            panel,
            text,
            (pad + pad // 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_step
    return panel


# ---------------------------------------------------------------------------
# Convenience wrapper: render a full side-by-side frame
# ---------------------------------------------------------------------------

def render_side_by_side(
    video_frame: np.ndarray,
    pitch_renderer: PitchRenderer,
    tracks: list,
    team_by_track: dict,
    *,
    draw_ground: bool = True,
    draw_goals: bool = True,
    edges: Sequence[GraphEdge] = (),
    overlay_lines: Sequence[OverlayLine] = (),
) -> np.ndarray:
    """
    Return a new frame consisting of:

    * LEFT: *video_frame* with the pipeline's usual annotations (boxes, IDs,
      legend, HUD) left untouched.
    * RIGHT: a top-down pitch diagram with the current player positions,
      Stage 7 teammate edges and, optionally, a Stage 6/7 text overlay.

    The returned array has the same height as ``video_frame`` and a width of
    ``video_frame.shape[1] + SEPARATOR_WIDTH + pitch_panel_width(height)``
    (see :func:`pitch_panel_width`).  The two halves are separated by a thin
    grey bar.

    Parameters
    ----------
    video_frame
        Current BGR frame from the original video (already annotated with
        boxes/labels by the pipeline's ``_annotate`` method).
    pitch_renderer
        A configured ``PitchRenderer`` instance.
    tracks
        List of ``TrackedObject`` instances for the current frame; each must
        carry a ``pitch_coordinate`` (the pipeline attaches it) to be drawn.
    team_by_track
        Dict mapping ``track_id`` → team id (``TEAM_A``/``TEAM_B``/``None``)
        or a string label.
    draw_ground
        Passed through to :meth:`PitchRenderer.render`.
    draw_goals
        Passed through to :meth:`PitchRenderer.render`.
    edges
        Stage 7 teammate connections to draw on the pitch.
    overlay_lines
        Stage 6/7 text lines stamped onto the pitch panel (after letterboxing,
        so the text stays crisp at the video's resolution).

    Returns
    -------
    np.ndarray
        Composite frame ``(H, W_total, 3)`` ready for VideoWriter output.
    """
    # ---- Left half: keep the original video as-is (already annotated) ----
    left = video_frame.copy()

    # ---- Right half: build the pitch ----
    # Convert tracks into render info (skip non-players gracefully).
    render_infos: list[_PlayerRenderInfo] = []
    for trk in tracks:
        info = PitchRenderer._from_track(trk, team_by_track)
        if info is not None:
            render_infos.append(info)

    right = pitch_renderer.render(
        render_infos,
        edges=edges,
        draw_ground=draw_ground,
        draw_goals=draw_goals,
    )

    # ---- Letterbox the pitch to the video height ----
    video_h, video_w = left.shape[:2]
    target_w = pitch_panel_width(video_h, pitch_renderer)
    pitch_h, pitch_w = right.shape[:2]
    if (pitch_w, pitch_h) != (target_w, video_h):
        right = cv2.resize(right, (target_w, video_h), interpolation=cv2.INTER_AREA)

    # Stage 6/7 text block - drawn after the resize so it is not blurred or
    # shrunk by the letterboxing.
    draw_overlay(right, overlay_lines)

    # Ensure both halves have the same height.
    left_h, left_w = left.shape[:2]
    right_h, right_w = right.shape[:2]
    assert left_h == right_h, "Heights must match after padding"

    # Build the composite: left video, separator, pitch
    separator = np.full((left_h, SEPARATOR_WIDTH, 3), (128, 128, 128), dtype=np.uint8)
    return np.hstack([left, separator, right])
