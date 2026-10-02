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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.pitch import PitchCoordinate, draw_pitch, foot_position

# ---------------------------------------------------------------------------
# Colours (BGR, OpenCV)
# ---------------------------------------------------------------------------

COLOR_TEAM_A: Tuple[int, int, int] = (255, 60, 0)       # blueish
COLOR_TEAM_B: Tuple[int, int, int] = (0, 165, 255)      # orange
COLOR_UNKNOWN_PLAYER: Tuple[int, int, int] = (255, 255, 255)  # neutral white
COLOR_BALL: Tuple[int, int, int] = (0, 255, 255)        # yellow

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
        # 2. Map each player's pitch coordinate to pixel positions and draw.
        # ------------------------------------------------------------------
        scale_x = self.rendered_width / float(self.pitch_length)
        scale_y = self.rendered_height / float(self.pitch_width)
        dot_r = max(3, self.rendered_height // 55)

        for info in players:
            # Clamp so wildly-bad projections still render inside the panel.
            px = min(max(int(round(info.coordinate.x * scale_x)), 0), self.rendered_width)
            # y=0 (near) -> bottom of panel: (pitch_width - y) * scale_y
            py = min(
                max(int(round((self.pitch_width - info.coordinate.y) * scale_y)), 0),
                self.rendered_height,
            )

            color = info.color
            # Draw player dot with outline
            cv2.circle(radar, (px, py), dot_r, color, -1, cv2.LINE_AA)
            cv2.circle(radar, (px, py), dot_r, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw track ID if available
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

        Returns ``None`` if the track has no meaningful pitch coordinate.
        """
        if track is None:
            return None

        track_id = getattr(track, "track_id", None)
        class_name = getattr(track, "class_name", None)

        # Only render players (the homography stage already excludes the ball).
        if class_name != "player":
            return None

        team = team_by_track.get(track_id)  # TEAM_A, TEAM_B, or None/UNKNOWN

        # Choose colour based on team assignment.
        if team == "TEAM_A" or team is None:
            # If team is not yet assigned we fall back to neutral/unknown.
            color = color_team_a if team == "TEAM_A" else COLOR_UNKNOWN_PLAYER
        elif team == "TEAM_B":
            color = color_team_b
        else:
            color = COLOR_UNKNOWN_PLAYER

        # Obtain pitch coordinate from the global pitch_by_track dict that the
        # pipeline maintains.  The renderer does not store its own copy – it
        # expects the caller to pass in pre-resolved info.  This keeps the
        # module pure and testable.
        coordinate = getattr(track, "pitch_coordinate", None)
        if coordinate is None:
            # If the caller hasn't pre-attached it, we cannot render – return None
            # so the caller can skip this track gracefully.
            return None

        return _PlayerRenderInfo(
            track_id=track_id,
            color=color,
            coordinate=coordinate,
        )

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
) -> np.ndarray:
    """
    Return a new frame consisting of:

    * LEFT half:  *video_frame* with player bounding boxes, IDs and team
      legend already drawn (the caller typically uses
    *right* half: a top-down pitch diagram with the current positions of all
    tracked players.

    The returned array has the same height as ``video_frame`` and a width
    equal to ``video_frame.shape[1] + pitch_renderer.rendered_width``.  The
    two halves are placed side-by-side with a thin grey separator line.

    Parameters
    ----------
    video_frame
        Current BGR frame from the original video (already annotated with
        boxes/labels by the pipeline's ``_annotate`` method).
    pitch_renderer
        A configured ``PitchRenderer`` instance.
    tracks
        List of ``TrackedObject`` instances for the current frame.
    team_by_track
        Dict mapping ``track_id`` → ``"TEAM_A"``/``"TEAM_B"``/``None``.
    draw_ground
        Passed through to :meth:`PitchRenderer.render`.
    draw_goals
        Passed through to :meth:`PitchRenderer.render`.

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
        render_infos, draw_ground=draw_ground, draw_goals=draw_goals
    )

    # ---- Combine side-by-side ----
    video_h, video_w = left.shape[:2]
    pitch_h, pitch_w = right.shape[:2]

    # Letterbox the pitch to match the video height, preserving aspect ratio.
    if pitch_h != video_h:
        scale = video_h / pitch_h
        new_w = int(pitch_w * scale)
        right = cv2.resize(right, (new_w, video_h), interpolation=cv2.INTER_AREA)
        pitch_h, pitch_w = video_h, new_w

    # Ensure both halves have the same height after potential padding.
    left_h, left_w = left.shape[:2]
    right_h, right_w = right.shape[:2]
    assert left_h == right_h, "Heights must match after padding"

    # Build the composite: left video, separator, pitch
    separator_w = 4  # thin grey line
    composite = np.hstack(
        [
            left,
            np.full((left_h, separator_w, 3), (128, 128, 128), dtype=np.uint8),
            right,
        ]
    )
    return composite