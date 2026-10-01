"""
Visualization Module
====================

Provides overlay drawing tools for broadcasting tactical insights on video frames:
- Player foot ellipses / bounding boxes with team color and track IDs
- Ball tracking highlighting
- Voronoi and Delaunay player connection graphs
- Passing lane overlays (green for open, red for blocked)
- Top-down 2D radar pitch overlay (minimap)
- Tactical metrics HUD (compactness, width, depth)
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from app.pitch import PitchCoordinate, draw_pitch
from app.tracker import TrackedObject

logger = logging.getLogger(__name__)


# Standard Tactical Color Palette (BGR for OpenCV)
# TEAM_A / TEAM_B colours are *annotation* colours (which real kit maps to
# which team is decided by clustering, never by a fixed colour rule).
COLOR_TEAM_A = (255, 60, 0)      # Blueish
COLOR_TEAM_B = (0, 165, 255)     # Orange
COLOR_UNKNOWN = (255, 255, 255)  # Neutral white for unclassified players
COLOR_BALL = (0, 255, 255)       # Yellow
COLOR_REFEREE = (0, 255, 0)      # Green
COLOR_LANE_OPEN = (50, 205, 50)  # Lime Green
COLOR_LANE_BLOCKED = (0, 0, 255) # Red

# Legend entries: label -> swatch colour (mirrors app.team_classifier labels).
TEAM_LEGEND_COLORS: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("TEAM_A", COLOR_TEAM_A),
    ("TEAM_B", COLOR_TEAM_B),
    ("UNKNOWN", COLOR_UNKNOWN),
)


class TacticalVisualizer:
    """
    Renders visual tactical annotations on video frames.
    """

    def __init__(
        self,
        pitch_radar_size: Tuple[int, int] = (300, 200),
        pitch_size: Tuple[float, float] = (105.0, 68.0),
    ) -> None:
        """
        Initialize the visualizer.

        Args:
            pitch_radar_size: (width, height) of the mini-map 2D pitch radar.
            pitch_size: (length, width) of the real pitch in meters - used to
                map pitch coordinates onto the radar drawing.
        """
        self.radar_width, self.radar_height = pitch_radar_size
        self.pitch_length, self.pitch_width = pitch_size
        logger.info(
            "Initialized TacticalVisualizer (radar_size=%dx%d, pitch=%.0fm x %.0fm)",
            self.radar_width,
            self.radar_height,
            self.pitch_length,
            self.pitch_width,
        )

    def draw_player_marker(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        track_id: Optional[int] = None,
        color: Tuple[int, int, int] = (255, 255, 255),
        confidence: Optional[float] = None,
        draw_box: bool = True,
    ) -> np.ndarray:
        """
        Draw a player's bounding box, a foot ellipse, and an ID / confidence label.

        The label (``#7`` or ``#7 0.87``) sits above the box so it never collides
        with the ellipse, and is repositioned inside the frame when the player is
        cut off by an edge.

        Args:
            frame: Video frame image.
            bbox: (x1, y1, x2, y2) player bounding box.
            track_id: Optional tracking identifier.
            color: BGR color tuple used for the box, ellipse, and label.
            confidence: Optional detection confidence to show next to the ID.
            draw_box: Draw the rectangle around the player (default True).

        Returns:
            np.ndarray: Annotated frame.
        """
        x1, y1, x2, y2 = map(int, bbox)
        center_x = (x1 + x2) // 2
        bottom_y = y2
        frame_h, frame_w = frame.shape[:2]

        # Bounding box around the player.
        if draw_box:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Foot ellipse
        axes = ((x2 - x1) // 2, (x2 - x1) // 6)
        cv2.ellipse(frame, (center_x, bottom_y), axes, 0, -45, 235, color, 2)

        # ID (+ confidence) label
        if track_id is not None and confidence is not None:
            label = f"#{track_id} {confidence:.2f}"
        elif track_id is not None:
            label = f"#{track_id}"
        elif confidence is not None:
            label = f"{confidence:.2f}"
        else:
            label = None

        if label:
            label_x = min(max(x1, 4), max(4, frame_w - 90))
            label_y = y1 - 8
            # Head cut off by the top edge -> show the label inside the box.
            if label_y < 14:
                label_y = min(y1 + 17, frame_h - 4)
            # Black 1px outline keeps the label readable on any background.
            # (A thick pass would widen OpenCV's glyph advance and ghost.)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cv2.putText(
                    frame,
                    label,
                    (label_x + dx, label_y + dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        return frame

    def draw_ball_marker(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        color: Tuple[int, int, int] = COLOR_BALL,
    ) -> np.ndarray:
        """
        Draw a marker pointer above the ball.

        Args:
            frame: Video frame image.
            bbox: (x1, y1, x2, y2) ball bounding box.
            color: BGR color tuple.

        Returns:
            np.ndarray: Annotated frame.
        """
        x1, y1, x2, y2 = map(int, bbox)
        center_x = (x1 + x2) // 2
        top_y = y1 - 5

        # Draw triangle pointer pointing down to the ball
        pts = np.array(
            [[center_x, top_y], [center_x - 6, top_y - 12], [center_x + 6, top_y - 12]],
            np.int32,
        )
        cv2.drawContours(frame, [pts], 0, color, -1)
        return frame

    def draw_team_legend(
        self,
        frame: np.ndarray,
        counts: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Draw the team colour legend (swatch + label + player count).

        The legend is what makes the per-team annotations readable: Team A and
        Team B players are drawn in their own marker colours, unclassified
        players stay neutral, and the counts show how many of each are visible
        right now.

        Args:
            frame: Video frame image.
            counts: Optional ``{"TEAM_A": n, "TEAM_B": n, "UNKNOWN": n}``.

        Returns:
            np.ndarray: Frame with the legend in the top-right corner.
        """
        counts = counts or {}
        row_h, pad, swatch = 22, 6, 14
        panel_w, panel_h = 132, pad * 2 + row_h * len(TEAM_LEGEND_COLORS)
        frame_h, frame_w = frame.shape[:2]
        x0 = max(4, frame_w - panel_w - 8)
        y0 = 8

        # Opaque dark panel keeps the legend readable over grass and crowds.
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (25, 25, 25), -1)

        for i, (label, color) in enumerate(TEAM_LEGEND_COLORS):
            y = y0 + pad + i * row_h
            cv2.rectangle(frame, (x0 + 8, y), (x0 + 8 + swatch, y + swatch), color, -1)
            cv2.rectangle(frame, (x0 + 8, y), (x0 + 8 + swatch, y + swatch), (0, 0, 0), 1)
            text = f"{label} {int(counts.get(label, 0))}"
            tx, ty = x0 + 8 + swatch + 8, y + swatch - 2
            # Same 1px outline trick as the ID labels (a thick pass would
            # widen the glyph advance and ghost the text).
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cv2.putText(
                    frame, text, (tx + dx, ty + dy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 1, cv2.LINE_AA,
                )
            cv2.putText(
                frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
        return frame

    def draw_team_connections(
        self,
        frame: np.ndarray,
        player_centers: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        max_distance: int = 150,
    ) -> np.ndarray:
        """
        Draw lines between nearby teammates to illustrate structural connections / mesh.

        Args:
            frame: Video frame image.
            player_centers: List of (x, y) player screen positions.
            color: Line color.
            max_distance: Maximum pixel distance to draw a connecting edge.

        Returns:
            np.ndarray: Annotated frame.
        """
        overlay = frame.copy()
        n = len(player_centers)
        for i in range(n):
            for j in range(i + 1, n):
                pt1 = player_centers[i]
                pt2 = player_centers[j]
                dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
                if dist <= max_distance:
                    cv2.line(overlay, pt1, pt2, color, 1, cv2.LINE_AA)

        # Blend overlay with original frame for subtle transparency
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        return frame

    def draw_radar_minimap(
        self,
        frame: np.ndarray,
        players: Sequence[Tuple[PitchCoordinate, Tuple[int, int, int], Optional[int]]],
        ball_coord: Optional[PitchCoordinate] = None,
    ) -> np.ndarray:
        """
        Draw the top-down pitch radar (Stage 4) in the bottom-left corner.

        The *original* video annotations stay untouched - the radar is added
        as an extra panel, so both views are preserved in one output.

        Orientation matches the camera view: x=0 (goal line) on the left and
        y=0 (near sideline) at the *bottom* of the panel, just like the video
        (the camera sits on the near side). The pitch axes themselves are
        unchanged - only the rendering flips the vertical axis.

        Args:
            frame: Video frame image.
            players: Sequence of ``(pitch_coordinate, bgr_color, track_id)``
                for every player visible this frame (``track_id`` may be None
                to skip the label).
            ball_coord: Optional ball position in pitch meters.

        Returns:
            np.ndarray: Frame with the embedded top-down radar.
        """
        radar = draw_pitch(self.pitch_length, self.pitch_width,
                           (self.radar_width, self.radar_height))
        # draw_pitch puts y=0 at the top; flip so y=0 (near sideline) sits at
        # the bottom of the panel - matching what the camera sees. The
        # standard markings are symmetric, so this only anchors the frame.
        radar = cv2.flip(radar, 0)
        scale_x = self.radar_width / float(self.pitch_length)
        scale_y = self.radar_height / float(self.pitch_width)
        dot_r = max(3, self.radar_height // 55)

        def to_px(coord: PitchCoordinate) -> Tuple[int, int]:
            # Clamped so wildly-bad projections still render inside the panel.
            x = min(max(coord.x, 0.0), self.pitch_length)
            y = min(max(coord.y, 0.0), self.pitch_width)
            # y grows toward the FAR side; the panel mirrors the camera, so
            # near (y=0) maps to the bottom row.
            return int(round(x * scale_x)), int(round((self.pitch_width - y) * scale_y))

        for coord, color, track_id in players:
            px, py = to_px(coord)
            cv2.circle(radar, (px, py), dot_r, color, -1, cv2.LINE_AA)
            cv2.circle(radar, (px, py), dot_r, (0, 0, 0), 1, cv2.LINE_AA)
            if track_id is not None:
                cv2.putText(
                    radar, str(track_id), (px + dot_r + 1, py - dot_r // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA,
                )

        if ball_coord is not None:
            bx, by = to_px(ball_coord)
            cv2.circle(radar, (bx, by), max(2, dot_r - 1), COLOR_BALL, -1, cv2.LINE_AA)
            cv2.circle(radar, (bx, by), max(2, dot_r - 1), (0, 0, 0), 1, cv2.LINE_AA)

        # Keep the panel readable on any frame size (small test clips too).
        margin = 8
        avail_w, avail_h = frame.shape[1] - 2 * margin, frame.shape[0] - 2 * margin
        if radar.shape[1] > avail_w or radar.shape[0] > avail_h:
            fit = min(avail_w / radar.shape[1], avail_h / radar.shape[0])
            radar = cv2.resize(
                radar,
                (max(1, int(radar.shape[1] * fit)), max(1, int(radar.shape[0] * fit))),
                interpolation=cv2.INTER_AREA,
            )

        h, w = radar.shape[:2]
        x0, y0 = margin, frame.shape[0] - h - margin
        frame[y0:y0 + h, x0:x0 + w] = radar
        cv2.rectangle(frame, (x0, y0), (x0 + w - 1, y0 + h - 1), (230, 230, 230), 1)
        return frame
