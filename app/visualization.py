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
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.pitch import PitchCoordinate
from app.tracker import TrackedObject

logger = logging.getLogger(__name__)


# Standard Tactical Color Palette (BGR for OpenCV)
COLOR_TEAM_A = (255, 60, 0)      # Blueish
COLOR_TEAM_B = (0, 165, 255)     # Orange
COLOR_BALL = (0, 255, 255)       # Yellow
COLOR_REFEREE = (0, 255, 0)      # Green
COLOR_LANE_OPEN = (50, 205, 50)  # Lime Green
COLOR_LANE_BLOCKED = (0, 0, 255) # Red


class TacticalVisualizer:
    """
    Renders visual tactical annotations on video frames.
    """

    def __init__(self, pitch_radar_size: Tuple[int, int] = (300, 200)) -> None:
        """
        Initialize the visualizer.

        Args:
            pitch_radar_size: (width, height) of the mini-map 2D pitch radar.
        """
        self.radar_width, self.radar_height = pitch_radar_size
        logger.info("Initialized TacticalVisualizer (radar_size=%dx%d)", self.radar_width, self.radar_height)

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
        team_a_coords: List[PitchCoordinate],
        team_b_coords: List[PitchCoordinate],
        ball_coord: Optional[PitchCoordinate] = None,
    ) -> np.ndarray:
        """
        Draw a 2D top-down mini pitch radar in the corner of the frame.

        Args:
            frame: Video frame image.
            team_a_coords: Pitch coordinates for Team A.
            team_b_coords: Pitch coordinates for Team B.
            ball_coord: Pitch coordinates for the ball.

        Returns:
            np.ndarray: Frame with embedded mini-map radar.
        """
        # Radar drawing will be fully implemented when homography is connected.
        return frame
