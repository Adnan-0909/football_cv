"""
Team Classification Module
==========================

Modular skeleton for classifying players into teams based on jersey colors.
Extracts color features from player torso bounding box crops and clusters them
(e.g., using KMeans or Gaussian Mixture Models in HSV/Lab color spaces).
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.config import TeamClassifierConfig
from app.tracker import TrackedObject

logger = logging.getLogger(__name__)


class TeamClassifier:
    """
    Distinguishes between Team 1, Team 2, Goalkeepers, and Match Officials.
    """

    def __init__(self, config: Optional[TeamClassifierConfig] = None) -> None:
        """
        Initialize the team classifier.

        Args:
            config: Configuration settings for color clustering and crop regions.
        """
        self.config = config or TeamClassifierConfig()
        self.team_colors: Dict[int, Tuple[int, int, int]] = {}
        logger.info("Initialized TeamClassifier (target_teams=%d)", self.config.n_teams)

    def extract_jersey_color(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        """
        Crop the player's torso area and extract dominant jersey color.

        Args:
            frame: Full video frame image.
            bbox: Player bounding box (x1, y1, x2, y2).

        Returns:
            np.ndarray: Representative color vector (e.g. RGB or HSV).
        """
        raise NotImplementedError("Color extraction will be implemented in the team classification phase.")

    def fit(self, player_crops: List[np.ndarray]) -> None:
        """
        Cluster jersey colors collected from initial frames to define team clusters.

        Args:
            player_crops: Collection of cropped jersey images from players.
        """
        raise NotImplementedError("Cluster training will be implemented in the team classification phase.")

    def predict_team(self, frame: np.ndarray, tracked_player: TrackedObject) -> int:
        """
        Assign a team ID (e.g., 0 or 1) to a tracked player.

        Args:
            frame: Current video frame.
            tracked_player: Tracked player object with bounding box.

        Returns:
            int: Predicted team identifier (0 for Team A, 1 for Team B, etc.).
        """
        raise NotImplementedError("Team prediction will be implemented in the team classification phase.")
