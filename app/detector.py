"""
Object Detection Module
======================

Detects football players, and the ball, using an Ultralytics YOLO model.
Detections are normalized into the project-wide :class:`Detection` structure
so downstream stages (tracking, team classification) never touch Ultralytics
types directly.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """
    Standardized detection data structure.

    Attributes:
        bbox: Bounding box coordinates in (x1, y1, x2, y2) format.
        confidence: Detection confidence score between 0.0 and 1.0.
        class_id: Numerical class ID (e.g. 0 for player, 32 for sports ball).
        class_name: Human-readable class name ("player", "referee", "ball").
    """
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


def inference_settings(config: ModelConfig, frame_width: int) -> Tuple[int, float]:
    """
    ``(imgsz, confidence floor)`` used to run inference for one frame.

    Two separations of responsibility live here:

    * **imgsz** - ``config.imgsz`` when set (> 0), otherwise the source frame
      width rounded down to a multiple of 32, never below 640. Ultralytics'
      implicit default of 640 would downscale 1280-wide broadcast footage
      before inference, which measurably loses distant players (evidence:
      person detections at conf >= 0.35 roughly double at native width).
    * **floor** - the lower of the display gate and ``candidate_threshold``:
      detections between the two thresholds exist only as *candidates* for the
      tracker (ByteTrack's stage-2 rescue works on detections down to
      ``track_low_thresh``), while drawing/reporting stays gated at
      ``confidence_threshold``. Using ``min`` guarantees the display gate can
      never be undercut by a higher candidate setting.

    Args:
        config: Model settings.
        frame_width: Source frame width in pixels (0 when unknown).

    Returns:
        Tuple of (imgsz, confidence floor).
    """
    if frame_width and frame_width > 0:
        imgsz = max(640, (int(frame_width) // 32) * 32)
    else:
        imgsz = 640
    if config.imgsz and config.imgsz > 0:
        imgsz = int(config.imgsz)
    floor = min(config.confidence_threshold, config.candidate_threshold)
    return imgsz, float(floor)


class FootballDetector:
    """
    Wrapper for Ultralytics YOLO model for football scene detection.

    Designed to detect:
    - Players (outfield players and goalkeepers)
    - Referees
    - Football / Ball

    The detector deliberately works with a low *candidate* floor (see
    :func:`inference_settings`) so the tracker receives the weak detections
    ByteTrack's low-score rescue stage is designed for; consumers that draw
    or report detections filter at ``confidence_threshold`` themselves.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        """
        Initialize the detector with model configuration.

        Args:
            config: Model configuration settings (weights path, confidence threshold, device).
        """
        self.config = config or ModelConfig()
        self.model = None
        logger.info("Initialized FootballDetector with weights: %s", self.config.weights_path)

    @property
    def is_loaded(self) -> bool:
        """Return True when the YOLO weights are loaded in memory."""
        return self.model is not None

    def load_model(self) -> None:
        """
        Load YOLO weights onto the configured hardware device (CPU / CUDA).

        If the configured weights file does not exist yet, Ultralytics downloads
        it automatically (requires network access) into the configured path.
        """
        from ultralytics import YOLO  # Verified imported in main.py

        weights = Path(self.config.weights_path)
        if not weights.exists():
            logger.warning(
                "Weights not found at %s - requesting automatic download of '%s' (network required).",
                weights,
                weights.name,
            )
        logger.info("Loading YOLO model from %s on device %s...", weights, self.config.device)
        try:
            self.model = YOLO(str(weights))
        except Exception as exc:  # pragma: no cover - depends on network / local file state
            raise RuntimeError(
                f"Failed to load YOLO weights from '{weights}'. "
                f"Verify the path in config.yaml or allow the automatic download to complete. Original error: {exc}"
            ) from exc
        logger.info("YOLO model ready (weights=%s, device=%s)", weights.name, self.config.device)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect players, referees, and ball in a single image frame.

        Args:
            frame: BGR image from OpenCV (numpy array).

        Returns:
            List[Detection]: List of standardized detections.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        if self.model is None:
            self.load_model()

        imgsz, floor = inference_settings(self.config, frame.shape[1])
        results = self.model.predict(
            source=frame,
            conf=floor,
            imgsz=imgsz,
            iou=self.config.iou_threshold,
            device=self.config.device,
            classes=sorted({self.config.player_class_id, self.config.ball_class_id}),
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names or []))

            for (x1, y1, x2, y2), confidence, class_id in zip(xyxy, confidences, class_ids):
                class_id = int(class_id)
                detections.append(
                    Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(confidence),
                        class_id=class_id,
                        class_name=self._resolve_class_name(class_id, names),
                    )
                )

        logger.debug("Frame detection produced %d object(s)", len(detections))
        return detections

    def _resolve_class_name(self, class_id: int, names: Dict[int, str]) -> str:
        """Map a COCO class ID to the project's vocabulary (player / ball / ...)."""
        if class_id == self.config.player_class_id:
            return "player"
        if class_id == self.config.ball_class_id:
            return "ball"
        return str(names.get(class_id, class_id))
