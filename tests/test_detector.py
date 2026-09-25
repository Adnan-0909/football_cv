"""
Unit Tests for the Detection Module
====================================

Model-free tests always run; the inference smoke test only runs when the YOLO
weights are already present locally (no network access during the test suite).
"""

from pathlib import Path

import numpy as np
import pytest

from app.config import PROJECT_ROOT, ModelConfig
from app.detector import Detection, FootballDetector

WEIGHTS_PATH = PROJECT_ROOT / "models" / "yolov8s.pt"


def test_detection_is_a_plain_data_structure():
    detection = Detection(bbox=(1.0, 2.0, 3.0, 4.0), confidence=0.75, class_id=0, class_name="player")
    assert detection.bbox == (1.0, 2.0, 3.0, 4.0)
    assert detection.confidence == 0.75


def test_class_names_map_to_project_vocabulary():
    config = ModelConfig(player_class_id=0, ball_class_id=32)
    detector = FootballDetector(config)
    names = {0: "person", 32: "sports ball", 1: "bicycle"}
    assert detector._resolve_class_name(0, names) == "player"
    assert detector._resolve_class_name(32, names) == "ball"
    assert detector._resolve_class_name(1, names) == "bicycle"
    assert detector._resolve_class_name(7, names) == "7"  # unknown -> raw id


def test_empty_frame_short_circuits_before_loading_the_model():
    detector = FootballDetector()
    assert detector.detect(None) == []
    assert detector.detect(np.empty((0, 0, 3), dtype=np.uint8)) == []
    assert detector.is_loaded is False


def test_detector_initializes_from_config():
    detector = FootballDetector()
    assert detector.is_loaded is False
    assert detector.config.confidence_threshold > 0


@pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason=f"YOLO weights not present at {WEIGHTS_PATH}")
def test_detector_runs_inference_on_a_blank_frame():
    detector = FootballDetector()
    detector.load_model()
    assert detector.is_loaded is True

    # A featureless frame should yield no confident player/ball detections.
    blank = np.full((480, 640, 3), (40, 90, 40), dtype=np.uint8)
    detections = detector.detect(blank)
    assert isinstance(detections, list)
    for detection in detections:
        assert isinstance(detection, Detection)
        assert 0.0 <= detection.confidence <= 1.0
        assert detection.class_name in {"player", "ball"}
