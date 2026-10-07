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
from app.detector import Detection, FootballDetector, inference_settings

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


# ---------------------------------------------------------------------- #
# Inference settings (resolution + candidate floor)
# ---------------------------------------------------------------------- #

def test_imgsz_follows_the_source_width_by_default():
    """Auto mode: native width (multiple of 32), never below 640."""
    config = ModelConfig(imgsz=0, confidence_threshold=0.35, candidate_threshold=0.10)
    assert inference_settings(config, 1280)[0] == 1280
    assert inference_settings(config, 1920)[0] == 1920
    assert inference_settings(config, 1282)[0] == 1280  # rounded down to /32
    assert inference_settings(config, 320)[0] == 640    # floor
    assert inference_settings(config, 0)[0] == 640      # unknown width


def test_explicit_imgsz_overrides_auto_mode():
    config = ModelConfig(imgsz=960)
    assert inference_settings(config, 1280)[0] == 960


def test_candidate_floor_never_undercuts_the_display_gate():
    """Detections for tracking go down to candidate_threshold; the display
    gate (confidence_threshold) is always <= the floor actually used."""
    display, candidates = 0.35, 0.10
    config = ModelConfig(confidence_threshold=display, candidate_threshold=candidates)
    imgsz, floor = inference_settings(config, 1280)
    assert floor == pytest.approx(candidates)
    assert floor <= display
    # A candidate setting ABOVE the display gate collapses to the display gate.
    high = ModelConfig(confidence_threshold=display, candidate_threshold=0.50)
    assert inference_settings(high, 1280)[1] == pytest.approx(display)


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
