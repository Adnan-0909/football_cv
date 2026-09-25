"""
Unit Tests for Architecture Module Imports
==========================================
Verifies that all foundational modules in app/ can be imported cleanly.
"""

def test_app_init():
    import app
    assert hasattr(app, "__version__")


def test_config_module():
    from app.config import AppConfig, load_config, setup_logging
    assert callable(load_config)
    assert callable(setup_logging)


def test_video_module():
    from app.video import VideoReader, VideoWriter
    assert VideoReader is not None
    assert VideoWriter is not None


def test_detector_module():
    from app.detector import FootballDetector, Detection
    detector = FootballDetector()
    assert detector is not None


def test_tracker_module():
    from app.tracker import PlayerTracker, TrackedObject
    tracker = PlayerTracker()
    assert tracker is not None


def test_team_classifier_module():
    from app.team_classifier import TeamClassifier
    classifier = TeamClassifier()
    assert classifier is not None


def test_pitch_module():
    from app.pitch import PitchTransformer, PitchCoordinate
    pitch = PitchTransformer()
    assert pitch is not None


def test_formation_module():
    from app.formation import FormationAnalyzer, TacticalMetrics
    analyzer = FormationAnalyzer()
    assert analyzer is not None


def test_passing_module():
    from app.passing import PassingAnalyzer, PassingLane
    analyzer = PassingAnalyzer()
    assert analyzer is not None


def test_visualization_module():
    from app.visualization import TacticalVisualizer
    visualizer = TacticalVisualizer()
    assert visualizer is not None
