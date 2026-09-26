"""
Unit Tests for Stage 3 - Team Classification
============================================

Covers jersey-colour feature extraction, two-team clustering, the UNKNOWN
gates (referee, goalkeeper, tiny/mixed crops, identical kits) and the temporal
smoothing that keeps a player's assignment stable across frames.

The assertions deliberately never check *which* group becomes TEAM_A: teams
are discovered from the footage, so no RGB value and no "blue = Team A" rule
may leak into the tests either.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytest

from app.config import TeamClassifierConfig
from app.team_classifier import TEAM_A, TEAM_B, TeamClassifier, team_label
from app.tracker import TrackedObject

FRAME_SIZE = (640, 360)  # (width, height)
GRASS = (45, 110, 45)

# Three deliberately different kit pairs (BGR): none of them is one of the
# annotation palette colours, and each pair must work without any tuning.
KIT_PAIRS = {
    "blue_red": ((200, 70, 40), (40, 45, 200)),
    "green_white": ((70, 170, 70), (245, 245, 245)),
    "yellow_magenta": ((40, 210, 230), (200, 40, 190)),
}


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def make_frame() -> np.ndarray:
    return np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), GRASS, dtype=np.uint8)


def draw_patch(frame: np.ndarray, bbox: Sequence[float], color: Sequence[int]) -> None:
    """Fill a bounding box with a solid kit colour."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    frame[y1:y2, x1:x2] = tuple(int(c) for c in color)


def player(track_id: int, bbox: Sequence[float], confidence: float = 0.9) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=tuple(float(v) for v in bbox),
        class_id=0,
        class_name="player",
        confidence=confidence,
    )


def grid_bboxes(n: int, frame_w: int = FRAME_SIZE[0]) -> List[Tuple[float, float, float, float]]:
    """Evenly spread tall boxes across the frame (one per player)."""
    margin, gap = 6, 8
    width = max(24, (frame_w - 2 * margin - (n - 1) * gap) // n)
    boxes = []
    for i in range(n):
        x1 = margin + i * (width + gap)
        boxes.append((float(x1), 60.0, float(x1 + width), 250.0))
    return boxes


def expected_feature(color: Sequence[int], value_weight: float = 0.5) -> np.ndarray:
    """Reference HSV -> [S*cos(H), S*sin(H), weight*V] conversion for one colour."""
    hsv = cv2.cvtColor(np.uint8([[list(color)]]), cv2.COLOR_BGR2HSV)[0, 0]
    hue = float(hsv[0]) * 2.0 * math.pi / 180.0  # OpenCV hue is stored as degrees/2
    sat = float(hsv[1]) / 255.0
    val = float(hsv[2]) / 255.0
    return np.array([sat * math.cos(hue), sat * math.sin(hue), value_weight * val], dtype=np.float32)


def scale(color: Sequence[int], factor: float) -> Tuple[int, int, int]:
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)  # type: ignore[return-value]


@dataclass
class PlayerSpec:
    """A synthetic player: fixed bbox, kit colour, optional per-frame transform."""

    track_id: int
    bbox: Tuple[float, float, float, float]
    color: Tuple[int, int, int]
    transform: Optional[Callable[[int], Tuple[int, int, int]]] = None


def play(
    classifier: TeamClassifier,
    specs: Sequence[PlayerSpec],
    frames: int,
) -> List[Dict[int, Optional[int]]]:
    """Feed static players through ``predict_teams`` and record every frame's labels."""
    history: List[Dict[int, Optional[int]]] = []
    for i in range(frames):
        frame = make_frame()
        tracks = []
        for spec in specs:
            color = spec.transform(i) if spec.transform else spec.color
            draw_patch(frame, spec.bbox, color)
            tracks.append(player(spec.track_id, spec.bbox))
        history.append(classifier.predict_teams(frame, tracks, i))
    return history


def two_team_specs(
    color_a: Tuple[int, int, int],
    color_b: Tuple[int, int, int],
    per_team: int = 6,
    first_id: int = 1,
    extra: Sequence[PlayerSpec] = (),
    width: int = FRAME_SIZE[0],
) -> List[PlayerSpec]:
    """Half the grid wears kit A, half kit B (left/right interleaved)."""
    box = grid_bboxes(2 * per_team + len(extra), frame_w=width)
    specs: List[PlayerSpec] = []
    track_id = first_id
    for i in range(per_team):
        specs.append(PlayerSpec(track_id, box[i], color_a))
        specs.append(PlayerSpec(track_id + per_team, box[per_team + i], color_b))
        track_id += 1
    # Extra players (referee, keeper, ...) get IDs past both team ranges.
    for offset, spec in enumerate(extra):
        specs.append(PlayerSpec(first_id + 2 * per_team + offset, box[2 * per_team + offset], spec.color))
    return specs


def group_of(history: List[Dict[int, Optional[int]]], frame: int, ids: Sequence[int]) -> set:
    return {history[frame][i] for i in ids}


# ---------------------------------------------------------------------- #
# Labels + probe behaviour
# ---------------------------------------------------------------------- #

def test_team_label_mapping():
    assert team_label(TEAM_A) == "TEAM_A"
    assert team_label(TEAM_B) == "TEAM_B"
    assert team_label(None) == "UNKNOWN"
    assert team_label(42) == "UNKNOWN"


def test_probe_style_call_on_a_tiny_frame_returns_unknown():
    """The pipeline probes predict_team() with a 2x2 frame - it must not raise."""
    classifier = TeamClassifier()
    dummy = player(-1, (0.0, 0.0, 1.0, 1.0))
    assert classifier.predict_team(np.zeros((2, 2, 3), dtype=np.uint8), dummy) is None
    assert not classifier.model_ready


# ---------------------------------------------------------------------- #
# Feature extraction (steps 1-2)
# ---------------------------------------------------------------------- #

def test_extract_jersey_color_matches_hsv_feature_math():
    color = (180, 60, 200)
    frame = make_frame()
    bbox = (100.0, 60.0, 160.0, 260.0)
    draw_patch(frame, bbox, color)

    feature = TeamClassifier().extract_jersey_color(frame, bbox)

    assert feature is not None
    np.testing.assert_allclose(feature, expected_feature(color), atol=1e-3)


def test_extract_rejects_tiny_and_invalid_crops():
    classifier = TeamClassifier()
    frame = make_frame()

    # Far-away player: the torso crop has fewer than min_crop_pixels pixels.
    tiny = (300.0, 100.0, 308.0, 110.0)
    draw_patch(frame, tiny, KIT_PAIRS["blue_red"][0])
    assert classifier.extract_jersey_color(frame, tiny) is None

    # Degenerate boxes never raise.
    assert classifier.extract_jersey_color(frame, (50.0, 50.0, 50.0, 50.0)) is None
    assert classifier.extract_jersey_color(frame, (700.0, 400.0, 740.0, 460.0)) is None


def test_extract_rejects_mixed_crop():
    """A crop with no dominant region (ads / crowd / occluders) is unusable."""
    rng = np.random.default_rng(7)
    frame = make_frame()
    # bbox -> torso crop covers rows 100..220, cols 117..197; all of it is noise.
    bbox = (100.0, 40.0, 214.0, 340.0)
    frame[90:230, 110:205] = rng.integers(0, 255, (140, 95, 3), dtype=np.uint8)

    assert TeamClassifier().extract_jersey_color(frame, bbox) is None


def test_extract_keeps_the_dominant_jersey_over_background():
    """Background inside the box is fine as long as the jersey dominates."""
    jersey = KIT_PAIRS["blue_red"][0]
    frame = make_frame()
    bbox = (100.0, 40.0, 160.0, 240.0)

    # Jersey stripe down the middle (~57% of the torso crop), grass on both
    # sides of it inside the crop.
    cv2.rectangle(frame, (110, 60), (150, 220), GRASS, -1)
    cv2.rectangle(frame, (118, 60), (142, 220), jersey, -1)

    feature = TeamClassifier().extract_jersey_color(frame, bbox)
    assert feature is not None, "dominant jersey region should be accepted"
    np.testing.assert_allclose(feature, expected_feature(jersey), atol=0.08)


def test_background_heavy_bbox_ends_up_unknown():
    """
    A box that is mostly grass cannot reveal a kit - and must not steal one.

    The extractor cannot tell "grass" from "jersey" without hardcoding colours,
    so it samples the dominant (background) region; that colour sits far from
    both kits, so the uncertainty gates keep the player at UNKNOWN while the
    properly detected players still form two clean teams.
    """
    team_a, team_b = KIT_PAIRS["blue_red"]
    # Layout limited to the left 480px so the sliver box cannot overlap a
    # properly detected player (that would corrupt *their* colour sample).
    specs = two_team_specs(team_a, team_b, per_team=5, width=480)
    sliver_bbox = (500.0, 100.0, 560.0, 300.0)  # torso crop ~19% jersey

    classifier = TeamClassifier()
    history = []
    for i in range(40):
        frame = make_frame()
        tracks = [player(s.track_id, s.bbox) for s in specs]
        for spec in specs:
            draw_patch(frame, spec.bbox, spec.color)
        # The 11th "player": grass in the box, only a jersey stripe left.
        cv2.rectangle(frame, (505, 105), (555, 295), GRASS, -1)
        cv2.rectangle(frame, (526, 105), (534, 295), team_a, -1)
        tracks.append(player(99, sliver_bbox))
        history.append(classifier.predict_teams(frame, tracks, i))

    final = history[-1]
    assert final[99] is None, "background-dominated player must stay UNKNOWN"
    team_ids = [s.track_id for s in specs]
    assert None not in {final[i] for i in team_ids}, f"good players disturbed: {final}"
    assert len({final[i] for i in team_ids}) == 2, "kits must still form two teams"


# ---------------------------------------------------------------------- #
# Two-team clustering (steps 3-5)
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("kit", sorted(KIT_PAIRS))
def test_two_kits_are_separated_and_stable(kit: str):
    """Both kits get exactly one team each, assigned for good (no flicker)."""
    color_a, color_b = KIT_PAIRS[kit]
    per_team = 6
    specs = two_team_specs(color_a, color_b, per_team=per_team)
    classifier = TeamClassifier()

    history = play(classifier, specs, frames=40)
    final = history[-1]

    # specs are interleaved A, B, A, B, ... -> every other entry per team.
    group_a = [s.track_id for s in specs[0::2]]
    group_b = [s.track_id for s in specs[1::2]]

    labels_a = group_of(history, 39, group_a)
    labels_b = group_of(history, 39, group_b)
    assert len(labels_a) == 1, f"kit A split: {labels_a}"
    assert len(labels_b) == 1, f"kit B split: {labels_b}"
    assert None not in labels_a and None not in labels_b, "everyone should be classified by frame 39"
    assert labels_a != labels_b, "the two kits must form two different teams"

    # Stability: once committed, labels never change again.
    for frame in range(20, 40):
        assert group_of(history, frame, group_a) == labels_a
        assert group_of(history, frame, group_b) == labels_b
    assert all(final[s.track_id] is not None for s in specs)


def test_referee_and_goalkeeper_stay_unknown():
    """A third / fourth colour is not a team: it must not be forced into A or B."""
    team_a, team_b = KIT_PAIRS["blue_red"]
    referee = (70, 190, 60)   # green kit, midway between blue and red in hue
    keeper = (200, 40, 190)   # magenta kit, also equidistant from both teams

    specs = two_team_specs(
        team_a, team_b, per_team=6,
        extra=[PlayerSpec(0, (0, 0, 0, 0), referee), PlayerSpec(0, (0, 0, 0, 0), keeper)],
    )
    classifier = TeamClassifier()
    history = play(classifier, specs, frames=40)
    final = history[-1]

    team_players = [s.track_id for s in specs[:12]]
    outliers = [s.track_id for s in specs[12:]]

    assert len(group_of(history, 39, team_players)) == 2, "kits must form two teams"
    assert all(final[i] in (TEAM_A, TEAM_B) for i in team_players)
    assert all(final[i] is None for i in outliers), f"referee/keeper labelled: {final}"


def test_identical_kits_stay_unknown():
    """One blob instead of two teams -> no model, nobody gets a forced label."""
    only_color = KIT_PAIRS["blue_red"][0]
    specs = [PlayerSpec(i, bbox, only_color) for i, bbox in enumerate(grid_bboxes(8))]
    classifier = TeamClassifier()

    history = play(classifier, specs, frames=30)

    assert all(label is None for frame in history for label in frame.values())


def test_far_away_player_stays_unknown_while_others_are_labelled():
    team_a, team_b = KIT_PAIRS["blue_red"]
    specs = two_team_specs(team_a, team_b, per_team=4, width=480)
    # A tiny player far from the others: never enough pixels for a colour.
    specs.append(PlayerSpec(99, (520.0, 150.0, 528.0, 162.0), team_a))
    classifier = TeamClassifier()

    history = play(classifier, specs, frames=30)
    final = history[-1]

    assert final[99] is None
    assert all(final[s.track_id] in (TEAM_A, TEAM_B) for s in specs[:8])
    assert len({final[s.track_id] for s in specs[:8]}) == 2
    # The tiny player never produced a usable observation.
    assert classifier.observed_tracks == 8


# ---------------------------------------------------------------------- #
# Temporal smoothing (step 6)
# ---------------------------------------------------------------------- #

def test_occluded_frames_do_not_change_a_committed_team():
    """Unreliable crops (occluder, noise) keep the last assignment alive."""
    team_a, team_b = KIT_PAIRS["blue_red"]
    specs = two_team_specs(team_a, team_b, per_team=5)
    victim = specs[0]
    classifier = TeamClassifier()

    history = play(classifier, specs, frames=25)
    committed = history[24][victim.track_id]
    assert committed is not None

    # Frames 25-31: the victim shrinks to an unusable crop (occluded / far).
    occluded = PlayerSpec(victim.track_id, (victim.bbox[0] + 8, 150.0, victim.bbox[0] + 16, 162.0), victim.color)
    specs2 = [occluded] + specs[1:]
    for i in range(25, 32):
        frame = make_frame()
        tracks = []
        for spec in specs2:
            draw_patch(frame, spec.bbox, spec.color)
            tracks.append(player(spec.track_id, spec.bbox))
        history.append(classifier.predict_teams(frame, tracks, i))

    assert history[-1][victim.track_id] == committed, "occlusion must not flip the team"

    # Back in view: still the same team (the occluded frames changed nothing).
    restored = play(classifier, specs, frames=8)
    assert restored[-1][victim.track_id] == committed


def test_a_changed_kit_needs_switch_frames_before_flipping():
    """A short inconsistency is hysteresis noise; only a lasting one switches."""
    team_a, team_b = KIT_PAIRS["blue_red"]
    specs = two_team_specs(team_a, team_b, per_team=5)
    victim = specs[0]
    classifier = TeamClassifier()

    history = play(classifier, specs, frames=25)
    original = history[24][victim.track_id]
    team_b_label = history[24][specs[1].track_id]
    assert original is not None and original != team_b_label

    # The victim genuinely changes kit (player swapped, wrong early fit, ...).
    switched = PlayerSpec(victim.track_id, victim.bbox, team_b)
    specs2 = [switched] + specs[1:]
    for i in range(25, 60):
        frame = make_frame()
        tracks = []
        for spec in specs2:
            draw_patch(frame, spec.bbox, spec.color)
            tracks.append(player(spec.track_id, spec.bbox))
        history.append(classifier.predict_teams(frame, tracks, i))

    # Right after switch_frames' worth of evidence the label must not have
    # changed yet, and once the new colour persists it must converge.
    assert history[25 + 10][victim.track_id] == original, "switched too eagerly"
    assert history[59][victim.track_id] == team_b_label, "never converged to the new kit"


def test_shadows_and_brightness_jitter_do_not_flip_labels():
    """Global lighting changes shift V but must not disturb assignments."""
    team_a, team_b = KIT_PAIRS["blue_red"]

    def brightness(i: int, phase: float = 0.0) -> float:
        # Clouds/shadow sweeping over the pitch (55%-100% brightness).
        return 0.55 + 0.45 * (0.5 + 0.5 * math.sin(i / 4.0 + phase))

    specs = two_team_specs(team_a, team_b, per_team=5)
    specs[0] = PlayerSpec(
        specs[0].track_id, specs[0].bbox, team_a,
        transform=lambda i: scale(team_a, brightness(i, 0.0)),
    )
    specs[1] = PlayerSpec(
        specs[1].track_id, specs[1].bbox, team_b,
        transform=lambda i: scale(team_b, brightness(i, 1.7)),
    )

    classifier = TeamClassifier()
    history = play(classifier, specs, frames=45)

    group_a = [s.track_id for s in specs[0::2]]
    group_b = [s.track_id for s in specs[1::2]]
    labels_a = group_of(history, 44, group_a)
    labels_b = group_of(history, 44, group_b)
    assert len(labels_a) == 1 and len(labels_b) == 1 and labels_a != labels_b
    for frame in range(25, 45):
        assert group_of(history, frame, group_a) == labels_a
        assert group_of(history, frame, group_b) == labels_b


# ---------------------------------------------------------------------- #
# Modularity / introspection
# ---------------------------------------------------------------------- #

def test_fit_from_pre_extracted_crops():
    """The offline fit() entry point builds the same kind of model."""
    color_a, color_b = KIT_PAIRS["blue_red"]
    crops = []
    for color, offset in ((color_a, 0), (color_b, 300)):
        solid = np.full((80, 60, 3), color, dtype=np.uint8)
        crops.extend([solid] * 12)

    classifier = TeamClassifier()
    assert not classifier.model_ready
    classifier.fit(crops)
    assert classifier.model_ready


def test_fit_with_unusable_crops_warns_and_keeps_no_model():
    classifier = TeamClassifier()
    classifier.fit([np.zeros((4, 4, 3), dtype=np.uint8)] * 3)
    assert not classifier.model_ready


def test_reset_clears_everything():
    specs = [PlayerSpec(i, bbox, KIT_PAIRS["blue_red"][i % 2]) for i, bbox in enumerate(grid_bboxes(8))]
    classifier = TeamClassifier()
    play(classifier, specs, frames=15)
    assert classifier.model_ready and classifier.observed_tracks == 8

    classifier.reset()
    assert not classifier.model_ready
    assert classifier.observed_tracks == 0


def test_config_exposes_smoothing_thresholds():
    cfg = TeamClassifierConfig(switch_frames=12, min_consistent_frames=3)
    classifier = TeamClassifier(cfg)
    assert classifier.config.switch_frames == 12
    assert classifier.config.min_consistent_frames == 3
    assert classifier.config.n_teams == 2
