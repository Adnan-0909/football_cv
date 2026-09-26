"""
Team Classification Module (Stage 3)
====================================

Assigns every tracked player to one of::

    TEAM_A (0) | TEAM_B (1) | UNKNOWN (None)

using **jersey colour statistics only** - no deep-learning classifier, no fixed
RGB values, and no hardcoded "blue = Team A". The two teams are discovered from
the footage itself:

1. **Torso crop.** For every player bbox the upper-body region is cropped
   (``torso_crop_top`` .. ``torso_crop_bottom`` of the box height, with the
   outer columns trimmed by ``torso_crop_side``), so grass, legs and most of
   the advertisement boards inside the box are excluded.
2. **HSV dominant colour.** The crop is converted to HSV and reduced to one
   dominant-colour feature: a small pixel-level k-means keeps the largest
   coherent region (the jersey) and represents it as::

       [S*cos(H), S*sin(H), value_weight * V]

   Hue is encoded circularly so 0/360 degrees are neighbours, saturation scales
   the chroma so washed-out white/grey kits collapse towards the origin instead
   of picking an arbitrary hue, and V keeps white vs black kits apart.
   Crops that are too small (far-away players) or whose pixels do not form one
   dominant region (heavy occlusion, ads / background dominating the box) are
   rejected as *unreliable* and simply produce no observation.
3. **Two-team clustering.** A k-means with ``n_teams`` clusters runs over the
   pooled features of *all* players. It is refitted periodically
   (``recluster_interval``) and each refit is re-anchored to the previous one,
   so the meaning of TEAM_A / TEAM_B never flips mid-run.
4. **Temporal smoothing.** Each track keeps a rolling feature window. A team is
   committed only after ``min_consistent_frames`` consecutive agreements and is
   only switched after ``switch_frames`` consecutive contradicting ones, so an
   assignment never flickers frame to frame. While a committed player is
   temporarily uncertain (occlusion, shadow, blur) the last assignment is kept.
5. **UNKNOWN when uncertain.** A player whose mean colour is too far from both
   cluster centres (referee, goalkeeper, odd kit, advertisement background) or
   almost equally close to both (similar kits) reports UNKNOWN. Players with no
   reliable observation yet are UNKNOWN as well.

Edge cases handled: referees and goalkeepers (distinct colour -> far from both
centres -> UNKNOWN), partially hidden and far-away players (dominant-colour
gate -> no observation), shadows (temporal smoothing over the rolling window),
similar kits (margin gate -> UNKNOWN) and advertisement background inside the
bbox (torso crop + dominant-region gate).

Modularity
----------
The stage is consumed through a single method so a future deep-learning team
model can be dropped in without touching the pipeline::

    teams = TeamClassifier(config).predict_teams(frame, tracks, frame_index)
    # or, for one player:
    team  = TeamClassifier(config).predict_team(frame, track)

Both return ``TEAM_A`` / ``TEAM_B`` / ``UNKNOWN``.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from itertools import permutations
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.config import TeamClassifierConfig
from app.tracker import TrackedObject

logger = logging.getLogger(__name__)

# Median distance -> sigma for a Gaussian-ish spread. The radius of a team
# cluster is a *robust* statistic so that a handful of absorbed outliers
# (referee, goalkeeper, background-heavy box) cannot inflate it.
_MAD_TO_SIGMA = 1.4826

# Public team identifiers. TEAM_A / TEAM_B are *relative* labels: which
# real-world kit becomes A and which becomes B is decided by clustering the
# actual video, never by a fixed colour rule.
TEAM_A: int = 0
TEAM_B: int = 1
TEAM_UNKNOWN: Optional[int] = None

TEAM_LABELS: Dict[Optional[int], str] = {
    TEAM_A: "TEAM_A",
    TEAM_B: "TEAM_B",
    TEAM_UNKNOWN: "UNKNOWN",
}


def team_label(team: Optional[int]) -> str:
    """Textual label (``TEAM_A`` / ``TEAM_B`` / ``UNKNOWN``) for a team id."""
    return TEAM_LABELS.get(team, "UNKNOWN")


# ---------------------------------------------------------------------- #
# Small dependency-free k-means (NumPy only)
# ---------------------------------------------------------------------- #

def _sq_dists(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Squared euclidean distances between (n, d) points and (m, d) centres."""
    d2 = (
        np.sum(points * points, axis=1)[:, None]
        - 2.0 * (points @ centers.T)
        + np.sum(centers * centers, axis=1)[None, :]
    )
    return np.maximum(d2, 0.0)


def _kmeanspp_init(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: spread the initial centres over the data mass."""
    n = data.shape[0]
    centers = np.empty((k, data.shape[1]), dtype=data.dtype)
    centers[0] = data[int(rng.integers(n))]
    closest = _sq_dists(data, centers[0:1]).ravel()
    for j in range(1, k):
        total = float(closest.sum())
        if total <= 1e-12:  # every point identical to the first centre
            idx = int(rng.integers(n))
        else:
            idx = int(rng.choice(n, p=closest / total))
        centers[j] = data[idx]
        closest = np.minimum(closest, _sq_dists(data, centers[j : j + 1]).ravel())
    return centers


def _kmeans(
    data: np.ndarray,
    k: int,
    *,
    iters: int,
    restarts: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Lloyd's algorithm with k-means++ init and several restarts.

    Returns ``(centers, labels, inertia)`` of the best restart. Robust to
    duplicate points (empty clusters keep their previous centre) and to
    ``k`` larger than the number of distinct points.
    """
    n = data.shape[0]
    k = max(1, min(int(k), n))
    best: Optional[Tuple[np.ndarray, np.ndarray, float]] = None

    for _ in range(max(1, int(restarts))):
        centers = _kmeanspp_init(data, k, rng)
        labels = np.full(n, -1, dtype=np.int64)
        for _ in range(max(1, int(iters))):
            new_labels = np.argmin(_sq_dists(data, centers), axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centers[j] = data[mask].mean(axis=0)

        labels = np.argmin(_sq_dists(data, centers), axis=1)
        dists = _sq_dists(data, centers)
        inertia = float(dists[np.arange(n), labels].sum())
        if best is None or inertia < best[2]:
            best = (centers.copy(), labels.copy(), inertia)

    assert best is not None  # loop always runs at least once
    return best


# ---------------------------------------------------------------------- #
# Per-track bookkeeping
# ---------------------------------------------------------------------- #

@dataclass
class _TrackState:
    """Rolling colour history and temporal smoothing state for one player."""

    features: Deque[np.ndarray]
    team: Optional[int] = None        # committed assignment
    candidate: Optional[int] = None   # assignment currently being confirmed
    streak: int = 0                   # consecutive frames backing `candidate`


@dataclass
class _TeamModel:
    """Result of one clustering round: one centre + spread per team."""

    centroids: np.ndarray   # (n_teams, 3) feature-space centres
    radii: np.ndarray       # (n_teams,) robust radius of each cluster
    separation: float       # distance between the two closest centres

    @property
    def n_teams(self) -> int:
        return int(self.centroids.shape[0])


class TeamClassifier:
    """
    Distinguishes Team A, Team B and UNKNOWN from jersey colour statistics.

    See the module docstring for the algorithm. The class is intentionally
    self-contained: the pipeline only calls :meth:`predict_teams`.
    """

    def __init__(self, config: Optional[TeamClassifierConfig] = None) -> None:
        """
        Args:
            config: Colour-clustering / crop / smoothing settings.
        """
        self.config = config or TeamClassifierConfig()
        if self.config.n_teams != 2:
            logger.warning(
                "team_classifier.n_teams=%d is not 2 - team labels stay "
                "positional (0=TEAM_A, 1=TEAM_B, ...).", self.config.n_teams,
            )
        self._states: Dict[int, _TrackState] = {}
        self._model: Optional[_TeamModel] = None
        self._rng = np.random.default_rng(self.config.random_seed)
        self._attempts = 0            # clustering rounds started
        self._last_fit_index = -10**9  # frame index of the last attempt
        logger.info(
            "Initialized TeamClassifier (teams=%d, torso crop %.2f-%.2f, "
            "commit=%d frames, switch=%d frames)",
            self.config.n_teams,
            self.config.torso_crop_top,
            self.config.torso_crop_bottom,
            self.config.min_consistent_frames,
            self.config.switch_frames,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict_teams(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackedObject],
        frame_index: int = 0,
    ) -> Dict[int, Optional[int]]:
        """
        Classify every player of one frame (the pipeline's per-frame entry point).

        Ingests the frame's jersey observations, refits the two-team colour
        model when enough data has accumulated, then returns the temporally
        smoothed assignment of each track.

        Args:
            frame: Current video frame (BGR).
            tracks: Player tracks visible in this frame.
            frame_index: Source frame index (drives the refit cadence).

        Returns:
            Mapping ``track_id -> TEAM_A | TEAM_B | UNKNOWN``.
        """
        for track in tracks:
            self._observe(track, frame)
        self._maybe_fit(frame_index)
        return {track.track_id: self._classify(self._state(track.track_id)) for track in tracks}

    def predict_team(self, frame: np.ndarray, tracked_player: TrackedObject) -> Optional[int]:
        """
        Classify a single tracked player against the current colour model.

        Consumes the model as-is (no refit); the pipeline normally goes through
        :meth:`predict_teams`, which also refreshes the model each frame.

        Args:
            frame: Current video frame (BGR).
            tracked_player: Tracked player with bounding box.

        Returns:
            ``TEAM_A`` (0), ``TEAM_B`` (1) or ``UNKNOWN`` (None).
        """
        self._observe(tracked_player, frame)
        return self._classify(self._state(tracked_player.track_id))

    def fit(self, player_crops: Sequence[np.ndarray]) -> None:
        """
        Fit the team colour model directly from pre-extracted jersey crops.

        Provided for offline / experimental use; the live pipeline builds the
        same model incrementally via :meth:`predict_teams`.

        Args:
            player_crops: Cropped jersey images (BGR), one per observation.
        """
        features = [
            feature
            for crop in player_crops
            if (feature := self._dominant_feature(np.asarray(crop))) is not None
        ]
        if len(features) < 2:
            logger.warning("fit(): only %d usable crops - model not built.", len(features))
            return
        self._publish(self._fit_model(np.stack(features, axis=0)))

    @property
    def model_ready(self) -> bool:
        """True once enough footage has been seen to separate two teams."""
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Feature extraction (steps 1-2)
    # ------------------------------------------------------------------ #

    def extract_jersey_color(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """
        Dominant-colour feature of the player's upper body.

        Args:
            frame: Full video frame (BGR).
            bbox: Player bounding box (x1, y1, x2, y2).

        Returns:
            3-D feature vector, or ``None`` when the crop is too small or not
            dominated by a single colour (unreliable observation).
        """
        crop = self._torso_crop(frame, bbox)
        if crop is None:
            return None
        return self._dominant_feature(crop)

    def _torso_crop(
        self,
        frame: np.ndarray,
        bbox: Sequence[float],
    ) -> Optional[np.ndarray]:
        """Upper-body region of a bounding box (excludes grass/legs/edges)."""
        x1, y1, x2, y2 = (float(v) for v in bbox)
        width = x2 - x1
        height = y2 - y1
        if width <= 1 or height <= 1 or frame is None or frame.size == 0:
            return None

        top = int(round(y1 + self.config.torso_crop_top * height))
        bottom = int(round(y1 + self.config.torso_crop_bottom * height))
        left = int(round(x1 + self.config.torso_crop_side * width))
        right = int(round(x2 - self.config.torso_crop_side * width))

        # Slicing clips automatically; guard against fully-empty / off-screen boxes.
        crop = frame[top:bottom, left:right]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return None
        return np.ascontiguousarray(crop)

    def _hsv_features(self, hsv: np.ndarray) -> np.ndarray:
        """Map an HSV image to the (N, 3) circular-hue feature space."""
        # OpenCV stores 8-bit hue as degrees/2 in [0, 180), so x2 to get the
        # full [0, 360) circle before encoding it circularly.
        hue = hsv[..., 0].astype(np.float32) * (2.0 * np.pi / 180.0)
        sat = hsv[..., 1].astype(np.float32) / 255.0
        val = hsv[..., 2].astype(np.float32) / 255.0
        features = np.empty(hue.shape + (3,), dtype=np.float32)
        features[..., 0] = sat * np.cos(hue)
        features[..., 1] = sat * np.sin(hue)
        features[..., 2] = val * self.config.value_weight
        return features.reshape(-1, 3)

    def _dominant_feature(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Reduce a BGR crop to its dominant (jersey) colour feature.

        Pixel-level k-means with ``n_dominant_colors`` clusters keeps the
        largest coherent region; a crop that is not dominated by one region is
        rejected, which is what keeps advertisement boards, occluders and
        background inside the box from hijacking the jersey colour.
        """
        if image is None or image.ndim != 3 or image.shape[0] == 0 or image.shape[1] == 0:
            return None
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        except cv2.error:  # pragma: no cover - defensive (odd crop shapes)
            return None

        features = self._hsv_features(hsv)
        n = features.shape[0]
        if n < self.config.min_crop_pixels:
            return None  # too far away / too small to trust

        max_pixels = self.config.max_crop_pixels
        if n > max_pixels:
            choice = self._rng.choice(n, size=max_pixels, replace=False)
            features = features[choice]
            n = max_pixels

        # Uniform crop (solid jersey): it *is* the dominant region.
        if float(features.std(axis=0).max()) < 1e-6:
            return features.mean(axis=0)

        centers, labels, _ = _kmeans(
            features,
            self.config.n_dominant_colors,
            iters=self.config.pixel_kmeans_iters,
            restarts=1,
            rng=self._rng,
        )
        counts = np.bincount(labels, minlength=centers.shape[0])
        dominant = int(np.argmax(counts))
        if counts[dominant] / float(n) < self.config.min_dominant_ratio:
            return None  # mixed crop: background dominates or kit is occluded
        return centers[dominant]

    # ------------------------------------------------------------------ #
    # Two-team clustering (steps 3-4)
    # ------------------------------------------------------------------ #

    def _state(self, track_id: int) -> _TrackState:
        state = self._states.get(track_id)
        if state is None:
            state = _TrackState(features=deque(maxlen=self.config.feature_history))
            self._states[track_id] = state
        return state

    def _observe(self, track: TrackedObject, frame: np.ndarray) -> None:
        """Extract the jersey feature of one player and append it to their window."""
        feature = self.extract_jersey_color(frame, track.bbox)
        if feature is not None:
            self._state(track.track_id).features.append(feature)

    def _collect_fit_samples(self) -> Tuple[np.ndarray, int]:
        """Pool recent per-track features (capped) for the clustering round."""
        chunks: List[np.ndarray] = []
        tracks_used = 0
        cap = self.config.max_samples_per_track
        for state in self._states.values():
            if not state.features:
                continue
            recent = list(state.features)[-cap:]
            chunks.append(np.stack(recent, axis=0))
            tracks_used += 1
        if not chunks:
            return np.empty((0, 3), dtype=np.float32), 0

        samples = np.concatenate(chunks, axis=0)
        if samples.shape[0] > self.config.max_cluster_samples:
            idx = self._rng.choice(samples.shape[0], size=self.config.max_cluster_samples, replace=False)
            samples = samples[idx]
        return samples, tracks_used

    def _maybe_fit(self, frame_index: int) -> None:
        """Refit (or first-fit) the colour model when due and enough data exists."""
        # First attempt runs as soon as the samples suffice; every later attempt
        # (successful or not - e.g. kits that only become separable once both
        # teams show up) waits for the refit interval.
        if self._attempts > 0 and (frame_index - self._last_fit_index) < self.config.recluster_interval:
            return

        samples, n_tracks = self._collect_fit_samples()
        if (
            samples.shape[0] < self.config.min_cluster_samples
            or n_tracks < self.config.min_cluster_tracks
        ):
            return  # not enough evidence yet - everyone stays UNKNOWN

        self._attempts += 1
        self._last_fit_index = frame_index
        self._publish(self._fit_model(samples))

    def _fit_model(self, samples: np.ndarray) -> Optional[_TeamModel]:
        """Cluster all observations into ``n_teams`` groups (None if not separable)."""
        centers, labels, _ = _kmeans(
            samples,
            self.config.n_teams,
            iters=self.config.kmeans_iters,
            restarts=self.config.kmeans_restarts,
            rng=self._rng,
        )
        if centers.shape[0] < 2:
            return None

        radii = np.empty(centers.shape[0], dtype=np.float64)
        for j in range(centers.shape[0]):
            members = samples[labels == j]
            if members.shape[0] >= 2:
                diff = members - centers[j]
                dists = np.sqrt(np.sum(diff * diff, axis=1))
                # Robust spread: the median is barely moved by outliers, so a
                # referee/keeper accidentally grouped with a team cannot widen
                # its own acceptance radius.
                radii[j] = float(np.median(dists) * _MAD_TO_SIGMA)
            else:
                radii[j] = 0.0

        pairwise = [
            float(np.linalg.norm(centers[i] - centers[j]))
            for i in range(centers.shape[0])
            for j in range(i + 1, centers.shape[0])
        ]
        separation = min(pairwise)
        mean_radius = float(radii.mean())

        # One blob instead of two kits (very similar jerseys): refuse to label
        # rather than split players arbitrarily.
        if separation < 1e-6 or separation < self.config.min_separation_ratio * max(mean_radius, 1e-6):
            return None

        # Floor the per-cluster radius so an extremely tight cluster cannot
        # reject perfectly normal observations through float noise.
        radii = np.maximum(radii, self.config.radius_floor_ratio * separation)
        return _TeamModel(
            centroids=centers.astype(np.float32),
            radii=radii.astype(np.float32),
            separation=separation,
        )

    def _publish(self, model: Optional[_TeamModel]) -> None:
        """Adopt a fresh model, keeping TEAM_A / TEAM_B meanings stable."""
        if model is None:
            return  # indistinguishable kits: keep the previous model (or none)
        if self._model is not None:
            model = self._align(model, self._model)
        self._model = model

    @staticmethod
    def _align(new: _TeamModel, old: _TeamModel) -> _TeamModel:
        """Reorder fresh centroids so they best match the previous round."""
        k = new.n_teams
        if old.n_teams != k or k > 6:  # k! permutations - guard against blowup
            return new
        best_perm: Optional[Tuple[int, ...]] = None
        best_cost = float("inf")
        for perm in permutations(range(k)):
            cost = sum(
                float(np.linalg.norm(new.centroids[perm[i]] - old.centroids[i]))
                for i in range(k)
            )
            if cost < best_cost:
                best_cost, best_perm = cost, perm
        assert best_perm is not None
        order = list(best_perm)
        return _TeamModel(
            centroids=new.centroids[order],
            radii=new.radii[order],
            separation=new.separation,
        )

    # ------------------------------------------------------------------ #
    # Assignment + temporal smoothing (steps 5-6)
    # ------------------------------------------------------------------ #

    def _raw_label(self, state: _TrackState) -> Optional[int]:
        """
        Nearest-team label from the current model, or ``None`` when uncertain.

        Uncertainty rules (both relative to the *data*, never to fixed colours):

        * too far from every cluster centre -> referee / goalkeeper / odd kit /
          advertisement colour,
        * almost equally close to two clusters -> similar kits / ambiguity.
        """
        if self._model is None or not state.features:
            return None
        window = list(state.features)[-self.config.mean_window :]
        mean = np.stack(window, axis=0).mean(axis=0)

        dists = np.linalg.norm(self._model.centroids - mean[None, :], axis=1)
        nearest = int(np.argmin(dists))
        d1 = float(dists[nearest])
        others = np.delete(dists, nearest)
        d2 = float(others.min()) if others.size else float("inf")

        if d1 > self.config.unknown_radius_scale * float(self._model.radii[nearest]):
            return None
        if d1 > self.config.unknown_margin_ratio * d2:
            return None
        return nearest

    def _classify(self, state: _TrackState) -> Optional[int]:
        """Apply the commit / switch hysteresis on top of the raw label."""
        raw = self._raw_label(state)

        if raw is None:
            # Uncertain right now: a committed team is kept (temporal stability
            # through occlusion / shadow); everything else stays UNKNOWN.
            return state.team

        if state.team is not None and raw == state.team:
            state.candidate = None
            state.streak = 0
            return state.team

        # raw disagrees with the committed team (or nothing is committed yet).
        if raw == state.candidate:
            state.streak += 1
        else:
            state.candidate = raw
            state.streak = 1

        needed = (
            self.config.min_consistent_frames
            if state.team is None
            else self.config.switch_frames
        )
        if state.streak >= needed:
            state.team = state.candidate
            state.candidate = None
            state.streak = 0
        return state.team

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Forget every track, observation and fitted model."""
        self._states.clear()
        self._model = None
        self._attempts = 0
        self._last_fit_index = -10**9
        self._rng = np.random.default_rng(self.config.random_seed)

    @property
    def observed_tracks(self) -> int:
        """Number of players with at least one reliable observation."""
        return sum(1 for state in self._states.values() if state.features)

    def __repr__(self) -> str:
        return (
            f"TeamClassifier(teams={self.config.n_teams}, model_ready={self.model_ready}, "
            f"tracks={len(self._states)})"
        )
