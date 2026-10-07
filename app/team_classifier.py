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
3. **Two-team clustering.** A k-means with ``n_teams`` clusters runs over one
   *vote* per player - the median of their rolling feature window - so a
   single contaminated frame cannot drag a colour and a lone referee only
   ever votes once. A fit whose smaller cluster looks like an outlier group
   (referee, grass-dominated players) is pruned and retried, and the last
   gate-passing structure wins, so the model ends up as ``{kit A} | {kit B}``
   rather than ``{odd colours} | {everyone merged}``. The model is refitted
   periodically (``recluster_interval``) and each refit is re-anchored to the
   previous one, so the meaning of TEAM_A / TEAM_B never flips mid-run.
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
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from app.config import TeamClassifierConfig
from app.tracker import TrackedObject

logger = logging.getLogger(__name__)

# Median distance -> sigma for a Gaussian-ish spread. The radius of a team
# cluster is a *robust* statistic so that a handful of absorbed outliers
# (referee, goalkeeper, background-heavy box) cannot inflate it.
_MAD_TO_SIGMA = 1.4826

# How many times a degenerate fit may prune its outlier cluster and retry
# before being refused. Two (= three chain rounds total) was the sweet spot
# on this footage: one prune usually turns "{odd colours} | {merged kits}"
# into "{kit A} | {kit B}", a second covers double-outlier pools, and a third
# only starts sub-splitting single kits into garbage candidates.
_MAX_PRUNE_ROUNDS = 2

# While the smaller cluster of a fit holds less than this share of the votes
# it is treated as a suspect outlier group and pruned; from 40% upwards both
# sides are substantial enough to be real teams (they can never both exceed
# 50%, so this still stops on any balanced split).
_STOP_SHARE = 0.4

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


def torso_roi(
    bbox: Sequence[float],
    config: TeamClassifierConfig,
) -> Tuple[int, int, int, int]:
    """
    Pixel rectangle ``(left, top, right, bottom)`` of the jersey ROI.

    Pure geometry shared by :meth:`TeamClassifier._torso_crop` (the pixels the
    classifier actually reads) and the ``--debug`` overlay, so the rectangle
    drawn on screen is exactly the region being classified: the upper/middle
    torso, with the head band, arms-outside edges, shorts/legs and most
    grass/background trimmed away.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    width = x2 - x1
    height = y2 - y1
    top = int(round(y1 + config.torso_crop_top * height))
    bottom = int(round(y1 + config.torso_crop_bottom * height))
    left = int(round(x1 + config.torso_crop_side * width))
    right = int(round(x2 - config.torso_crop_side * width))
    return left, top, right, bottom


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
        self._anchor_model: Optional[_TeamModel] = None
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
        # Only players visible *now* vote in the next fit: states of retired
        # tracks are kept for their label history, but their months-old colours
        # must never pollute a refit (after a camera cut the pool would
        # otherwise be dominated by players who left the pitch ages ago).
        self._maybe_fit(frame_index, {track.track_id for track in tracks})
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
        if x2 - x1 <= 1 or y2 - y1 <= 1 or frame is None or frame.size == 0:
            return None

        left, top, right, bottom = torso_roi(bbox, self.config)

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

    def _collect_fit_samples(self, live_ids: Set[int]) -> Tuple[np.ndarray, np.ndarray, int]:
        """Pool recent per-track features (capped) plus one "vote" per track.

        Only tracks in ``live_ids`` (currently visible players) contribute:
        states of retired tracks are kept for their label history, but their
        stale colours must not enter a refit - after a camera cut the pool
        would otherwise be dominated by players long gone from the pitch.

        The pooled features only prove that a fit may be attempted; the
        clustering itself runs on the votes - the per-track *median* feature -
        so a single background-contaminated frame (grass, crowd, advertisement
        boards) cannot drag a player's colour, and a lone referee only ever
        contributes one vote instead of a whole cluster's worth of samples.
        Tracks with fewer than ``min_vote_features`` observations abstain: the
        first boxes of a new track usually sit on grass, so their median is
        background, not jersey.
        """
        chunks: List[np.ndarray] = []
        votes: List[np.ndarray] = []
        tracks_used = 0
        cap = self.config.max_samples_per_track
        for track_id, state in self._states.items():
            if track_id not in live_ids or len(state.features) < self.config.min_vote_features:
                continue
            recent = list(state.features)[-cap:]
            chunk = np.stack(recent, axis=0)
            chunks.append(chunk)
            votes.append(np.median(chunk, axis=0))
            tracks_used += 1
        if not chunks:
            empty = np.empty((0, 3), dtype=np.float32)
            return empty, empty.copy(), 0

        samples = np.concatenate(chunks, axis=0)
        if samples.shape[0] > self.config.max_cluster_samples:
            idx = self._rng.choice(samples.shape[0], size=self.config.max_cluster_samples, replace=False)
            samples = samples[idx]
        return samples, np.stack(votes, axis=0), tracks_used

    def _maybe_fit(self, frame_index: int, live_ids: Set[int]) -> None:
        """Refit (or first-fit) the colour model when due and enough data exists."""
        # First fit retries every frame until a valid two-team model is established;
        # once established, reclustering obeys the recluster interval cooldown.
        if self._model is not None and (frame_index - self._last_fit_index) < self.config.recluster_interval:
            return

        samples, votes, n_tracks = self._collect_fit_samples(live_ids)
        if (
            samples.shape[0] < self.config.min_cluster_samples
            or n_tracks < self.config.min_cluster_tracks
        ):
            return  # not enough evidence yet - everyone stays UNKNOWN

        self._attempts += 1
        self._last_fit_index = frame_index
        self._publish(self._fit_model(votes))

    def _fit_model(self, samples: np.ndarray) -> Optional[_TeamModel]:
        """Cluster the observations into ``n_teams`` groups (None if not separable).

        k-means runs on the full vote set first; while its smaller cluster
        holds less than ``_STOP_SHARE`` of the votes, that cluster is treated
        as a suspect outlier group (a lone referee or keeper, grass-dominated
        players - on wide shots they form one far-away cluster while both real
        kits merge into the other), it is pruned and the fit retried. Every
        structure that passes the separation gates - spread ratio *and*
        absolute centre distance - is remembered and the **last** one wins:
        the chain ends on ``{kit A} | {kit B}`` as soon as both sides are
        balanced, whereas pruning a genuine small side makes the refit on the
        remaining votes fail the gates, leaving that side's structure as the
        last valid candidate. If no candidate ever passes the gates, or the
        final one holds less than ``min_minority_share``, the fit is refused
        and the previous model (or UNKNOWN) stands.
        """
        work = samples
        best: Optional[Tuple[float, _TeamModel]] = None  # (minority share, model)
        for _ in range(1 + _MAX_PRUNE_ROUNDS):
            centers, labels, _ = _kmeans(
                work,
                self.config.n_teams,
                iters=self.config.kmeans_iters,
                restarts=self.config.kmeans_restarts,
                rng=self._rng,
            )
            if centers.shape[0] < 2:
                break
            counts = np.bincount(labels, minlength=centers.shape[0])
            minority_share = float(counts.min()) / max(int(counts.sum()), 1)

            radii = np.empty(centers.shape[0], dtype=np.float64)
            for j in range(centers.shape[0]):
                members = work[labels == j]
                if members.shape[0] >= 2:
                    diff = members - centers[j]
                    dists = np.sqrt(np.sum(diff * diff, axis=1))
                    # Robust spread: the median is barely moved by outliers,
                    # so a referee/keeper accidentally grouped with a team
                    # cannot widen its own acceptance radius.
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

            # One blob instead of two kits (very similar jerseys) never
            # qualifies as a candidate - rather than split players arbitrarily.
            # The ratio alone is not enough: a sub-split of a SINGLE kit is
            # tight (tiny radii) and clears it easily, yet labelling bright vs
            # dim white as two teams is exactly the failure we are fighting.
            # The absolute floor rejects near-identical pairs instead, while
            # genuine kit pairs on this footage sit far above it.
            if separation >= self.config.min_separation_distance and (
                separation >= self.config.min_separation_ratio * max(mean_radius, 1e-6)
            ):
                # Radius floor: an extremely tight cluster must not reject
                # perfectly normal observations through float noise.
                model = _TeamModel(
                    centroids=centers.astype(np.float32),
                    radii=np.maximum(radii, self.config.radius_floor_ratio * separation).astype(np.float32),
                    separation=separation,
                )
                best = (minority_share, model)  # last gate-passing structure

            if minority_share >= _STOP_SHARE:
                break  # both sides substantial: a real split
            keep = labels == int(np.argmax(counts))  # drop the suspect minority
            if bool(np.all(keep)) or int(np.sum(keep)) < 4:
                break  # nothing left to prune, or too few votes for two clusters
            work = work[keep]

        if best is None or best[0] < self.config.min_minority_share:
            return None  # no valid structure, or a degenerate leftover
        return best[1]

    def _publish(self, model: Optional[_TeamModel]) -> None:
        """Adopt a fresh model, keeping TEAM_A / TEAM_B meanings stable."""
        if model is None:
            return  # indistinguishable kits: keep the previous model (or none)
        if self._anchor_model is None:
            self._anchor_model = model
        else:
            model = self._align(model, self._anchor_model)
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

    def _window_mean(self, state: _TrackState) -> Optional[np.ndarray]:
        """Mean feature of the track's most recent observations (or None)."""
        if not state.features:
            return None
        window = list(state.features)[-self.config.mean_window :]
        return np.stack(window, axis=0).mean(axis=0)

    def _raw_label(self, state: _TrackState) -> Optional[int]:
        """
        Nearest-team label from the current model, or ``None`` when uncertain.

        Uncertainty rules (both relative to the *data*, never to fixed colours):

        * too far from every cluster centre -> referee / goalkeeper / odd kit /
          advertisement colour,
        * almost equally close to two clusters -> similar kits / ambiguity.
        """
        if self._model is None:
            return None
        mean = self._window_mean(state)
        if mean is None:
            return None

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

    def team_confidence(self, track_id: int) -> float:
        """
        Model support (0..1) for this track's current team evidence.

        Display/introspection helper for the ``--debug`` overlay. It combines
        the two data-driven gates :meth:`_raw_label` applies, rescaled to
        [0, 1]:

        * margin - how much closer the track's mean feature is to its
          reference centre than to the best rival (0 = ambiguous, 1 = far
          clear of the other team),
        * radius - how far inside the UNKNOWN radius the evidence sits
          (0 = at/outside the outlier gate, 1 = dead centre).

        The returned value is the smaller of both, so ``conf > 0`` exactly
        when the observation would pass the gates, and the reference centre
        is the *committed* team when one exists (confidence in the label the
        video shows), otherwise the nearest centre. Hysteresis streaks are
        deliberately not part of this number - it measures evidence, not
        bookkeeping.

        Args:
            track_id: Persistent track ID.

        Returns:
            Confidence in [0, 1]; 0.0 when no model exists yet or the track
            has no reliable observations.
        """
        state = self._states.get(track_id)
        if self._model is None or state is None:
            return 0.0
        mean = self._window_mean(state)
        if mean is None:
            return 0.0

        dists = np.linalg.norm(self._model.centroids - mean[None, :], axis=1)
        ref = state.team if state.team is not None else int(np.argmin(dists))
        if ref < 0 or ref >= dists.shape[0]:
            ref = int(np.argmin(dists))
        d_ref = float(dists[ref])
        others = np.delete(dists, ref)
        d_other = float(others.min()) if others.size else float("inf")

        if np.isinf(d_other):
            margin = 1.0  # single-cluster model: no rival to be ambiguous with
        else:
            margin = float(np.clip((d_other - d_ref) / max(d_other, 1e-12), 0.0, 1.0))

        gate = self.config.unknown_radius_scale * float(self._model.radii[ref])
        if gate <= 0.0:
            radius = 1.0 if d_ref <= 0.0 else 0.0
        else:
            radius = float(np.clip(1.0 - d_ref / gate, 0.0, 1.0))
        return float(min(margin, radius))

    def reset(self) -> None:
        """Forget every track, observation and fitted model."""
        self._states.clear()
        self._model = None
        self._anchor_model = None
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
