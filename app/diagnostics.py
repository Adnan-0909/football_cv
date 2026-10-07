"""
Run Diagnostics
===============

Per-frame detection / tracking / team bookkeeping for a pipeline run.

The purpose is *attribution*: when players are missing from the output or a
team label looks wrong, these numbers say whether the failure belongs to
detection, tracking or team classification - instead of guessing from the
rendered video.

Pure bookkeeping: recording never influences detection, tracking or
classification, and every metric is derived from what those stages already
produced:

* detection: candidates per frame, the display-grade subset (>= the configured
             confidence threshold, i.e. what runs with a display-only
             detector counted as "the" detections), frames with none at all
* tracking:  creations (unique IDs), lifetimes, gaps between sightings,
             losses (gone before the final frame)
* teams:     label timeline per track, commits, team-label switches per track

The pipeline calls :meth:`RunDiagnostics.record` once per frame after Stage 3
has refreshed ``team_by_track``, then stores :meth:`RunDiagnostics.summary`
into ``PipelineStats`` at the end of the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Dict, List, Optional, Sequence, Tuple

from app.detector import Detection
from app.tracker import TrackedObject

# Committed team label of a track: TEAM_A (0), TEAM_B (1) or None (UNKNOWN).
Label = Optional[int]


def _stats(values: Sequence[int]) -> Dict[str, float]:
    """mean / median / max of a non-empty sample (zeros for an empty one)."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0}
    return {"mean": float(mean(values)), "median": float(median(values)), "max": int(max(values))}


@dataclass
class TrackRecord:
    """Lifecycle and team-label history of one persistent track ID."""

    track_id: int
    first_seen: int
    last_seen: int
    frames_active: int = 0
    # Longest run of consecutive frames the track was absent between two
    # sightings (>0 means ByteTrack bridged a detection gap for this ID).
    max_gap: int = 0
    # (frame, label) recorded on the first sighting and whenever the label
    # *changes* - a compact timeline of the track's committed team.
    label_changes: List[Tuple[int, Label]] = field(default_factory=list)

    @property
    def span(self) -> int:
        """Frames from first to last sighting (inclusive)."""
        return self.last_seen - self.first_seen + 1

    @property
    def final_label(self) -> Label:
        """Committed label at the track's last sighting."""
        return self.label_changes[-1][1] if self.label_changes else None

    def label_events(self) -> Tuple[int, int, int]:
        """
        ``(commits, team_switches, withdrawals)`` over the timeline.

        * commit:     UNKNOWN -> TEAM_A/B (the 5-frame confirmation fired)
        * switch:     TEAM_A <-> TEAM_B (the 8-frame contradiction fired)
        * withdrawal: TEAM_A/B -> UNKNOWN (kept for completeness; the
                      hysteresis normally never un-commits a team)
        """
        commits = switches = withdrawals = 0
        for (_, prev), (_, cur) in zip(self.label_changes, self.label_changes[1:]):
            if prev is None and cur is not None:
                commits += 1
            elif prev is not None and cur is None:
                withdrawals += 1
            elif prev is not None and cur is not None and prev != cur:
                switches += 1
        return commits, switches, withdrawals


class RunDiagnostics:
    """
    Collects per-frame detection / tracking / team statistics for one run.

    One :meth:`record` call per frame; :meth:`summary` / :meth:`text` are
    read-only aggregations to be called after the run finished.

    Args:
        display_threshold: Confidence at/above which a candidate counts as
            *display-grade*. Kept as a parameter so runs with a lower
            detection floor (candidates fed to the tracker) remain directly
            comparable to runs that only ever emitted display detections.
    """

    def __init__(self, display_threshold: float = 0.35) -> None:
        self.display_threshold = float(display_threshold)
        self.frames = 0
        self.last_frame_index: Optional[int] = None
        self.detections_per_frame: List[int] = []
        self.player_detections_per_frame: List[int] = []
        self.display_detections_per_frame: List[int] = []
        self.display_player_detections_per_frame: List[int] = []
        self.active_players_per_frame: List[int] = []
        self.tracks: Dict[int, TrackRecord] = {}

    # ------------------------------------------------------------------ #
    # Collection
    # ------------------------------------------------------------------ #

    def record(
        self,
        frame_index: int,
        detections: Sequence[Detection],
        tracks: Sequence[TrackedObject],
        team_by_track: Dict[int, Optional[int]],
    ) -> None:
        """
        Register one processed frame.

        Args:
            frame_index: Source frame index.
            detections: Raw Stage 1 detections of this frame.
            tracks: Active tracks of this frame (Stage 2 output).
            team_by_track: Committed team labels (Stage 3 output), keyed by
                track ID; only the IDs active this frame are read.
        """
        self.frames += 1
        self.last_frame_index = frame_index
        self.detections_per_frame.append(len(detections))
        self.player_detections_per_frame.append(
            sum(1 for d in detections if d.class_name == "player")
        )
        self.display_detections_per_frame.append(
            sum(1 for d in detections if d.confidence >= self.display_threshold)
        )
        self.display_player_detections_per_frame.append(
            sum(
                1
                for d in detections
                if d.class_name == "player"
                and d.confidence >= self.display_threshold
            )
        )

        players = [t for t in tracks if t.class_name == "player"]
        self.active_players_per_frame.append(len(players))

        for track in players:
            label = team_by_track.get(track.track_id)
            record = self.tracks.get(track.track_id)
            if record is None:
                self.tracks[track.track_id] = TrackRecord(
                    track_id=track.track_id,
                    first_seen=frame_index,
                    last_seen=frame_index,
                    frames_active=1,
                    label_changes=[(frame_index, label)],
                )
                continue
            gap = frame_index - record.last_seen - 1
            if gap > record.max_gap:
                record.max_gap = gap
            record.last_seen = frame_index
            record.frames_active += 1
            if not record.label_changes or record.label_changes[-1][1] != label:
                record.label_changes.append((frame_index, label))

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        """JSON-serializable aggregate of everything collected."""
        records = list(self.tracks.values())
        final_labels = [r.final_label for r in records]

        commits_total = switches_total = withdrawals_total = 0
        switches_per_track: Dict[int, int] = {}
        stable_per_label = {"TEAM_A": 0, "TEAM_B": 0}
        for record in records:
            commits, switches, withdrawals = record.label_events()
            commits_total += commits
            switches_total += switches
            withdrawals_total += withdrawals
            if switches:
                switches_per_track[record.track_id] = switches
            # "stable": final label is a team and the track never flip-flopped
            # between teams (UNKNOWN -> team commits are normal, not churn).
            if record.final_label is not None and switches == 0 and withdrawals == 0:
                key = "TEAM_A" if record.final_label == 0 else "TEAM_B"
                stable_per_label[key] += 1

        durations = [r.frames_active for r in records]
        losses = sum(1 for r in records if self.last_frame_index is not None
                     and r.last_seen < self.last_frame_index)

        return {
            "frames": self.frames,
            "detections": {
                **_stats(self.detections_per_frame),
                "player_mean": float(mean(self.player_detections_per_frame))
                if self.player_detections_per_frame else 0.0,
                "zero_frames": sum(1 for n in self.detections_per_frame if n == 0),
                "zero_player_frames": sum(
                    1 for n in self.player_detections_per_frame if n == 0
                ),
                # Same statistics restricted to display-grade candidates -
                # directly comparable across runs with different detector floors.
                "display_threshold": self.display_threshold,
                "display_mean": float(mean(self.display_detections_per_frame))
                if self.display_detections_per_frame else 0.0,
                "display_max": int(max(self.display_detections_per_frame))
                if self.display_detections_per_frame else 0,
                "display_player_mean": float(mean(self.display_player_detections_per_frame))
                if self.display_player_detections_per_frame else 0.0,
                "display_zero_frames": sum(
                    1 for n in self.display_detections_per_frame if n == 0
                ),
            },
            "active_tracks": _stats(self.active_players_per_frame),
            "tracks": {
                "unique": len(records),
                "creations": len(records),
                "losses": losses,
                "mean_duration": float(mean(durations)) if durations else 0.0,
                "median_duration": float(median(durations)) if durations else 0.0,
                "max_duration": int(max(durations)) if durations else 0,
                "short_tracks": sum(1 for d in durations if d < 15),
                "bridged_gaps": sum(1 for r in records if r.max_gap > 0),
                "max_gap": int(max((r.max_gap for r in records), default=0)),
            },
            "teams": {
                "final_TEAM_A": sum(1 for l in final_labels if l == 0),
                "final_TEAM_B": sum(1 for l in final_labels if l == 1),
                "final_UNKNOWN": sum(1 for l in final_labels if l is None),
                "stable_TEAM_A": stable_per_label["TEAM_A"],
                "stable_TEAM_B": stable_per_label["TEAM_B"],
                "commits_total": commits_total,
                "ab_switches_total": switches_total,
                "withdrawals_total": withdrawals_total,
                "tracks_with_switch": len(switches_per_track),
                "switches_per_track": dict(
                    sorted(switches_per_track.items(), key=lambda kv: (-kv[1], kv[0]))
                ),
            },
        }

    def text(self) -> str:
        """Human-readable report (one metric per line) for the run log."""
        s = self.summary()
        det, tr, tm = s["detections"], s["tracks"], s["teams"]
        act = s["active_tracks"]
        lines = [
            f"frames processed              : {s['frames']}",
            f"detections / frame            : mean {det['mean']:.1f} "
            f"(median {det['median']:.0f}, max {det['max']}) - "
            f"players mean {det['player_mean']:.1f}",
            f"display-grade >= {det['display_threshold']:.2f}     : mean "
            f"{det['display_mean']:.1f} (max {det['display_max']}), "
            f"{det['display_zero_frames']} frames without one",
            f"frames with 0 detections      : {det['zero_frames']} "
            f"({det['zero_player_frames']} with 0 player detections)",
            f"active players / frame        : mean {act['mean']:.1f}, max {act['max']}",
            f"tracks                        : {tr['unique']} created, {tr['losses']} lost "
            f"before the final frame",
            f"track duration                : mean {tr['mean_duration']:.1f} frames "
            f"(median {tr['median_duration']:.0f}, max {tr['max_duration']}), "
            f"{tr['short_tracks']} shorter than 15 frames",
            f"gap bridging                  : {tr['bridged_gaps']} tracks resumed after a gap "
            f"(longest {tr['max_gap']} frames)",
            f"final labels                  : TEAM_A {tm['final_TEAM_A']}, "
            f"TEAM_B {tm['final_TEAM_B']}, UNKNOWN {tm['final_UNKNOWN']}",
            f"stable (no team flip-flops)   : TEAM_A {tm['stable_TEAM_A']}, "
            f"TEAM_B {tm['stable_TEAM_B']}",
            f"team-label events             : {tm['commits_total']} commits, "
            f"{tm['ab_switches_total']} A<->B switches, "
            f"{tm['withdrawals_total']} withdrawals "
            f"({tm['tracks_with_switch']} tracks ever switched)",
        ]
        per_track = tm["switches_per_track"]
        if per_track:
            shown = ", ".join(f"#{tid}x{n}" for tid, n in list(per_track.items())[:12])
            more = "" if len(per_track) <= 12 else f", +{len(per_track) - 12} more"
            lines.append(f"switches per track            : {shown}{more}")
        return "\n".join(lines)
