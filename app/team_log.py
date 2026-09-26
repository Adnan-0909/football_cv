"""
Per-Frame Team Records
======================

Stores the team assignment of every tracked player in every processed frame and
can export it as CSV (Stage 3 of the pipeline).

Every tracked player observed in a frame produces one :class:`TeamRecord`::

    {
        "frame": 100,
        "timestamp": 4.04,
        "player_id": 7,
        "team": "TEAM_A",
        "center": [cx, cy],
    }

``team`` is one of ``TEAM_A`` / ``TEAM_B`` / ``UNKNOWN`` - the labels produced
by :func:`app.team_classifier.team_label`. Which real-world kit maps to which
label is decided by clustering the footage, never hardcoded.

``TeamLog.save_csv()`` writes the columns::

    frame,timestamp,player_id,team,cx,cy

Like :class:`app.track_log.TrackLog`, ``frame`` indexes the *source* video and
``timestamp`` is its position in seconds, so team rows line up exactly with the
per-frame tracking CSV regardless of ``video.frame_stride``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

from app.team_classifier import team_label

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.tracker import TrackedObject

# Column order of the exported CSV - part of the output contract.
CSV_FIELDS: Tuple[str, ...] = (
    "frame",
    "timestamp",
    "player_id",
    "team",
    "cx",
    "cy",
)

VALID_TEAMS: Tuple[str, ...] = ("TEAM_A", "TEAM_B", "UNKNOWN")


@dataclass(frozen=True, slots=True)
class TeamRecord:
    """One tracked player's team assignment in one frame."""

    frame: int
    timestamp: float
    player_id: int
    team: str
    center: Tuple[float, float]

    def __post_init__(self) -> None:
        if self.team not in VALID_TEAMS:
            raise ValueError(f"team must be one of {VALID_TEAMS}, got {self.team!r}")

    @classmethod
    def from_track(
        cls,
        frame: int,
        timestamp: float,
        track: "TrackedObject",
        team: Optional[int],
    ) -> "TeamRecord":
        """
        Build a record from a :class:`~app.tracker.TrackedObject`.

        Args:
            frame: Source frame index.
            timestamp: Source timestamp in seconds.
            track: The tracked player.
            team: Raw team id (``0`` / ``1`` / ``None``); mapped to its
                textual label via :func:`app.team_classifier.team_label`.
        """
        x1, y1, x2, y2 = (float(v) for v in track.bbox)
        return cls(
            frame=int(frame),
            timestamp=float(timestamp),
            player_id=int(track.track_id),
            team=team_label(team),
            center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        )

    @property
    def cx(self) -> float:
        """Horizontal centre of the bounding box."""
        return self.center[0]

    @property
    def cy(self) -> float:
        """Vertical centre of the bounding box."""
        return self.center[1]

    def to_dict(self) -> Dict[str, object]:
        """Plain-dict view matching the schema documented in the module docstring."""
        return {
            "frame": self.frame,
            "timestamp": self.timestamp,
            "player_id": self.player_id,
            "team": self.team,
            "center": list(self.center),
        }

    def to_row(self) -> Tuple[object, ...]:
        """Flat CSV row aligned with :data:`CSV_FIELDS`."""
        return (
            self.frame,
            self.timestamp,
            self.player_id,
            self.team,
            self.center[0],
            self.center[1],
        )


class TeamLog:
    """
    Ordered collection of :class:`TeamRecord` objects for a whole run.

    Records are appended frame by frame while the pipeline runs; the log keeps
    everything in memory and exports on demand (``--team-csv`` in main.py).
    """

    CSV_FIELDS = CSV_FIELDS

    def __init__(self, records: Optional[Sequence[TeamRecord]] = None) -> None:
        self._records: List[TeamRecord] = list(records) if records else []

    # ------------------------------------------------------------------ #
    # Collection
    # ------------------------------------------------------------------ #

    def add(self, record: TeamRecord) -> None:
        """Append a single record."""
        self._records.append(record)

    def add_team(
        self,
        frame: int,
        timestamp: float,
        track: "TrackedObject",
        team: Optional[int],
    ) -> TeamRecord:
        """Record a tracked player's team for this frame and return the new record."""
        record = TeamRecord.from_track(frame, timestamp, track, team)
        self._records.append(record)
        return record

    def extend(self, records: Sequence[TeamRecord]) -> None:
        """Append several records at once."""
        self._records.extend(records)

    def clear(self) -> None:
        """Drop every record (e.g. before re-using the log for a new video)."""
        self._records.clear()

    # ------------------------------------------------------------------ #
    # Access
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[TeamRecord]:
        return iter(self._records)

    @property
    def records(self) -> List[TeamRecord]:
        """Snapshot copy of every record in chronological order."""
        return list(self._records)

    @property
    def unique_ids(self) -> Set[int]:
        """All player IDs seen during the run."""
        return {r.player_id for r in self._records}

    @property
    def frames(self) -> Set[int]:
        """Source frame indices that produced at least one record."""
        return {r.frame for r in self._records}

    @property
    def team_counts(self) -> Dict[str, int]:
        """Unique players seen with each label over the whole run."""
        per_team: Dict[str, Set[int]] = {label: set() for label in VALID_TEAMS}
        for record in self._records:
            per_team[record.team].add(record.player_id)
        return {label: len(ids) for label, ids in per_team.items()}

    def for_frame(self, frame: int) -> List[TeamRecord]:
        """All records belonging to one source frame."""
        return [r for r in self._records if r.frame == frame]

    def to_dicts(self) -> List[Dict[str, object]]:
        """Every record as a plain dict (handy for JSON export later)."""
        return [r.to_dict() for r in self._records]

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def save_csv(self, path: Union[str, Path]) -> Path:
        """
        Write the log as CSV and return the path written.

        Missing parent directories are created. The header row is always
        written, even for an empty log.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_FIELDS)
            for record in self._records:
                writer.writerow(record.to_row())
        return path

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "TeamLog":
        """Re-read a log previously written by :meth:`save_csv`."""
        path = Path(path)
        records: List[TeamRecord] = []
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(
                    TeamRecord(
                        frame=int(row["frame"]),
                        timestamp=float(row["timestamp"]),
                        player_id=int(row["player_id"]),
                        team=row["team"],
                        center=(float(row["cx"]), float(row["cy"])),
                    )
                )
        return cls(records)

    def __repr__(self) -> str:
        return f"TeamLog(records={len(self._records)}, players={len(self.unique_ids)})"
