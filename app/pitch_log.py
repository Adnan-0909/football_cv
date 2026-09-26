"""
Per-Frame Pitch Records (Stage 4)
=================================

Stores the **top-down pitch position (meters)** of every tracked player in
every processed frame and exports it as CSV.

Every tracked player observed in a frame produces one :class:`PitchRecord`::

    {
        "frame": 100,
        "timestamp": 4.04,
        "player_id": 7,
        "team": "TEAM_A",
        "pitch_x": 63.2,     # meters along the pitch length (0 -> 105)
        "pitch_y": 41.8,     # meters along the pitch width  (0 -> 68)
    }

Positions come from :func:`app.pitch.foot_position` (bottom-centre of the
bounding box) projected through the manually calibrated homography - see
:mod:`app.pitch` for the coordinate-system convention.

``PitchLog.save_csv()`` writes the columns::

    frame,timestamp,player_id,team,pitch_x,pitch_y

``frame`` indexes the *source* video and ``timestamp`` is its position in
seconds, so pitch rows line up exactly with the Stage 1/2/3 CSVs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from app.team_classifier import team_label

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.pitch import PitchCoordinate
    from app.tracker import TrackedObject

# Column order of the exported CSV - part of the output contract.
CSV_FIELDS: Tuple[str, ...] = (
    "frame",
    "timestamp",
    "player_id",
    "team",
    "pitch_x",
    "pitch_y",
)


@dataclass(frozen=True, slots=True)
class PitchRecord:
    """One tracked player's pitch position in one frame (meters)."""

    frame: int
    timestamp: float
    player_id: int
    team: str
    pitch_x: float
    pitch_y: float

    @classmethod
    def from_track(
        cls,
        frame: int,
        timestamp: float,
        track: "TrackedObject",
        team: Optional[int],
        coordinate: "PitchCoordinate",
    ) -> "PitchRecord":
        """
        Build a record from a tracked player and its projected position.

        Args:
            frame: Source frame index.
            timestamp: Source timestamp in seconds.
            track: The tracked player.
            team: Raw team id (``0`` / ``1`` / ``None``).
            coordinate: Projected pitch position (meters).
        """
        return cls(
            frame=int(frame),
            timestamp=float(timestamp),
            player_id=int(track.track_id),
            team=team_label(team),
            pitch_x=float(coordinate.x),
            pitch_y=float(coordinate.y),
        )

    @property
    def position(self) -> Tuple[float, float]:
        """(pitch_x, pitch_y) in meters."""
        return (self.pitch_x, self.pitch_y)

    def to_dict(self) -> Dict[str, object]:
        """Plain-dict view matching the schema documented in the module docstring."""
        return {
            "frame": self.frame,
            "timestamp": self.timestamp,
            "player_id": self.player_id,
            "team": self.team,
            "pitch_x": self.pitch_x,
            "pitch_y": self.pitch_y,
        }

    def to_row(self) -> Tuple[object, ...]:
        """Flat CSV row aligned with :data:`CSV_FIELDS`."""
        return (
            self.frame,
            self.timestamp,
            self.player_id,
            self.team,
            self.pitch_x,
            self.pitch_y,
        )


class PitchLog:
    """
    Ordered collection of :class:`PitchRecord` objects for a whole run.

    Populated by the pipeline only when a pitch calibration is loaded
    (``python main.py --calibrate``); exported with ``--pitch-csv``.
    """

    CSV_FIELDS = CSV_FIELDS

    def __init__(self, records: Optional[Sequence[PitchRecord]] = None) -> None:
        self._records: List[PitchRecord] = list(records) if records else []

    # ------------------------------------------------------------------ #
    # Collection
    # ------------------------------------------------------------------ #

    def add(self, record: PitchRecord) -> None:
        """Append a single record."""
        self._records.append(record)

    def add_pitch(
        self,
        frame: int,
        timestamp: float,
        track: "TrackedObject",
        team: Optional[int],
        coordinate: "PitchCoordinate",
    ) -> PitchRecord:
        """Record a tracked player's pitch position for this frame."""
        record = PitchRecord.from_track(frame, timestamp, track, team, coordinate)
        self._records.append(record)
        return record

    def extend(self, records: Sequence[PitchRecord]) -> None:
        """Append several records at once."""
        self._records.extend(records)

    def clear(self) -> None:
        """Drop every record."""
        self._records.clear()

    # ------------------------------------------------------------------ #
    # Access
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[PitchRecord]:
        return iter(self._records)

    @property
    def records(self) -> List[PitchRecord]:
        """Snapshot copy of every record in chronological order."""
        return list(self._records)

    @property
    def unique_ids(self) -> set:
        """All player IDs seen during the run."""
        return {r.player_id for r in self._records}

    @property
    def frames(self) -> set:
        """Source frame indices that produced at least one record."""
        return {r.frame for r in self._records}

    def for_frame(self, frame: int) -> List[PitchRecord]:
        """All records belonging to one source frame."""
        return [r for r in self._records if r.frame == frame]

    def to_dicts(self) -> List[Dict[str, object]]:
        """Every record as a plain dict."""
        return [r.to_dict() for r in self._records]

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def save_csv(self, path: Union[str, Path]) -> Path:
        """
        Write the log as CSV and return the path written.

        Missing parent directories are created. The header row is always
        :data:`CSV_FIELDS`.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_FIELDS)
            for record in self._records:
                writer.writerow(record.to_row())
        return destination
