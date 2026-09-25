"""
Per-Frame Track Records
=======================

Stores the tracking result of every processed frame in a flat, machine-readable
form and can export it as CSV.

Every tracked player observed in a frame produces one :class:`TrackRecord`::

    {
        "frame": 100,
        "timestamp": 4.04,
        "player_id": 7,
        "bbox": [x1, y1, x2, y2],
        "center": [cx, cy],
    }

``frame`` is the index of the frame in the *source* video (0-based) and
``timestamp`` is its position in seconds, so records can be lined up with the
original footage regardless of ``video.frame_stride``.

``TrackLog.save_csv()`` writes the columns::

    frame,timestamp,player_id,x1,y1,x2,y2,cx,cy

Only players are recorded: non-player detections (the ball) are excluded from
tracking by ``tracker.player_only`` and never receive a ``player_id``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.tracker import TrackedObject

# Column order of the exported CSV - part of the output contract.
CSV_FIELDS: Tuple[str, ...] = (
    "frame",
    "timestamp",
    "player_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
)


def _center(bbox: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass(frozen=True, slots=True)
class TrackRecord:
    """One tracked player in one frame."""

    frame: int
    timestamp: float
    player_id: int
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]

    @classmethod
    def from_track(
        cls,
        frame: int,
        timestamp: float,
        track: "TrackedObject",
    ) -> "TrackRecord":
        """Build a record from a :class:`~app.tracker.TrackedObject`."""
        bbox = tuple(float(v) for v in track.bbox)
        return cls(
            frame=int(frame),
            timestamp=float(timestamp),
            player_id=int(track.track_id),
            bbox=bbox,  # type: ignore[arg-type]
            center=_center(bbox),
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
            "bbox": list(self.bbox),
            "center": list(self.center),
        }

    def to_row(self) -> Tuple[object, ...]:
        """Flat CSV row aligned with :data:`CSV_FIELDS`."""
        x1, y1, x2, y2 = self.bbox
        return (
            self.frame,
            self.timestamp,
            self.player_id,
            x1,
            y1,
            x2,
            y2,
            self.center[0],
            self.center[1],
        )


class TrackLog:
    """
    Ordered collection of :class:`TrackRecord` objects for a whole run.

    Records are appended frame by frame while the pipeline runs and can be
    queried or exported afterwards. The log keeps everything in memory; for very
    long clips prefer ``--stride`` / ``--max-frames``.
    """

    CSV_FIELDS = CSV_FIELDS

    def __init__(self, records: Optional[Sequence[TrackRecord]] = None) -> None:
        self._records: List[TrackRecord] = list(records) if records else []

    # ------------------------------------------------------------------ #
    # Collection
    # ------------------------------------------------------------------ #

    def add(self, record: TrackRecord) -> None:
        """Append a single record."""
        self._records.append(record)

    def add_track(self, frame: int, timestamp: float, track: "TrackedObject") -> TrackRecord:
        """Record a tracked player for this frame and return the new record."""
        record = TrackRecord.from_track(frame, timestamp, track)
        self._records.append(record)
        return record

    def extend(self, records: Sequence[TrackRecord]) -> None:
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

    def __iter__(self) -> Iterator[TrackRecord]:
        return iter(self._records)

    @property
    def records(self) -> List[TrackRecord]:
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

    def for_frame(self, frame: int) -> List[TrackRecord]:
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
    def from_csv(cls, path: Union[str, Path]) -> "TrackLog":
        """Re-read a log previously written by :meth:`save_csv`."""
        path = Path(path)
        records: List[TrackRecord] = []
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(
                    TrackRecord(
                        frame=int(row["frame"]),
                        timestamp=float(row["timestamp"]),
                        player_id=int(row["player_id"]),
                        bbox=(
                            float(row["x1"]),
                            float(row["y1"]),
                            float(row["x2"]),
                            float(row["y2"]),
                        ),
                        center=(float(row["cx"]), float(row["cy"])),
                    )
                )
        return cls(records)

    def __repr__(self) -> str:
        return f"TrackLog(records={len(self._records)}, players={len(self.unique_ids)})"
