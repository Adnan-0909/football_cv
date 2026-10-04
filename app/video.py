"""
Video Processing Module
=======================

Provides modular, easy-to-use abstractions for reading and writing football match videos
using OpenCV (cv2). Supports generator-based frame-by-frame streaming to keep memory usage low.
"""

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """
    Reads video files frame-by-frame using OpenCV.

    Features:
    - Low memory footprint: frames are yielded lazily on-demand.
    - Context manager support (`with VideoReader(...) as reader:`).
    - Configurable frame stride and max frame limit.
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        frame_stride: int = 1,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Initialize the video reader.

        Args:
            video_path: Path to the input video file.
            frame_stride: Process every Nth frame (1 = every frame).
            max_frames: Maximum number of frames to read (None for all frames).
        """
        self.video_path = Path(video_path)
        self.frame_stride = max(1, frame_stride)
        self.max_frames = max_frames
        self._cap: Optional[cv2.VideoCapture] = None

        if not self.video_path.exists():
            logger.warning("Input video does not exist yet at: %s", self.video_path)

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def open(self) -> None:
        """Open the video stream."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found at: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise IOError(f"Could not open video file: {self.video_path}")

        logger.info(
            "Opened video: %s (Resolution: %dx%d, FPS: %.2f, Frames: %d)",
            self.video_path.name,
            self.width,
            self.height,
            self.fps,
            self.total_frames,
        )

    def release(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("Released video capture resource for %s", self.video_path.name)

    @property
    def is_opened(self) -> bool:
        """Return True if the video is currently open."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def width(self) -> int:
        """Video frame width in pixels."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Video frame height in pixels."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        """Frames per second of the video."""
        if self._cap is None:
            return 0.0
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    @property
    def total_frames(self) -> int:
        """Total number of frames in the video."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yield frames from the video.

        Yields:
            Tuple[int, np.ndarray]: (frame_index, frame_image_bgr)
        """
        if not self.is_opened:
            self.open()

        current_frame_idx = 0
        yielded_count = 0

        while self.is_opened:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                break

            if current_frame_idx % self.frame_stride == 0:
                yield current_frame_idx, frame
                yielded_count += 1

                if self.max_frames is not None and yielded_count >= self.max_frames:
                    logger.info("Reached maximum requested frames: %d", self.max_frames)
                    break

            current_frame_idx += 1


class VideoWriter:
    """
    Writes processed frames to a video file using OpenCV.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float,
        width: int,
        height: int,
        fourcc_code: str = "mp4v",
    ) -> None:
        """
        Initialize video writer.

        Args:
            output_path: Target video file path.
            fps: Frame rate for the written video.
            width: Frame width in pixels.
            height: Frame height in pixels.
            fourcc_code: 4-character codec code (default 'mp4v').
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.fourcc_code = fourcc_code
        self._writer: Optional[cv2.VideoWriter] = None

    def __enter__(self) -> "VideoWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def open(self) -> None:
        """Open the video writer."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc_code)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        if not self._writer.isOpened():
            raise IOError(f"Could not open video writer for {self.output_path}")

        logger.info("Opened video writer: %s (%dx%d @ %.2f fps)", self.output_path.name, self.width, self.height, self.fps)

    def isOpened(self) -> bool:
        """Whether the underlying OpenCV writer opened successfully."""
        return self._writer is not None and self._writer.isOpened()

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame."""
        if self._writer is None or not self._writer.isOpened():
            raise RuntimeError("VideoWriter is not open. Call open() or use context manager.")
        self._writer.write(frame)

    def release(self) -> None:
        """Release video writer resources."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Saved output video to: %s", self.output_path)
