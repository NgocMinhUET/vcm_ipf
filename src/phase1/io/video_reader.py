"""Video/frame reader with metadata extraction.

Supports both video files and directories of image frames.
Provides an iterator interface for memory-efficient processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from phase1.core.constants import SUPPORTED_VIDEO_EXTENSIONS
from phase1.utils.log import get_logger

logger = get_logger("io.video_reader")


class VideoReader:
    """Lazy video reader wrapping OpenCV VideoCapture.

    Args:
        source: Path to a video file or directory of frame images.
        max_frames: Stop after this many frames (None = read all).
    """

    def __init__(self, source: str | Path, max_frames: Optional[int] = None):
        self.source = Path(source)
        self.max_frames = max_frames
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_paths: Optional[list[Path]] = None
        self._is_video = False

        if self.source.is_file() and self.source.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            self._is_video = True
            self._cap = cv2.VideoCapture(str(self.source))
            if not self._cap.isOpened():
                raise IOError(f"Cannot open video: {self.source}")
        elif self.source.is_dir():
            exts = {".png", ".jpg", ".jpeg", ".bmp"}
            self._frame_paths = sorted(
                [p for p in self.source.iterdir() if p.suffix.lower() in exts]
            )
            if not self._frame_paths:
                raise FileNotFoundError(f"No image frames found in: {self.source}")
        else:
            raise FileNotFoundError(f"Invalid video source: {self.source}")

        logger.info("VideoReader initialized: %s (video=%s)", self.source.name, self._is_video)

    @property
    def total_frames(self) -> int:
        if self._is_video and self._cap is not None:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._frame_paths is not None:
            return len(self._frame_paths)
        return 0

    @property
    def fps(self) -> float:
        if self._is_video and self._cap is not None:
            return float(self._cap.get(cv2.CAP_PROP_FPS))
        return 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        """Returns (width, height)."""
        if self._is_video and self._cap is not None:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        if self._frame_paths:
            img = cv2.imread(str(self._frame_paths[0]))
            if img is not None:
                h, w = img.shape[:2]
                return (w, h)
        return (0, 0)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_index, bgr_frame) tuples."""
        count = 0
        limit = self.max_frames or float("inf")

        if self._is_video and self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while count < limit:
                ret, frame = self._cap.read()
                if not ret:
                    break
                yield count, frame
                count += 1
        elif self._frame_paths is not None:
            for p in self._frame_paths:
                if count >= limit:
                    break
                frame = cv2.imread(str(p))
                if frame is None:
                    logger.warning("Skipping unreadable frame: %s", p.name)
                    continue
                yield count, frame
                count += 1

        logger.info("VideoReader finished: %d frames read", count)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self) -> None:
        self.release()
