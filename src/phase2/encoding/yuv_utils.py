"""YUV 4:2:0 I/O utilities.

Provides reading/writing of raw YUV 4:2:0 planar files and
conversion to/from numpy arrays and PNG frames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Tuple

import numpy as np

logger = logging.getLogger("phase2.encoding.yuv_utils")


def yuv420_frame_size(width: int, height: int) -> int:
    """Compute raw byte size of one YUV 4:2:0 frame."""
    return width * height * 3 // 2


def read_yuv420_frame(
    f,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one YUV 4:2:0 frame from an open file handle.

    Returns:
        Tuple of (Y, U, V) as uint8 numpy arrays.
        Y is (height, width), U and V are (height//2, width//2).
    """
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    y_data = f.read(y_size)
    u_data = f.read(uv_size)
    v_data = f.read(uv_size)

    if len(y_data) < y_size:
        raise EOFError("Unexpected end of YUV file")

    Y = np.frombuffer(y_data, dtype=np.uint8).reshape(height, width)
    U = np.frombuffer(u_data, dtype=np.uint8).reshape(height // 2, width // 2)
    V = np.frombuffer(v_data, dtype=np.uint8).reshape(height // 2, width // 2)

    return Y, U, V


def iter_yuv420_frames(
    yuv_path: str,
    width: int,
    height: int,
    n_frames: int = 0,
) -> Generator[Tuple[int, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """Iterate over frames in a YUV 4:2:0 file.

    Yields:
        (frame_idx, Y, U, V) tuples.
    """
    frame_size = yuv420_frame_size(width, height)
    path = Path(yuv_path).expanduser()

    with open(path, "rb") as f:
        idx = 0
        while True:
            if n_frames > 0 and idx >= n_frames:
                break
            try:
                Y, U, V = read_yuv420_frame(f, width, height)
                yield idx, Y, U, V
                idx += 1
            except EOFError:
                break


def yuv420_to_bgr(
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """Convert YUV 4:2:0 to BGR (OpenCV format).

    Uses BT.601 conversion matrix.
    """
    import cv2

    height, width = Y.shape
    U_up = cv2.resize(U, (width, height), interpolation=cv2.INTER_LINEAR)
    V_up = cv2.resize(V, (width, height), interpolation=cv2.INTER_LINEAR)

    yuv = np.stack([Y, U_up, V_up], axis=2)
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return bgr


def save_frames_from_yuv(
    yuv_path: str,
    output_dir: str,
    width: int,
    height: int,
    n_frames: int = 0,
    original_width: int = 0,
    original_height: int = 0,
) -> int:
    """Extract frames from YUV file and save as PNG.

    If original_width/height are specified and differ from width/height,
    crops the padded region to restore original dimensions.

    Returns:
        Number of frames saved.
    """
    import cv2

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx, Y, U, V in iter_yuv420_frames(yuv_path, width, height, n_frames):
        bgr = yuv420_to_bgr(Y, U, V)

        if original_height > 0 and original_width > 0:
            if original_height != height or original_width != width:
                bgr = bgr[:original_height, :original_width]

        frame_path = out_dir / f"{idx:06d}.png"
        cv2.imwrite(str(frame_path), bgr)
        count += 1

    return count
