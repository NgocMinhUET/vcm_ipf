"""PSNR computation for full-frame and ROI regions.

Computes:
    - PSNR_full: standard full-frame PSNR (Y channel)
    - PSNR_ROI: PSNR measured only within object bounding box regions
    - PSNR_BG: PSNR measured only in background regions

ROI PSNR is the critical VCM metric: it measures quality where machine
vision tasks operate, isolating the benefit of ROI-aware QP allocation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from phase2.encoding.yuv_utils import iter_yuv420_frames

logger = logging.getLogger("phase2.evaluation.psnr")


@dataclass
class FramePSNR:
    """PSNR metrics for a single frame."""
    frame_idx: int
    psnr_y_full: float
    psnr_y_roi: float
    psnr_y_bg: float
    mse_y_full: float
    mse_y_roi: float
    mse_y_bg: float
    n_roi_pixels: int
    n_bg_pixels: int


@dataclass
class SequencePSNR:
    """Aggregated PSNR metrics for a sequence."""
    n_frames: int
    psnr_y_full_mean: float
    psnr_y_roi_mean: float
    psnr_y_bg_mean: float
    psnr_y_full_std: float
    psnr_y_roi_std: float
    per_frame: List[FramePSNR]


def _mse_to_psnr(mse: float, peak: float = 255.0) -> float:
    """Convert MSE to PSNR in dB."""
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(peak * peak / mse)


def _compute_roi_mask(
    height: int,
    width: int,
    boxes: List[Tuple[float, float, float, float]],
    expansion: float = 0.1,
) -> np.ndarray:
    """Create binary ROI mask from bounding boxes.

    Args:
        height: Frame height.
        width: Frame width.
        boxes: List of (x1, y1, x2, y2) bounding boxes.
        expansion: Fractional expansion of each box.

    Returns:
        Boolean mask of shape (height, width).
    """
    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        bw = x2 - x1
        bh = y2 - y1
        ex = bw * expansion
        ey = bh * expansion
        r1 = max(0, int(y1 - ey))
        r2 = min(height, int(y2 + ey))
        c1 = max(0, int(x1 - ex))
        c2 = min(width, int(x2 + ex))
        mask[r1:r2, c1:c2] = True
    return mask


def compute_frame_psnr(
    orig_y: np.ndarray,
    recon_y: np.ndarray,
    boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    frame_idx: int = 0,
    expansion: float = 0.1,
) -> FramePSNR:
    """Compute PSNR for a single frame with ROI breakdown.

    Args:
        orig_y: Original Y channel (height, width), uint8.
        recon_y: Reconstructed Y channel (height, width), uint8.
        boxes: Optional list of (x1, y1, x2, y2) object bounding boxes.
        frame_idx: Frame index for labeling.
        expansion: Fractional expansion of boxes for ROI mask.
    """
    h, w = orig_y.shape
    orig = orig_y.astype(np.float64)
    recon = recon_y.astype(np.float64)
    diff_sq = (orig - recon) ** 2

    mse_full = float(np.mean(diff_sq))
    psnr_full = _mse_to_psnr(mse_full)

    mse_roi = mse_full
    mse_bg = mse_full
    n_roi = h * w
    n_bg = 0

    if boxes:
        roi_mask = _compute_roi_mask(h, w, boxes, expansion)
        bg_mask = ~roi_mask
        n_roi = int(np.sum(roi_mask))
        n_bg = int(np.sum(bg_mask))

        if n_roi > 0:
            mse_roi = float(np.mean(diff_sq[roi_mask]))
        if n_bg > 0:
            mse_bg = float(np.mean(diff_sq[bg_mask]))

    psnr_roi = _mse_to_psnr(mse_roi)
    psnr_bg = _mse_to_psnr(mse_bg)

    return FramePSNR(
        frame_idx=frame_idx,
        psnr_y_full=psnr_full,
        psnr_y_roi=psnr_roi,
        psnr_y_bg=psnr_bg,
        mse_y_full=mse_full,
        mse_y_roi=mse_roi,
        mse_y_bg=mse_bg,
        n_roi_pixels=n_roi,
        n_bg_pixels=n_bg,
    )


def compute_sequence_psnr(
    original_yuv: str,
    reconstructed_yuv: str,
    width: int,
    height: int,
    n_frames: int,
    boxes_per_frame: Optional[List[List[Tuple[float, float, float, float]]]] = None,
    expansion: float = 0.1,
) -> SequencePSNR:
    """Compute PSNR metrics for an entire sequence.

    Args:
        original_yuv: Path to original raw YUV file.
        reconstructed_yuv: Path to reconstructed raw YUV file.
        width: Frame width.
        height: Frame height.
        n_frames: Number of frames.
        boxes_per_frame: Optional per-frame bounding boxes for ROI PSNR.
        expansion: Box expansion fraction.

    Returns:
        SequencePSNR with per-frame and aggregated metrics.
    """
    per_frame = []

    orig_gen = iter_yuv420_frames(original_yuv, width, height, n_frames)
    recon_gen = iter_yuv420_frames(reconstructed_yuv, width, height, n_frames)

    for (idx_o, Y_o, _, _), (idx_r, Y_r, _, _) in zip(orig_gen, recon_gen):
        boxes = None
        if boxes_per_frame and idx_o < len(boxes_per_frame):
            boxes = boxes_per_frame[idx_o]

        fp = compute_frame_psnr(Y_o, Y_r, boxes, idx_o, expansion)
        per_frame.append(fp)

    if not per_frame:
        return SequencePSNR(
            n_frames=0,
            psnr_y_full_mean=0, psnr_y_roi_mean=0, psnr_y_bg_mean=0,
            psnr_y_full_std=0, psnr_y_roi_std=0,
            per_frame=[],
        )

    full_arr = np.array([f.psnr_y_full for f in per_frame])
    roi_arr = np.array([f.psnr_y_roi for f in per_frame])
    bg_arr = np.array([f.psnr_y_bg for f in per_frame])

    return SequencePSNR(
        n_frames=len(per_frame),
        psnr_y_full_mean=float(np.mean(full_arr)),
        psnr_y_roi_mean=float(np.mean(roi_arr)),
        psnr_y_bg_mean=float(np.mean(bg_arr)),
        psnr_y_full_std=float(np.std(full_arr)),
        psnr_y_roi_std=float(np.std(roi_arr)),
        per_frame=per_frame,
    )
