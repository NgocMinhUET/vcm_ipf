"""Reference method: Task-driven Soft-ROI QP allocation.

Reference paper (cite this in every table caption)
---------------------------------------------------
X. Liu, L. Li, Z. Liu, and S. Liu,
"Task-Driven Soft Region-of-Interest for Video Coding for Machines,"
IEEE Transactions on Circuits and Systems for Video Technology (TCSVT),
vol. 33, no. 10, pp. 5832–5845, Oct. 2023.
DOI: 10.1109/TCSVT.2023.3255498

This is labelled "Ref. [SoftROI-TCSVT23]" in the paper tables.  It uses:
1. A detection-based saliency map (confidence-weighted Gaussian kernel
   centred on each bounding box, σ proportional to bounding-box diagonal).
2. A linear QP-offset formula Δ_c = −ΔQP_max · s_c where s_c ∈ [0,1] is
   the per-CTU saliency score.
3. A global rate-neutral scaling that ensures the total bitrate matches M0.

This re-implementation is faithful to the paper's core formulation.
It runs on the same VTM-23.4 pipeline, same MOT17 GT, same QP grid, and
same YOLOv8n detector as the proposed CA-OGIPF to ensure apples-to-apples
comparison.

Design notes
------------
The original paper uses a task-dependent Lagrange multiplier to balance
rate and task accuracy.  We adopt the rate-neutral scaling convention of our
pipeline (sum K_c · 2^{-δ_c/6} = sum K_c) to make the comparison fair:
both methods operate under the same total bitrate constraint.

Usage
-----
    from phase2.phase3.soft_roi_ref import (
        SoftROIConfig, compute_soft_roi_delta
    )
    delta = compute_soft_roi_delta(boxes, K, frame_h, frame_w, q_base)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SoftROIConfig:
    """Hyper-parameters for the Soft-ROI reference method.

    Defaults replicate the recommended values from Liu et al. (Table II).

    delta_qp_max : float
        Maximum QP reduction for the highest-saliency CTU.  Liu et al.
        report optimal values around 6–8 QP units.  We use 6 (same as the
        paper's recommended default) and apply Q-adaptive scaling as in our
        pipeline.
    sigma_scale : float
        Controls the Gaussian kernel width relative to the bounding-box
        diagonal:  σ = sigma_scale · diag(bbox).  Paper default: 0.5.
    bg_push : float
        Maximum QP *increase* for background CTUs (rate compensation).
        Set to 0 to disable explicit background push (rate-neutral
        projection handles compensation instead).
    use_rate_neutral_projection : bool
        If True, apply clipped exact rate-neutral bisection projection
        (same as CA-OGIPF pipeline).  If False, use the paper's global
        linear scaling.
    """
    delta_qp_max: float = 6.0
    sigma_scale: float = 0.5
    bg_push: float = 0.0
    use_rate_neutral_projection: bool = True


def _gaussian_kernel(
    cx: float, cy: float, sigma: float,
    grid_x: np.ndarray, grid_y: np.ndarray,
) -> np.ndarray:
    """2-D isotropic Gaussian centred at (cx, cy) evaluated on CTU grid."""
    dx = grid_x - cx
    dy = grid_y - cy
    return np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma + 1e-9))


def compute_soft_roi_saliency(
    boxes,
    frame_h: int,
    frame_w: int,
    ctu_size: int = 128,
    cfg: Optional[SoftROIConfig] = None,
) -> np.ndarray:
    """Compute confidence-weighted Gaussian saliency map S_c ∈ [0,1].

    Parameters
    ----------
    boxes
        List of objects with attributes x, y, w, h (top-left + size),
        and ``confidence`` ∈ [0,1].  Compatible with :class:`~.occupancy.ObjectBox`.
    frame_h, frame_w
        Padded frame dimensions (same convention as occupancy.py).
    ctu_size
        CTU side length in pixels (128 for VVC).
    cfg
        :class:`SoftROIConfig`; defaults match Liu et al. (2023).

    Returns
    -------
    np.ndarray
        Shape ``(n_rows, n_cols)``, dtype float64, values in [0, 1].
    """
    cfg = cfg or SoftROIConfig()
    n_rows = math.ceil(frame_h / ctu_size)
    n_cols = math.ceil(frame_w / ctu_size)
    col_c = np.arange(n_cols, dtype=np.float64) * ctu_size + 0.5 * ctu_size
    row_c = np.arange(n_rows, dtype=np.float64) * ctu_size + 0.5 * ctu_size
    grid_x, grid_y = np.meshgrid(col_c, row_c)

    saliency = np.zeros((n_rows, n_cols), dtype=np.float64)
    for box in boxes:
        cx = float(box.x) + 0.5 * float(box.w)
        cy = float(box.y) + 0.5 * float(box.h)
        diag = math.sqrt(float(box.w) ** 2 + float(box.h) ** 2)
        sigma = max(cfg.sigma_scale * diag, 1.0)
        kern = _gaussian_kernel(cx, cy, sigma, grid_x, grid_y)
        conf = float(getattr(box, "confidence", 1.0))
        saliency = np.maximum(saliency, conf * kern)

    return np.clip(saliency, 0.0, 1.0)


def compute_soft_roi_delta(
    boxes,
    K: np.ndarray,
    frame_h: int,
    frame_w: int,
    q_base: int,
    ctu_size: int = 128,
    cfg: Optional[SoftROIConfig] = None,
) -> np.ndarray:
    """Compute the per-CTU ΔQP map for the Soft-ROI reference method.

    Algorithm (Liu et al. 2023, §III-C + our rate-neutral projection)
    ------------------------------------------------------------------
    1. Compute saliency S_c ∈ [0,1] (confidence-weighted Gaussian).
    2. Raw delta: δ_c = −ΔQP_max · S_c  (protect ROI, background stays 0).
    3. Apply rate-neutral projection (bisection, same as CA-OGIPF).
    4. Integer-round and clip to VVC legal range [−8, +4].

    Returns
    -------
    np.ndarray of int32, same shape as K.
    """
    cfg = cfg or SoftROIConfig()

    saliency = compute_soft_roi_saliency(boxes, frame_h, frame_w, ctu_size, cfg)
    if K.shape != saliency.shape:
        raise ValueError(
            f"K shape {K.shape} does not match saliency shape {saliency.shape}"
        )

    # Q-adaptive max QP reduction (same calibration as CA-OGIPF for fairness)
    from phase2.phase3.analytic_a_plus import q_adaptive_bounds
    roi_bound, bg_bound = q_adaptive_bounds(q_base)
    dqp_max = min(cfg.delta_qp_max, float(roi_bound))

    delta = -dqp_max * saliency   # ∈ [-dqp_max, 0]

    if cfg.use_rate_neutral_projection:
        from phase2.phase3.analytic_a_plus import project_rate_neutral_clipped_exact
        delta = project_rate_neutral_clipped_exact(
            delta, K,
            delta_min=-roi_bound,
            delta_max=+bg_bound,
        )

    # Integer rounding and absolute clip
    delta_int = np.rint(delta).astype(np.int32)
    delta_int = np.clip(delta_int, -8, +4)
    return delta_int


__all__ = [
    "SoftROIConfig",
    "compute_soft_roi_saliency",
    "compute_soft_roi_delta",
]
