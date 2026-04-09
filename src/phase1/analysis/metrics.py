"""Quantitative metrics for comparing QP map methods.

Implements the key measurements needed for scientific comparison:
    1. QP distribution statistics (mean, std, min, max per frame)
    2. Temporal QP variance (frame-to-frame stability)
    3. Spatial smoothness (gradient magnitude)
    4. ROI/BG separation (inter-class QP gap)
    5. Temporal jitter (per-CTU QP change across frames)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from phase1.utils.log import get_logger

logger = get_logger("analysis.metrics")


@dataclass
class FrameQPStats:
    """QP statistics for a single frame."""
    frame_idx: int
    n_objects: int
    qp_min: int
    qp_max: int
    qp_mean: float
    qp_std: float
    qp_median: float
    roi_ctu_count: int
    bg_ctu_count: int
    roi_qp_mean: float
    bg_qp_mean: float
    importance_mean: float
    importance_max: float


@dataclass
class TemporalMetrics:
    """Temporal stability metrics across a sequence."""
    method_id: str
    n_frames: int
    qp_mean_of_means: float
    qp_std_of_means: float
    mean_frame_to_frame_delta: float
    max_frame_to_frame_delta: float
    per_ctu_temporal_std_mean: float
    per_ctu_temporal_std_max: float
    temporal_jitter_index: float


def compute_frame_stats(
    qp_map: np.ndarray,
    importance_map: np.ndarray,
    frame_idx: int,
    n_objects: int,
    mu: float = 0.3,
) -> FrameQPStats:
    """Compute detailed QP statistics for one frame."""
    roi_mask = importance_map >= mu
    bg_mask = ~roi_mask

    return FrameQPStats(
        frame_idx=frame_idx,
        n_objects=n_objects,
        qp_min=int(np.min(qp_map)),
        qp_max=int(np.max(qp_map)),
        qp_mean=float(np.mean(qp_map)),
        qp_std=float(np.std(qp_map)),
        qp_median=float(np.median(qp_map)),
        roi_ctu_count=int(np.sum(roi_mask)),
        bg_ctu_count=int(np.sum(bg_mask)),
        roi_qp_mean=float(np.mean(qp_map[roi_mask])) if np.any(roi_mask) else float(np.mean(qp_map)),
        bg_qp_mean=float(np.mean(qp_map[bg_mask])) if np.any(bg_mask) else float(np.mean(qp_map)),
        importance_mean=float(np.mean(importance_map)),
        importance_max=float(np.max(importance_map)),
    )


def compute_temporal_metrics(
    qp_maps: list[np.ndarray],
    method_id: str,
    skip_first_n: int = 0,
) -> TemporalMetrics:
    """Compute temporal stability metrics across all frames.

    This is the KEY metric for proving temporal consistency (KPI-2 in charter).

    Args:
        qp_maps: List of 2D QP arrays, one per frame.
        method_id: Method identifier for labeling.
        skip_first_n: Number of initial frames to exclude from metrics.
            This removes transient warmup effects (EMA convergence,
            track-age ramp) that would unfairly penalize methods with
            temporal state. Set to 0 for full-sequence evaluation.
    """
    if skip_first_n > 0 and len(qp_maps) > skip_first_n:
        qp_maps = qp_maps[skip_first_n:]

    n_frames = len(qp_maps)
    if n_frames < 2:
        return TemporalMetrics(
            method_id=method_id, n_frames=n_frames,
            qp_mean_of_means=0, qp_std_of_means=0,
            mean_frame_to_frame_delta=0, max_frame_to_frame_delta=0,
            per_ctu_temporal_std_mean=0, per_ctu_temporal_std_max=0,
            temporal_jitter_index=0,
        )

    means = [float(np.mean(qp)) for qp in qp_maps]

    deltas = [abs(means[i+1] - means[i]) for i in range(n_frames - 1)]

    stacked = np.stack(qp_maps, axis=0).astype(np.float64)  # (T, R, C)
    per_ctu_std = np.std(stacked, axis=0)  # (R, C)

    per_ctu_deltas = []
    for i in range(n_frames - 1):
        delta = np.abs(qp_maps[i+1].astype(float) - qp_maps[i].astype(float))
        per_ctu_deltas.append(float(np.mean(delta)))

    return TemporalMetrics(
        method_id=method_id,
        n_frames=n_frames,
        qp_mean_of_means=float(np.mean(means)),
        qp_std_of_means=float(np.std(means)),
        mean_frame_to_frame_delta=float(np.mean(deltas)),
        max_frame_to_frame_delta=float(np.max(deltas)),
        per_ctu_temporal_std_mean=float(np.mean(per_ctu_std)),
        per_ctu_temporal_std_max=float(np.max(per_ctu_std)),
        temporal_jitter_index=float(np.mean(per_ctu_deltas)),
    )


def compute_spatial_smoothness(qp_map: np.ndarray) -> float:
    """Mean absolute gradient magnitude of QP map.

    Lower = smoother QP transitions (less blocking artifacts at CTU boundaries).
    """
    if qp_map.shape[0] < 2 or qp_map.shape[1] < 2:
        return 0.0
    qp_f = qp_map.astype(np.float64)
    gy = np.abs(np.diff(qp_f, axis=0))  # vertical gradients
    gx = np.abs(np.diff(qp_f, axis=1))  # horizontal gradients
    return float(np.mean(gy) + np.mean(gx)) / 2.0
