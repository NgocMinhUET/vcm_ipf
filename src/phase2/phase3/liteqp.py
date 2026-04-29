"""LiteQP utilities for lightweight residual QP estimation.

LiteQP is a conservative Phase-3 extension for VCM-oriented CTU-level
QP allocation.  It does **not** let a learned model freely predict QP.
Instead it uses a theory-guided analytic prior and allows a lightweight
regressor/CNN to learn only a bounded residual correction.

Pipeline:
    importance/rate features -> RD-log A+ prior -> residual model
    -> rate-neutral projection -> Q-adaptive clipping -> delta-QP map.

The module is intentionally NumPy-only so it can run on the analysis box
without PyTorch.  A tiny CNN can be added later, but the first publishable
baseline should be the residual ridge/MLP-style regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LiteQPBounds:
    """QP-dependent safety bounds for signed delta-QP maps.

    delta < 0 means better quality / more bits for ROI.
    delta > 0 means lower quality / fewer bits for background.
    """

    roi: float  # maximum magnitude for negative delta, e.g. 6 => min delta = -6
    bg: float   # maximum positive delta, e.g. 3 => max delta = +3


def q_adaptive_bounds(qp_base: int) -> LiteQPBounds:
    """Return conservative Q-adaptive delta-QP bounds.

    Motivation:
      * At low QP, large background penalties create CTU discontinuities and
        can hurt prediction / task accuracy.  Keep +delta small.
      * At high QP, compression is already strong, so larger background
        deltas and stronger ROI protection are safer.

    Initial schedule is deliberately conservative and should be ablated.
    """
    qp = float(qp_base)
    bg = float(np.clip(2.0 + 0.13 * (qp - 27.0), 2.0, 4.0))
    roi = float(np.clip(4.0 + 0.20 * (qp - 27.0), 4.0, 7.0))
    return LiteQPBounds(roi=roi, bg=bg)


def q_adaptive_scale(qp_base: int) -> float:
    """Scale RD-log allocation according to compression regime."""
    return float(np.clip(0.85 + 0.03 * (float(qp_base) - 32.0), 0.70, 1.15))


def normalize01(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo + eps), 0.0, 1.0)


def percentile_norm(arr: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0,
                    eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    lo = float(np.percentile(arr, p_lo))
    hi = float(np.percentile(arr, p_hi))
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo + eps), 0.0, 1.0)


def safe_rate_complexity(K: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Return a positive CTU rate-complexity map.

    If no rate surrogate is available, use an all-ones map.  This keeps the
    method usable as a drop-in replacement and makes the K-aware version an
    ablation rather than a hard dependency.
    """
    if K is None:
        return np.ones(shape, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if K.shape != shape:
        raise ValueError(f"K shape {K.shape} does not match expected {shape}")
    K = np.nan_to_num(K, nan=1.0, posinf=1.0, neginf=1.0)
    med = float(np.median(K[K > 0])) if np.any(K > 0) else 1.0
    return np.maximum(K, 1e-6 * med)


def compute_rdlog_aplus_delta(
    phi: np.ndarray,
    K: Optional[np.ndarray],
    qp_base: int,
    eps_phi: float = 1e-3,
    beta_k: float = 0.5,
    kappa: float = 0.05,
    project: bool = True,
) -> np.ndarray:
    """Compute the RD-log A+ analytic prior delta-QP map.

    The score xi measures task importance per estimated coding cost:
        xi_c = (phi_c + eps) / (K_norm_c + kappa)^beta.

    The raw log-domain allocation is:
        delta_c = -6 * s(Q_b) * log2(xi_c / mean_K(xi)).

    High xi => negative delta (more bits). Low xi => positive delta.
    The result is projected and clipped for VVC-safe operation.
    """
    phi = percentile_norm(np.asarray(phi, dtype=np.float64))
    shape = phi.shape
    K_map = safe_rate_complexity(K, shape)

    # Normalize K by its positive median so beta_k has stable meaning.
    K_med = float(np.median(K_map[K_map > 0])) if np.any(K_map > 0) else 1.0
    K_norm = K_map / max(K_med, 1e-9)

    xi = (phi + eps_phi) / np.power(K_norm + kappa, beta_k)
    xi = np.maximum(xi, 1e-9)

    # K-weighted mean gives a rate-aware centre.  This is safer than a simple
    # arithmetic mean when a few high-complexity CTUs dominate rate.
    centre = float(np.sum(K_map * xi) / (np.sum(K_map) + 1e-9))
    centre = max(centre, 1e-9)

    delta = -6.0 * q_adaptive_scale(qp_base) * np.log2(xi / centre)

    bounds = q_adaptive_bounds(qp_base)
    delta = np.clip(delta, -bounds.roi, bounds.bg)
    if project:
        delta = project_rate_neutral(delta, K_map)
        delta = np.clip(delta, -bounds.roi, bounds.bg)
    return delta.astype(np.float64)


def project_rate_neutral(delta: np.ndarray, K: Optional[np.ndarray]) -> np.ndarray:
    """First-order rate-neutral projection.

    For small QP perturbations, rate change is approximately proportional to
    K_c * delta_c.  Removing the K-weighted mean keeps the map roughly neutral
    before VTM makes the exact RDO decision.
    """
    delta = np.asarray(delta, dtype=np.float64)
    K_map = safe_rate_complexity(K, delta.shape)
    offset = float(np.sum(K_map * delta) / (np.sum(K_map) + 1e-9))
    return delta - offset


def spatial_context_features(phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return 3x3-neighbour mean and gradient magnitude for a CTU grid."""
    phi = percentile_norm(phi)
    h, w = phi.shape
    padded = np.pad(phi, 1, mode="edge")
    neigh = np.zeros_like(phi, dtype=np.float64)
    for r in range(h):
        for c in range(w):
            neigh[r, c] = float(np.mean(padded[r:r + 3, c:c + 3]))

    gy = np.zeros_like(phi, dtype=np.float64)
    gx = np.zeros_like(phi, dtype=np.float64)
    gy[1:-1, :] = (phi[2:, :] - phi[:-2, :]) * 0.5
    gx[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) * 0.5
    grad = np.sqrt(gx * gx + gy * gy)
    return neigh, grad


def build_liteqp_features(
    phi: np.ndarray,
    K: Optional[np.ndarray],
    qp_base: int,
    delta_a_plus: np.ndarray,
    prev_delta: Optional[np.ndarray] = None,
    sigma_y: Optional[np.ndarray] = None,
    motion: Optional[np.ndarray] = None,
    reliability: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Build per-CTU features for LiteQP residual regression.

    Feature layout:
      0 phi_hat
      1 K_norm
      2 q_base_norm
      3 sigma_y_norm
      4 motion_norm
      5 reliability
      6 prev_delta_norm
      7 phi_neighbor_mean
      8 phi_grad
      9 delta_a_plus_norm
      10 phi_hat * q_base_norm
      11 phi_hat * K_norm
    """
    phi_hat = percentile_norm(phi)
    shape = phi_hat.shape
    K_map = safe_rate_complexity(K, shape)
    K_norm = K_map / (float(np.median(K_map[K_map > 0])) + 1e-9)
    K_norm = np.clip(K_norm, 0.0, 5.0) / 5.0

    q_norm = np.full(shape, (float(qp_base) - 32.0) / 10.0, dtype=np.float64)
    neigh, grad = spatial_context_features(phi_hat)

    def opt_norm(x: Optional[np.ndarray], default: float = 0.0) -> np.ndarray:
        if x is None:
            return np.full(shape, default, dtype=np.float64)
        return normalize01(np.asarray(x, dtype=np.float64))

    sigma = opt_norm(sigma_y)
    mot = opt_norm(motion)
    rel = np.ones(shape, dtype=np.float64) if reliability is None else np.clip(reliability, 0.0, 1.0)
    prev = np.zeros(shape, dtype=np.float64) if prev_delta is None else np.clip(prev_delta / 8.0, -1.0, 1.0)
    da = np.clip(delta_a_plus / 8.0, -1.0, 1.0)

    feats = np.stack([
        phi_hat,
        K_norm,
        q_norm,
        sigma,
        mot,
        rel,
        prev,
        neigh,
        grad,
        da,
        phi_hat * q_norm,
        phi_hat * K_norm,
    ], axis=-1)
    return feats.reshape(-1, feats.shape[-1]).astype(np.float64), shape


def fit_ridge_regression(X: np.ndarray, y: np.ndarray,
                         sample_weight: Optional[np.ndarray] = None,
                         alpha: float = 1e-3) -> Dict[str, np.ndarray]:
    """Fit a tiny weighted ridge residual regressor in closed form."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    Xn = (X - mean) / std
    Xb = np.column_stack([np.ones(len(Xn)), Xn])

    if sample_weight is None:
        W = np.ones(len(Xb), dtype=np.float64)
    else:
        W = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        W = np.maximum(W, 1e-6)
    Xw = Xb * np.sqrt(W)[:, None]
    yw = y * np.sqrt(W)

    reg = alpha * np.eye(Xb.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0  # do not regularize intercept
    coef = np.linalg.solve(Xw.T @ Xw + reg, Xw.T @ yw)
    return {"coef": coef, "mean": mean, "std": std}


def predict_ridge(model: Dict[str, np.ndarray], X: np.ndarray,
                  residual_bound: float = 2.0) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Xn = (X - model["mean"]) / (model["std"] + 1e-9)
    Xb = np.column_stack([np.ones(len(Xn)), Xn])
    y = Xb @ model["coef"]
    return np.clip(y, -residual_bound, residual_bound)


def write_delta_map(delta: np.ndarray, output_path, frame_idx: int,
                    delta_min: int = -8, delta_max: int = 4) -> None:
    """Write a VTM-compatible delta-QP text map."""
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.rint(delta), delta_min, delta_max).astype(np.int32)
    n_rows, n_cols = clipped.shape
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            f"# frame={frame_idx} rows={n_rows} cols={n_cols} type=delta "
            f"delta_min={delta_min} delta_max={delta_max} method=LiteQP\n"
        )
        for r in range(n_rows):
            f.write(" ".join(f"{int(v):+d}" for v in clipped[r]) + "\n")
