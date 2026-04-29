"""Phase 3 Stage C — Build the LiteQP residual training dataset.

For each (sequence, frame, CTU, Q_b) we emit a row containing:

* **Per-CTU features** (input to the regressor / tiny CNN):
    phi, K_c, K_c_norm, q_base_norm, sigma_y_proxy, motion_proxy,
    temporal_reliability, prev_delta_norm, phi_neighbor_mean, phi_grad
* **Analytic prior** delta_a_plus from RD-log A+ (analytic_a_plus.py).
* **Teacher label** delta_star = arg min over a discrete delta grid of the
  RD-Lagrangian on the calibrated rate / task surrogates plus a soft
  Tikhonov term that anchors the choice to delta_a_plus:

      δ*_c = arg min_{δ ∈ Δ} [ ΔR_c(δ) − λ_task · Φ_c · τ(δ)
                              + λ_s · |δ − δ_a_plus_c| ]

  where
    * ΔR_c(δ) = K_c · (2^{-(Q_b+δ)/6} − 2^{-Q_b/6})    (R-λ rate model)
    * τ(δ)    = -η · max(0, δ) + η · ξ · max(0, -δ)    (asymmetric task)

This teacher is intentionally simple — it is the **closed-form RD-Lagrangian**
on the same surrogates used elsewhere, with a Tikhonov anchor that forces
the label to stay near the analytic A+ allocation. The MLP then only needs
to learn a small **residual** (typically ±1–2 QP) on top of A+, which is
both safer (rate-neutrality is approximately preserved) and easier to
generalise (small-data regime, 3 sequences).

Run with::

    python -m phase2.phase3.build_liteqp_dataset \
        --rate-npz ~/Minh/ipf/phase3_outputs/rate/MOT17-04-DPM/rate_surrogate.npz \
        --saliency-dir ~/Minh/ipf/phase3_outputs/saliency/MOT17-04-DPM \
        --frames-dir ~/Minh/ipf/datasets/MOT17/MOT17/train/MOT17-04-DPM/img1 \
        --sequence MOT17-04-DPM \
        --qp-list 27 32 37 42 \
        --delta-list -8 -6 -4 -2 0 2 4 \
        --lambda-task 5.0 --lambda-anchor 0.6 \
        --eta 0.06 --xi 0.15 \
        --output ~/Minh/ipf/phase3_outputs/oracle/MOT17-04-DPM_liteqp.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from phase2.phase3.analytic_a_plus import (
    AnalyticAPlusConfig,
    compute_a_plus_delta,
    project_rate_neutral_exact,
    q_adaptive_bounds,
)

logger = logging.getLogger("phase2.phase3.build_liteqp_dataset")


# ---------------------------------------------------------------------------
# Helpers (mirroring rate_surrogate.py + occlusion_saliency.py)
# ---------------------------------------------------------------------------

def _percentile_norm(arr: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> np.ndarray:
    lo = float(np.percentile(arr, p_lo))
    hi = float(np.percentile(arr, p_hi))
    return np.clip((arr - lo) / max(1e-9, hi - lo), 0.0, 1.0)


def _per_ctu_luma_std(frame_bgr: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """σ_Y per CTU; falls back to constant 1.0 if frame missing."""
    y = (0.114 * frame_bgr[..., 0]
         + 0.587 * frame_bgr[..., 1]
         + 0.299 * frame_bgr[..., 2]).astype(np.float64)
    h, w = y.shape
    bh = max(1, (h + n_rows - 1) // n_rows)
    bw = max(1, (w + n_cols - 1) // n_cols)
    out = np.zeros((n_rows, n_cols), dtype=np.float64)
    for r in range(n_rows):
        for c in range(n_cols):
            block = y[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            out[r, c] = float(np.std(block)) if block.size else 0.0
    return out


def _per_ctu_temporal_diff(prev_bgr: np.ndarray, curr_bgr: np.ndarray,
                            n_rows: int, n_cols: int) -> np.ndarray:
    yp = 0.114 * prev_bgr[..., 0] + 0.587 * prev_bgr[..., 1] + 0.299 * prev_bgr[..., 2]
    yc = 0.114 * curr_bgr[..., 0] + 0.587 * curr_bgr[..., 1] + 0.299 * curr_bgr[..., 2]
    diff = np.abs(yc.astype(np.float64) - yp.astype(np.float64))
    h, w = diff.shape
    bh = max(1, (h + n_rows - 1) // n_rows)
    bw = max(1, (w + n_cols - 1) // n_cols)
    out = np.zeros((n_rows, n_cols), dtype=np.float64)
    for r in range(n_rows):
        for c in range(n_cols):
            block = diff[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            out[r, c] = float(np.mean(block)) if block.size else 0.0
    return out


def _phi_neighbor_mean(phi: np.ndarray) -> np.ndarray:
    """3x3 box-mean of phi (zero-padded, the centre cell is included)."""
    h, w = phi.shape
    out = np.zeros_like(phi)
    for r in range(h):
        for c in range(w):
            r0, r1 = max(0, r - 1), min(h, r + 2)
            c0, c1 = max(0, c - 1), min(w, c + 2)
            out[r, c] = float(np.mean(phi[r0:r1, c0:c1]))
    return out


def _phi_gradient(phi: np.ndarray) -> np.ndarray:
    """L1 gradient magnitude (forward differences, zero-padded)."""
    h, w = phi.shape
    gh = np.zeros_like(phi); gw = np.zeros_like(phi)
    gh[:-1, :] = np.abs(phi[1:, :] - phi[:-1, :])
    gw[:, :-1] = np.abs(phi[:, 1:] - phi[:, :-1])
    return gh + gw


# ---------------------------------------------------------------------------
# Teacher label: closed-form RD-Lagrangian over a discrete δ grid
# ---------------------------------------------------------------------------

def _delta_rate_per_ctu(K_c: float, q_base: int, delta: int,
                         residual_scale: float) -> float:
    """ΔR_c(δ) in raw units (matches rate_surrogate sign convention)."""
    return float(residual_scale * K_c
                 * (2.0 ** (-(q_base + delta) / 6.0)
                    - 2.0 ** (-q_base / 6.0)))


def _delta_task_per_ctu(phi: float, delta: int,
                         eta: float, xi: float) -> float:
    """Asymmetric per-CTU mAP response: positive when δ < 0 (better quality)."""
    pos = max(0, delta)
    neg = max(0, -delta)
    return float(-(phi * eta * pos) + (phi * eta * xi * neg))


def _teacher_delta_star(
    phi_grid: np.ndarray,
    K_grid: np.ndarray,
    delta_a_plus_grid: np.ndarray,
    q_base: int,
    delta_grid: List[int],
    lambda_task: float,
    lambda_anchor: float,
    eta: float,
    xi: float,
    residual_scale: float,
) -> np.ndarray:
    """Per-CTU δ_star minimising rate − λ_task · task + λ_s · |δ − δ_a+|."""
    n_rows, n_cols = phi_grid.shape
    deltas = np.asarray(delta_grid, dtype=np.float64)
    out = np.zeros((n_rows, n_cols), dtype=np.float64)

    for r in range(n_rows):
        for c in range(n_cols):
            phi_c = float(phi_grid[r, c])
            K_c = float(K_grid[r, c])
            anchor_c = float(delta_a_plus_grid[r, c])
            best_score = float("inf"); best_delta = 0
            for d_int in delta_grid:
                d = float(d_int)
                d_r = _delta_rate_per_ctu(K_c, q_base, d_int, residual_scale)
                d_m = _delta_task_per_ctu(phi_c, d_int, eta, xi)
                score = d_r - lambda_task * d_m + lambda_anchor * abs(d - anchor_c)
                if score < best_score:
                    best_score = score; best_delta = d_int
            out[r, c] = float(best_delta)
    return out


# ---------------------------------------------------------------------------
# Frame loading (lazy — we only need per-CTU stats, not full encoding)
# ---------------------------------------------------------------------------

def _load_frame_bgr(frames_dir: Path, frame_idx: int,
                     pattern: str = "{:06d}.jpg") -> Optional[np.ndarray]:
    path = frames_dir / pattern.format(frame_idx + 1)  # MOT17 is 1-indexed
    if not path.exists():
        path = frames_dir / pattern.format(frame_idx)
    if not path.exists():
        return None
    try:
        import cv2
    except ImportError:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LiteQP residual dataset builder")
    parser.add_argument("--rate-npz", required=True,
                        help="Per-frame K_grids from rate_surrogate.py")
    parser.add_argument("--saliency-dir", required=True,
                        help="phi_oracle*.npy directory from occlusion_saliency.py")
    parser.add_argument("--frames-dir", default="",
                        help="Original frame directory (for σ_Y / motion). "
                             "If empty, those features are filled with placeholders.")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--qp-list", nargs="+", type=int, default=[27, 32, 37, 42])
    parser.add_argument("--delta-list", nargs="+", type=int,
                        default=[-8, -6, -4, -2, 0, 2, 4])
    parser.add_argument("--lambda-task", type=float, default=5.0,
                        help="Task gain in the Lagrangian (per per-CTU surrogate scale)")
    parser.add_argument("--lambda-anchor", type=float, default=0.6,
                        help="Tikhonov weight pulling δ* toward δ_a+")
    parser.add_argument("--eta", type=float, default=0.06)
    parser.add_argument("--xi", type=float, default=0.15)
    parser.add_argument("--residual-bound", type=float, default=2.0,
                        help="Cap on |δ_star − δ_a+| (saved as a row field)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rate_data = np.load(Path(args.rate_npz).expanduser().resolve())
    K_grids = rate_data["K_grids"]                # (n_frames, n_rows, n_cols)
    residual_qps = rate_data["residuals_qp"].tolist()
    residual_scales = rate_data["residuals_scale"].tolist()
    residual_map = {int(q): float(s) for q, s in zip(residual_qps, residual_scales)}
    n_frames, n_rows, n_cols = K_grids.shape
    logger.info("Loaded rate surrogate: %d frames, %dx%d CTU grid",
                n_frames, n_rows, n_cols)

    sal_dir = Path(args.saliency_dir).expanduser().resolve()
    frames_dir = (Path(args.frames_dir).expanduser().resolve()
                  if args.frames_dir else None)
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-compute per-frame σ_Y and motion proxies if frames available.
    sigma_per_frame: List[np.ndarray] = []
    motion_per_frame: List[np.ndarray] = []
    prev_bgr = None
    if frames_dir is not None:
        logger.info("Pre-computing per-CTU σ_Y / motion stats for %d frames", n_frames)
        for fi in range(n_frames):
            curr_bgr = _load_frame_bgr(frames_dir, fi)
            if curr_bgr is None:
                sigma_per_frame.append(np.ones((n_rows, n_cols)))
                motion_per_frame.append(np.zeros((n_rows, n_cols)))
                continue
            sigma_per_frame.append(_per_ctu_luma_std(curr_bgr, n_rows, n_cols))
            if prev_bgr is None:
                motion_per_frame.append(np.zeros((n_rows, n_cols)))
            else:
                motion_per_frame.append(
                    _per_ctu_temporal_diff(prev_bgr, curr_bgr, n_rows, n_cols))
            prev_bgr = curr_bgr
        # Normalise σ_Y and motion to 0-1 range (per sequence) for stable training.
        sigma_arr = np.stack(sigma_per_frame, axis=0)
        motion_arr = np.stack(motion_per_frame, axis=0)
        sigma_arr = sigma_arr / max(1e-9, float(np.percentile(sigma_arr, 95)))
        motion_arr = motion_arr / max(1e-9, float(np.percentile(motion_arr, 95)))
        sigma_per_frame = [sigma_arr[i] for i in range(n_frames)]
        motion_per_frame = [motion_arr[i] for i in range(n_frames)]
    else:
        logger.warning("No frames-dir; filling σ_Y/motion features with placeholders.")
        sigma_per_frame = [np.ones((n_rows, n_cols)) for _ in range(n_frames)]
        motion_per_frame = [np.zeros((n_rows, n_cols)) for _ in range(n_frames)]

    cfg = AnalyticAPlusConfig()
    n_written = 0
    n_clipped = 0

    with open(out_path, "w", encoding="utf-8") as f:
        # Track per-frame previous δ_a+ to populate prev_delta_norm.
        prev_delta_grid = {qp: np.zeros((n_rows, n_cols)) for qp in args.qp_list}

        for fi in range(n_frames):
            sal_path = sal_dir / f"phi_oracle_{fi:06d}.npy"
            if not sal_path.exists():
                logger.warning("frame %06d: missing saliency, skipping", fi)
                continue
            phi_raw = np.load(sal_path).astype(np.float64)
            if phi_raw.shape != (n_rows, n_cols):
                logger.warning("frame %06d: saliency shape %s ≠ %s, skipping",
                               fi, phi_raw.shape, (n_rows, n_cols))
                continue
            phi_norm = _percentile_norm(phi_raw)
            phi_nbr = _phi_neighbor_mean(phi_norm)
            phi_grd = _phi_gradient(phi_norm)
            K_grid = K_grids[fi]
            sigma_grid = sigma_per_frame[fi]
            motion_grid = motion_per_frame[fi]
            K_norm = K_grid / (np.median(K_grid[K_grid > 0]) + 1e-9
                               if (K_grid > 0).any() else 1.0)

            for qp in args.qp_list:
                resid = residual_map.get(int(qp), 1.0)
                # 1) analytic prior
                delta_a = compute_a_plus_delta(phi_norm, K_grid, qp, cfg)
                # rate-neutral projection (helps the teacher stay realistic)
                delta_a = project_rate_neutral_exact(delta_a, K_grid)
                # 2) teacher
                delta_star = _teacher_delta_star(
                    phi_norm, K_grid, delta_a, qp, args.delta_list,
                    args.lambda_task, args.lambda_anchor,
                    args.eta, args.xi, residual_scale=resid,
                )
                # 3) emit per-CTU rows
                for r in range(n_rows):
                    for c in range(n_cols):
                        residual = float(delta_star[r, c] - delta_a[r, c])
                        if abs(residual) > args.residual_bound:
                            n_clipped += 1
                            residual = float(
                                np.clip(residual, -args.residual_bound,
                                        args.residual_bound))
                        row = {
                            "sequence": args.sequence,
                            "frame_idx": fi,
                            "row": r,
                            "col": c,
                            "q_base": int(qp),
                            # ── features ────────────────────────────────────
                            "phi": float(phi_norm[r, c]),
                            "K_c": float(K_grid[r, c]),
                            "K_c_norm": float(K_norm[r, c]),
                            "q_base_norm": float((qp - 32.0) / 10.0),
                            "sigma_y": float(sigma_grid[r, c]),
                            "motion_proxy": float(motion_grid[r, c]),
                            "temporal_reliability": 1.0,  # placeholder
                            "prev_delta": float(prev_delta_grid[qp][r, c]),
                            "phi_neighbor_mean": float(phi_nbr[r, c]),
                            "phi_grad": float(phi_grd[r, c]),
                            # ── targets ─────────────────────────────────────
                            "delta_a_plus": float(delta_a[r, c]),
                            "delta_star": float(delta_star[r, c]),
                            "residual": residual,
                        }
                        f.write(json.dumps(row) + "\n")
                        n_written += 1
                # update history for next frame
                prev_delta_grid[qp] = delta_a.copy()

    logger.info("Wrote %d rows to %s (%d residuals clipped to ±%.1f)",
                n_written, out_path, n_clipped, args.residual_bound)


if __name__ == "__main__":
    main()
