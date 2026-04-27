"""Phase 3 Stage B.2 — Per-CTU rate surrogate (luma variance + R-λ theory).

Reference: ``11_PHASE3_RESEARCH_PROTOCOL.md`` §4.1 (rate surrogate);
Sullivan & Wiegand 1998 ("Rate-distortion optimization for video
compression"); Lin & Chao TCSVT 2016 (per-CTU complexity ≈ luma
standard deviation in HEVC).

Model
-----
Per-CTU bitrate at QP ``Q`` follows the canonical R-λ relationship

        R_c(Q) = K_c · 2^{-Q/6}                         (eqn. 4)

with the "complexity constant" ``K_c`` driven by local image statistics:

        K_c = α · σ_Y(c)^ρ · (1 + |MV(c)|)^ν             (eqn. 5)

We calibrate the global multipliers (α, ρ, ν) by regressing the
**total per-frame bits** of the existing pilot_v1 M0 encodes against the
per-CTU prediction summed over the frame. This needs no new VTM run —
it only requires:

* The original frames (for σ_Y(c)).
* Optional precomputed motion vectors (``--mv-dir``); when missing we
  fall back to a temporal-difference proxy.
* The pilot_v1 ``experiment_summary.json`` (provides per-frame total
  bits at multiple QPs for M0).

After calibration the surrogate exposes a closed-form

        ΔR_c(δ) = K_c · (2^{-(Q_b + δ)/6} - 2^{-Q_b/6})

which is summed across all CTUs / frames to predict the rate cost of
any candidate δ-map. This is several orders of magnitude faster than
re-running VTM.

Run with::

    python -m phase2.phase3.rate_surrogate \
        --frames-dir /data/MOT17/MOT17-04-DPM/img1 \
        --pilot-summary ~/Minh/ipf/phase2_outputs/pilot_v1/experiment_summary.json \
        --sequence MOT17-04-DPM \
        --n-frames 50 \
        --ctu-size 128 \
        --output ~/Minh/ipf/phase3_outputs/rate/MOT17-04-DPM/rate_surrogate.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("phase2.phase3.rate_surrogate")


# ---------------------------------------------------------------------------
# Per-CTU image statistics.
# ---------------------------------------------------------------------------

def _per_ctu_luma_std(frame_bgr: np.ndarray, ctu_size: int) -> np.ndarray:
    """Return per-CTU σ of the Y channel (BT.601). Shape (n_rows, n_cols)."""
    y = (0.114 * frame_bgr[..., 0]
         + 0.587 * frame_bgr[..., 1]
         + 0.299 * frame_bgr[..., 2]).astype(np.float64)
    h, w = y.shape
    n_rows = (h + ctu_size - 1) // ctu_size
    n_cols = (w + ctu_size - 1) // ctu_size
    out = np.zeros((n_rows, n_cols), dtype=np.float64)
    for r in range(n_rows):
        for c in range(n_cols):
            y0 = r * ctu_size; y1 = min(y0 + ctu_size, h)
            x0 = c * ctu_size; x1 = min(x0 + ctu_size, w)
            block = y[y0:y1, x0:x1]
            if block.size == 0:
                out[r, c] = 0.0
            else:
                out[r, c] = float(np.std(block))
    return out


def _per_ctu_temporal_diff(prev_bgr: np.ndarray, curr_bgr: np.ndarray,
                           ctu_size: int) -> np.ndarray:
    """Mean |Δluma| per CTU. A zero-cost proxy for motion magnitude."""
    yp = (0.114 * prev_bgr[..., 0] + 0.587 * prev_bgr[..., 1] + 0.299 * prev_bgr[..., 2])
    yc = (0.114 * curr_bgr[..., 0] + 0.587 * curr_bgr[..., 1] + 0.299 * curr_bgr[..., 2])
    diff = np.abs(yc.astype(np.float64) - yp.astype(np.float64))
    h, w = diff.shape
    n_rows = (h + ctu_size - 1) // ctu_size
    n_cols = (w + ctu_size - 1) // ctu_size
    out = np.zeros((n_rows, n_cols), dtype=np.float64)
    for r in range(n_rows):
        for c in range(n_cols):
            y0 = r * ctu_size; y1 = min(y0 + ctu_size, h)
            x0 = c * ctu_size; x1 = min(x0 + ctu_size, w)
            block = diff[y0:y1, x0:x1]
            out[r, c] = float(np.mean(block)) if block.size else 0.0
    return out


# ---------------------------------------------------------------------------
# Calibration of (alpha, rho, nu) from pilot_v1 M0 totals.
# ---------------------------------------------------------------------------

def _load_pilot_m0_bits(summary_path: Path, sequence: str) -> Dict[int, float]:
    """Return {qp: total_bits} for sequence's M0 row (one entry per QP)."""
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("results", data) if isinstance(data, dict) else data
    out: Dict[int, float] = {}
    for r in rows:
        if r.get("sequence") != sequence or r.get("method") != "M0":
            continue
        qp = int(r.get("qp_base") or r.get("qp") or 0)
        # Prefer explicit total_bits; else derive from kbps × seconds.
        if "total_bits" in r:
            bits = float(r["total_bits"])
        else:
            kbps = float((r.get("rate") or {}).get("kbps", r.get("bitrate_kbps", 0.0)))
            n = int(r.get("n_frames", 50))
            fps = float(r.get("fps", 30.0))
            bits = kbps * 1000.0 * (n / fps)
        out[qp] = bits
    return out


def calibrate_surrogate(
    sigma_per_frame: List[np.ndarray],
    motion_per_frame: List[np.ndarray],
    qp_to_bits: Dict[int, float],
) -> Tuple[float, float, float, Dict[int, float]]:
    """Fit (α, ρ, ν) and per-QP residual scale.

    Strategy: jointly optimize the log-bits regression

        log B(Q) = log α + ρ · log Σ σ̄ + ν · log Σ μ̄ + (-Q · ln 2 / 6)

    where the last term is fixed (slope = -ln2/6 ≈ -0.1155 in nats per QP).
    Subtracting the fixed slope leaves a small linear regression in the
    aggregate luma + motion features.
    """
    if not qp_to_bits:
        raise SystemExit("calibrate_surrogate: no QP -> bits mapping provided")

    # Frame-aggregated features (one scalar per frame).
    sig_tot = np.array([float(s.sum()) for s in sigma_per_frame], dtype=np.float64)
    mot_tot = np.array([float(m.sum()) + 1.0 for m in motion_per_frame], dtype=np.float64)

    # Build the regression rows: one per (QP × frame).
    rows = []
    targets = []
    for qp, bits in sorted(qp_to_bits.items()):
        if bits <= 0:
            continue
        # Distribute total bits proportionally to per-frame predicted activity
        # so that the regression sees per-frame, not averaged, behaviour.
        weight = sig_tot * mot_tot
        weight = weight / max(1e-9, weight.sum())
        for fi in range(len(sigma_per_frame)):
            b_fi = bits * weight[fi]
            if b_fi <= 0:
                continue
            rows.append([1.0, math.log(max(1e-6, sig_tot[fi])),
                         math.log(max(1e-6, mot_tot[fi]))])
            targets.append(math.log(b_fi) + qp * math.log(2.0) / 6.0)

    A = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    log_alpha, rho, nu = float(coef[0]), float(coef[1]), float(coef[2])
    alpha = math.exp(log_alpha)

    # Residual per-QP scale to absorb any systematic mismatch.
    residual: Dict[int, float] = {}
    for qp, bits in qp_to_bits.items():
        pred = sum(alpha * (s.sum() ** rho) * ((m.sum() + 1.0) ** nu)
                   for s, m in zip(sigma_per_frame, motion_per_frame)) * (2.0 ** (-qp / 6.0))
        residual[qp] = float(bits / max(1e-9, pred))

    return alpha, rho, nu, residual


def _per_ctu_K(sigma_grid: np.ndarray, motion_grid: np.ndarray,
               alpha: float, rho: float, nu: float) -> np.ndarray:
    """K_c grid given the calibrated parameters."""
    return alpha * (sigma_grid ** rho) * ((motion_grid + 1.0) ** nu)


# ---------------------------------------------------------------------------
# CLI driver.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 rate surrogate calibrator")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--pilot-summary", required=True,
                        help="Path to pilot_v1 experiment_summary.json")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--n-frames", type=int, default=50)
    parser.add_argument("--ctu-size", type=int, default=128)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output", required=True,
                        help="Output .npz with K_c grids and (alpha, rho, nu)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import cv2
    frames_dir = Path(args.frames_dir).expanduser().resolve()
    paths = sorted([p for p in frames_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.frame_stride > 1:
        paths = paths[::args.frame_stride]
    paths = paths[:args.n_frames]
    if not paths:
        raise SystemExit(f"No frames in {frames_dir}")

    sigma_grids: List[np.ndarray] = []
    motion_grids: List[np.ndarray] = []
    prev = None
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Could not read %s — skip", p)
            continue
        sigma_grids.append(_per_ctu_luma_std(img, args.ctu_size))
        if prev is not None:
            motion_grids.append(_per_ctu_temporal_diff(prev, img, args.ctu_size))
        else:
            motion_grids.append(np.zeros_like(sigma_grids[-1]))
        prev = img

    qp_to_bits = _load_pilot_m0_bits(Path(args.pilot_summary).expanduser().resolve(),
                                     args.sequence)
    if not qp_to_bits:
        raise SystemExit(
            f"No pilot_v1 M0 entries for sequence={args.sequence}. "
            f"Run pilot_v1 first or pass --pilot-summary with a richer file.")
    logger.info("M0 pilot bits %s: %s", args.sequence,
                {k: round(v, 1) for k, v in qp_to_bits.items()})

    alpha, rho, nu, residuals = calibrate_surrogate(sigma_grids, motion_grids, qp_to_bits)
    logger.info("Calibrated  α=%.6g  ρ=%.3f  ν=%.3f", alpha, rho, nu)
    logger.info("Per-QP residual scale: %s",
                {k: round(v, 3) for k, v in residuals.items()})

    K_grids = np.stack([_per_ctu_K(s, m, alpha, rho, nu)
                        for s, m in zip(sigma_grids, motion_grids)], axis=0)

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        K_grids=K_grids,            # (n_frames, n_rows, n_cols)
        sigma_grids=np.stack(sigma_grids, axis=0),
        motion_grids=np.stack(motion_grids, axis=0),
        alpha=np.float64(alpha),
        rho=np.float64(rho),
        nu=np.float64(nu),
        residuals_qp=np.array(sorted(residuals.keys()), dtype=np.int32),
        residuals_scale=np.array([residuals[q] for q in sorted(residuals.keys())],
                                 dtype=np.float64),
        ctu_size=np.int32(args.ctu_size),
        sequence=np.array(args.sequence),
    )
    logger.info("Wrote %s  (K_grids shape=%s)", out, K_grids.shape)


# Public helper for downstream modules (fit_parametric, build_oracle).
def predict_delta_rate(K_grid: np.ndarray, q_base: int, delta: np.ndarray,
                        residual_scale: float = 1.0) -> np.ndarray:
    """Return ΔR_c (per CTU) in the same units as the calibration totals.

    Parameters
    ----------
    K_grid : (n_rows, n_cols)
    q_base : int
    delta  : (n_rows, n_cols)  signed integer offsets
    residual_scale : float — multiplier from the per-QP residual table.
    """
    base = 2.0 ** (-q_base / 6.0)
    pert = 2.0 ** (-(q_base + delta) / 6.0)
    return residual_scale * K_grid * (pert - base)


if __name__ == "__main__":
    main()
