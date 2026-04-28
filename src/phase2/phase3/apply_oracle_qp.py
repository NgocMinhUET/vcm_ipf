"""Phase 3 — Oracle-Direct QP mapping.

Applies a physically-motivated asymmetric power-law QP offset function
directly on the pre-computed **occlusion saliency** maps
(``phi_oracle_norm_*.npy``) produced by :mod:`phase2.phase3.occlusion_saliency`.

Motivation
----------
Parametric surrogate fitting (Phase 3 Stage B.4) repeatedly converged to
degenerate boundary solutions because the task surrogate penalised ALL
CTUs (including background) for positive delta, making delta_bg collapse
to its minimum. The fundamental cause is that the occlusion saliency
distribution is extremely skewed (p50 ≈ 0.001) so any percentile-based
threshold filters nothing.

This script bypasses the fitting stage entirely.  It uses the oracle
saliency as a direct importance signal and the following
**physically-motivated parameters** (derived from pilot_v1 + quick_v3
measurements):

    mu        = 0.35   — top 35 % of importance = ROI
    delta_roi = 6      — lower ROI QP by 6 steps
    delta_bg  = 4      — raise background QP by 4 steps
    gamma_roi = 1.0    — linear ROI transition
    gamma_bg  = 1.0    — linear background transition

These are equivalent to the L2-asym formula family (theta_L2) with
the saliency map used as the input instead of the IPF field.

Academic justification
----------------------
Occlusion saliency (Zeiler & Fergus, ECCV 2014; ``phi_oracle``) is the
gold-standard task-importance signal: it measures exactly how much the
detector's confidence drops when a CTU is degraded.  Using it directly as
a QP modulator is semantically equivalent to "rate allocation proportional
to task importance" — the core VCM design principle.

By contrast, the physics-based IPF achieves ρ_spearman ≈ 0.27 vs
phi_oracle (measured in this study), confirming that oracle-direct
allocation should outperform IPF-based allocation.

Run with::

    python -m phase2.phase3.apply_oracle_qp \
        --saliency-dir ~/Minh/ipf/phase3_outputs/saliency/MOT17-04-DPM \
        --output-dir   ~/Minh/ipf/phase3_outputs/learned/oracle_MOT17-04-DPM/M4 \
        --n-frames 50 --ctu-rows 9 --ctu-cols 15
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("phase2.phase3.apply_oracle_qp")


# ---------------------------------------------------------------------------
# Asymmetric power-law (identical to qp_mapper._asymmetric_delta).
# ---------------------------------------------------------------------------

def asymmetric_delta(
    phi_norm: np.ndarray,
    mu: float,
    delta_roi: float,
    delta_bg: float,
    gamma_roi: float,
    gamma_bg: float,
) -> np.ndarray:
    """Return signed QP offsets for a 2-D normalized importance map."""
    delta = np.zeros_like(phi_norm, dtype=np.float64)
    roi_mask = phi_norm >= mu
    bg_mask = ~roi_mask
    if roi_mask.any():
        s = ((phi_norm[roi_mask] - mu) / (1.0 - mu + 1e-12)) ** gamma_roi
        delta[roi_mask] = -delta_roi * s
    if bg_mask.any():
        s = ((mu - phi_norm[bg_mask]) / (mu + 1e-12)) ** gamma_bg
        delta[bg_mask] = +delta_bg * s
    return delta


def write_delta_map(
    delta: np.ndarray,
    output_path: Path,
    frame_idx: int,
    delta_min: int = -8,
    delta_max: int = 4,
) -> None:
    """Write a Phase-2-compatible type=delta QP map."""
    clipped = np.clip(np.rint(delta), delta_min, delta_max).astype(np.int32)
    n_rows, n_cols = clipped.shape
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            f"# frame={frame_idx} rows={n_rows} cols={n_cols} type=delta "
            f"delta_min={delta_min} delta_max={delta_max}\n"
        )
        for row in range(n_rows):
            f.write(" ".join(f"{clipped[row, col]:+d}" for col in range(n_cols)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oracle-direct QP mapping: phi_oracle_norm → dQP maps"
    )
    parser.add_argument(
        "--saliency-dir", required=True,
        help="Directory with phi_oracle_norm_*.npy (from occlusion_saliency.py)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory; writes qp_vtm_delta/qp_XXXXXX.txt",
    )
    parser.add_argument("--n-frames", type=int, default=50)
    parser.add_argument("--ctu-rows", type=int, default=9)
    parser.add_argument("--ctu-cols", type=int, default=15)
    # Physically-motivated defaults from pilot_v1 + quick_v3 analysis.
    parser.add_argument("--mu",        type=float, default=0.35)
    parser.add_argument("--delta-roi", type=float, default=6.0)
    parser.add_argument("--delta-bg",  type=float, default=4.0)
    parser.add_argument("--gamma-roi", type=float, default=1.0)
    parser.add_argument("--gamma-bg",  type=float, default=1.0)
    parser.add_argument("--delta-min", type=int,   default=-8)
    parser.add_argument("--delta-max", type=int,   default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sal_dir = Path(args.saliency_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    qp_dir = out_dir / "qp_vtm_delta"
    qp_dir.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = args.ctu_rows, args.ctu_cols
    n_written = 0

    for fi in range(args.n_frames):
        sal_path = sal_dir / f"phi_oracle_norm_{fi:06d}.npy"
        if not sal_path.exists():
            logger.warning("frame %06d: saliency file not found at %s — writing zeros",
                           fi, sal_path)
            phi = np.zeros((n_rows, n_cols), dtype=np.float32)
        else:
            phi = np.load(sal_path).astype(np.float64)
            if phi.shape != (n_rows, n_cols):
                # Resize by max-pooling to the target grid.
                from math import ceil
                src_h, src_w = phi.shape
                bh = ceil(src_h / n_rows)
                bw = ceil(src_w / n_cols)
                resized = np.zeros((n_rows, n_cols), dtype=np.float64)
                for r in range(n_rows):
                    for c in range(n_cols):
                        block = phi[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                        resized[r, c] = float(block.max()) if block.size else 0.0
                phi = resized

        delta = asymmetric_delta(
            phi,
            mu=args.mu,
            delta_roi=args.delta_roi,
            delta_bg=args.delta_bg,
            gamma_roi=args.gamma_roi,
            gamma_bg=args.gamma_bg,
        )
        write_delta_map(delta, qp_dir / f"qp_{fi:06d}.txt", fi,
                        args.delta_min, args.delta_max)
        n_written += 1

    logger.info("Wrote %d delta maps to %s", n_written, qp_dir)
    logger.info(
        "Parameters: mu=%.2f  delta_roi=%.1f  delta_bg=%.1f  "
        "gamma_roi=%.1f  gamma_bg=%.1f",
        args.mu, args.delta_roi, args.delta_bg, args.gamma_roi, args.gamma_bg,
    )

    # Write a metadata file for traceability.
    import json
    meta = {
        "method": "oracle_direct",
        "saliency_dir": str(sal_dir),
        "mu": args.mu,
        "delta_roi": args.delta_roi,
        "delta_bg": args.delta_bg,
        "gamma_roi": args.gamma_roi,
        "gamma_bg": args.gamma_bg,
        "delta_clip": [args.delta_min, args.delta_max],
        "n_frames": args.n_frames,
        "ctu_grid": [n_rows, n_cols],
    }
    with open(out_dir / "oracle_qp_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
