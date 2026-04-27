"""Phase 3 Stage B.3 — Build the oracle dataset from cheap surrogates.

Reference: ``11_PHASE3_RESEARCH_PROTOCOL.md`` §3 + §4.

This script materializes the (features, δ, ΔR, ΔmAP) tuples needed by
:mod:`phase2.phase3.fit_parametric`, **without running VTM**. The
ground-truth (ΔR, ΔmAP) values are predicted from the closed-form
surrogates trained in:

* :mod:`phase2.phase3.rate_surrogate` (per-CTU K_c grids).
* :mod:`phase2.phase3.occlusion_saliency` (per-CTU Φ_oracle grids).

For each (sequence, frame, Q_base) cell of the experimental grid we
sample a small number of CTUs and apply a discrete δ ∈ Δ. We then
predict:

    ΔR(c, δ)  = K_c · (2^{-(Q_b+δ)/6} - 2^{-Q_b/6})
    ΔmAP(c,δ) = -Φ_oracle_norm(c) · η · max(0, δ)
              + +Φ_oracle_norm(c) · η · ξ · max(0, -δ)   (asymmetric)

where η is the sensitivity gain (default 0.06 per QP step) and
ξ ∈ [0, 0.3] is the diminishing return for δ < 0 (lower QP rarely
*increases* mAP for boxes that already detect well).

The resulting Parquet (or JSONL) is consumed verbatim by
``fit_parametric.py``. Because surrogate evaluation is essentially
free (all NumPy vector ops), we generate ~50 000 samples in a few
seconds — orders of magnitude richer than what we could ever afford
with real VTM.

Run with::

    python -m phase2.phase3.build_oracle \
        --rate-npz ~/Minh/ipf/phase3_outputs/rate/MOT17-04-DPM/rate_surrogate.npz \
        --saliency-dir ~/Minh/ipf/phase3_outputs/saliency/MOT17-04-DPM \
        --phi-ipf-dir ~/Minh/ipf/phase1_outputs_v2/ipf_lp_pinf_MOT17-04-DPM/M4 \
        --sequence MOT17-04-DPM \
        --qp-list 27 32 37 42 \
        --delta-list -8 -4 -2 0 +2 +4 \
        --samples-per-cell 200 \
        --output ~/Minh/ipf/phase3_outputs/oracle/MOT17-04-DPM.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("phase2.phase3.build_oracle")


# ---------------------------------------------------------------------------
# Feature loading helpers — mirror occlusion_saliency._load_phi_ipf.
# ---------------------------------------------------------------------------

def _load_phi_grid(grid_dir: Path, frame_idx: int, n_rows: int, n_cols: int,
                    name_pattern: str = "field_") -> Optional[np.ndarray]:
    candidates = [
        grid_dir / f"{name_pattern}{frame_idx:06d}.npy",
        grid_dir / f"phi_{frame_idx:06d}.npy",
        grid_dir / f"importance_{frame_idx:06d}.npy",
    ]
    for path in candidates:
        if path.exists():
            arr = np.load(path).astype(np.float64)
            if arr.ndim == 2 and arr.shape == (n_rows, n_cols):
                return arr
            if arr.ndim == 2:
                ctu_h = max(1, math.ceil(arr.shape[0] / n_rows))
                ctu_w = max(1, math.ceil(arr.shape[1] / n_cols))
                pooled = np.zeros((n_rows, n_cols), dtype=np.float64)
                for r in range(n_rows):
                    for c in range(n_cols):
                        block = arr[r * ctu_h:(r + 1) * ctu_h,
                                    c * ctu_w:(c + 1) * ctu_w]
                        pooled[r, c] = float(block.max() if block.size else 0.0)
                return pooled
    return None


def _percentile_norm(arr: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> np.ndarray:
    lo = float(np.percentile(arr, p_lo))
    hi = float(np.percentile(arr, p_hi))
    return np.clip((arr - lo) / max(1e-9, hi - lo), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Surrogate evaluation.
# ---------------------------------------------------------------------------

def _delta_rate_kbps(K_grid: np.ndarray, q_base: int, delta: np.ndarray,
                      residual_scale: float, fps: float, n_frames_total: int) -> float:
    """Sum K_c · (2^{-(Q_b+δ)/6} - 2^{-Q_b/6}) and convert to kbps."""
    base = 2.0 ** (-q_base / 6.0)
    pert = 2.0 ** (-(q_base + delta) / 6.0)
    delta_bits = float((residual_scale * K_grid * (pert - base)).sum())
    return delta_bits * fps / max(1, n_frames_total) / 1000.0


def _delta_map_local(phi_oracle_norm: np.ndarray, delta: np.ndarray,
                      eta: float = 0.06, xi: float = 0.15) -> float:
    """Asymmetric task model: high-QP corrupts, low-QP only mildly improves."""
    pos = np.maximum(0, delta)
    neg = np.maximum(0, -delta)
    return float(-(phi_oracle_norm * eta * pos).sum()
                 + (phi_oracle_norm * eta * xi * neg).sum())


# ---------------------------------------------------------------------------
# Sample-per-cell oracle assembly.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 surrogate-driven oracle builder")
    parser.add_argument("--rate-npz", required=True,
                        help="Output of rate_surrogate.py")
    parser.add_argument("--saliency-dir", required=True,
                        help="Directory of phi_oracle*.npy from occlusion_saliency.py")
    parser.add_argument("--phi-ipf-dir", default="",
                        help="Phase 1 field tensor directory (optional; required for "
                             "level 1+ features beyond Φ_oracle)")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--qp-list", nargs="+", type=int, default=[27, 32, 37, 42])
    parser.add_argument("--delta-list", nargs="+", type=int,
                        default=[-8, -6, -4, -2, 0, 2, 4])
    parser.add_argument("--samples-per-cell", type=int, default=200,
                        help="CTU samples per (frame, QP, δ) combination")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--eta", type=float, default=0.06,
                        help="Task sensitivity per QP step (mAP units).")
    parser.add_argument("--xi", type=float, default=0.15,
                        help="Asymmetry: gain ratio for δ<0 vs δ>0.")
    parser.add_argument("--output", required=True,
                        help="Output JSONL (or .parquet if pyarrow available)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rate_data = np.load(Path(args.rate_npz).expanduser().resolve())
    K_grids = rate_data["K_grids"]   # (n_frames, n_rows, n_cols)
    residual_qps = rate_data["residuals_qp"].tolist()
    residual_scales = rate_data["residuals_scale"].tolist()
    residual_map = {int(q): float(s) for q, s in zip(residual_qps, residual_scales)}
    n_frames, n_rows, n_cols = K_grids.shape
    logger.info("Rate surrogate loaded: %d frames, %dx%d CTU grid",
                n_frames, n_rows, n_cols)

    saliency_dir = Path(args.saliency_dir).expanduser().resolve()
    phi_ipf_dir = Path(args.phi_ipf_dir).expanduser().resolve() if args.phi_ipf_dir else None

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total_ctus_per_frame = n_rows * n_cols

    n_written = 0
    out_path_jsonl = out_path
    if out_path.suffix.lower() == ".parquet":
        # Buffer all rows then convert at the end.
        buffer: List[dict] = []
    else:
        buffer = None
        out_path_jsonl = out_path

    if buffer is None:
        f_jsonl = open(out_path_jsonl, "w", encoding="utf-8")
    else:
        f_jsonl = None

    try:
        for fi in range(n_frames):
            phi_ora = np.load(saliency_dir / f"phi_oracle_{fi:06d}.npy")
            phi_ora_norm = _percentile_norm(phi_ora)
            if phi_ora.shape != (n_rows, n_cols):
                logger.warning("frame %d: saliency shape %s != grid %s — skipping",
                               fi, phi_ora.shape, (n_rows, n_cols))
                continue
            phi_ipf = (_load_phi_grid(phi_ipf_dir, fi, n_rows, n_cols)
                       if phi_ipf_dir else None)
            if phi_ipf is None:
                phi_ipf = phi_ora_norm.copy()
                phi_ipf_used = "FALLBACK_oracle"
            else:
                phi_ipf = _percentile_norm(phi_ipf)
                phi_ipf_used = "ipf"

            phi_max = phi_ipf
            phi_sum = phi_ipf
            phi_l2 = phi_ipf
            phi_l4 = phi_ipf

            for qp in args.qp_list:
                resid = residual_map.get(int(qp), 1.0)
                # Per-frame total: distribute the per-frame share so the surrogate
                # totals match the calibration target.
                K_grid = K_grids[fi]
                for delta_val in args.delta_list:
                    n_samples = min(args.samples_per_cell, n_total_ctus_per_frame)
                    flat_indices = rng.choice(n_total_ctus_per_frame,
                                              size=n_samples, replace=False)
                    rows = flat_indices // n_cols
                    cols = flat_indices % n_cols
                    delta_grid_local = np.zeros((n_rows, n_cols), dtype=np.int32)
                    delta_grid_local[rows, cols] = delta_val
                    # Frame-level surrogate predictions for this perturbation.
                    delta_kbps = _delta_rate_kbps(
                        K_grid, qp, delta_grid_local.astype(np.float64),
                        residual_scale=resid,
                        fps=args.fps, n_frames_total=n_frames,
                    )
                    delta_map = _delta_map_local(
                        phi_ora_norm * (delta_grid_local != 0).astype(np.float64),
                        delta_grid_local.astype(np.float64),
                        eta=args.eta, xi=args.xi,
                    )

                    # Per-CTU rows (the optimizer needs per-CTU features).
                    for r, c in zip(rows.tolist(), cols.tolist()):
                        feat = [
                            float(phi_max[r, c]),
                            float(phi_sum[r, c]),
                            float(phi_l2[r, c]),
                            float(phi_l4[r, c]),
                            float(qp),
                            float(phi_ora_norm[r, c]),     # extra: oracle saliency
                            float(K_grid[r, c]),           # extra: rate sensitivity
                        ]
                        # Per-CTU ΔR / ΔmAP attributable to this single CTU only.
                        d_r = (resid * K_grid[r, c]
                               * (2.0 ** (-(qp + delta_val) / 6.0)
                                  - 2.0 ** (-qp / 6.0))) * args.fps / 1000.0 / max(1, n_frames)
                        d_m = (-(phi_ora_norm[r, c] * args.eta * max(0, delta_val))
                               + (phi_ora_norm[r, c] * args.eta * args.xi
                                  * max(0, -delta_val)))
                        row = {
                            "features": feat,
                            "delta": int(delta_val),
                            "delta_rate_kbps": float(d_r),
                            "delta_map50": float(d_m),
                            "sequence": args.sequence,
                            "qp_base": int(qp),
                            "frame_idx": int(fi),
                            "row": int(r),
                            "col": int(c),
                            "frame_delta_rate_kbps": float(delta_kbps),
                            "frame_delta_map": float(delta_map),
                            "phi_source": phi_ipf_used,
                        }
                        if buffer is not None:
                            buffer.append(row)
                        else:
                            f_jsonl.write(json.dumps(row) + "\n")
                        n_written += 1
        logger.info("Wrote %d oracle rows", n_written)
    finally:
        if f_jsonl is not None:
            f_jsonl.close()

    if buffer is not None:
        # Convert to parquet at the end.
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # noqa: BLE001
            logger.error("pyarrow missing (%s) — falling back to JSONL", exc)
            jsonl_path = out_path.with_suffix(".jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for row in buffer:
                    f.write(json.dumps(row) + "\n")
            logger.info("Wrote JSONL fallback %s", jsonl_path)
        else:
            t = pa.Table.from_pylist(buffer)
            pq.write_table(t, out_path)
            logger.info("Wrote Parquet %s", out_path)


if __name__ == "__main__":
    main()
