"""Phase 3 Stage B.5 — Apply the fitted formula θ* to generate dQP maps.

Reference: ``11_PHASE3_RESEARCH_PROTOCOL.md`` §5.

Given the JSON output of :mod:`phase2.phase3.fit_parametric` (which
contains the best-Level name plus its θ vector), this script generates
the per-frame ``qp_*.txt`` delta-QP maps that Phase 2's encoder can
ingest directly. Output layout matches :mod:`phase1.export.qp_exporter`
delta format (header tag ``type=delta``).

The features for each CTU are reconstructed from:

* The aggregated Φ field saved by Phase 1 (used as a single "phi_max"
  proxy in all four Lp slots — see note below).
* Optionally the saliency map (``phi_oracle``) for diagnostics.
* The base QP swept by Phase 2 (passed via ``--qp-list``).

NOTE on Lp features. The current Phase 1 only saves the *aggregated*
field for the chosen ``superposition`` mode. Until we modify Phase 1
to also dump per-object kernels (so that Level 4 can be evaluated with
true phi_sum / phi_l2 / phi_l4), Levels 1/2/3/5 are the operative
choices. Level 4 collapses to Level 2 in this configuration but the
fit_parametric script will still rank it correctly because all
candidate p-values share the same loss when only one aggregator is
visible.

Run with::

    python -m phase2.phase3.apply_formula \
        --theta ~/Minh/ipf/phase3_outputs/fit/MOT17/fit_results.json \
        --phi-ipf-dir ~/Minh/ipf/phase1_outputs_v2/ipf_lp_pinf_MOT17-04-DPM/M4 \
        --output-dir ~/Minh/ipf/phase1_outputs_v3/learned_MOT17-04-DPM/M4 \
        --qp-list 27 32 37 42 \
        --n-frames 50 \
        --ctu-rows 9 --ctu-cols 15
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from phase2.phase3.fit_parametric import (
    ALL_LEVELS,
    FunctionFamily,
    IDX_PHI_MAX,
)

logger = logging.getLogger("phase2.phase3.apply_formula")


def _load_theta(path: Path) -> Tuple[FunctionFamily, np.ndarray, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    best_name = data.get("best_level")
    if not best_name:
        raise SystemExit(f"No 'best_level' key in {path}")
    family = next((lvl for lvl in ALL_LEVELS if lvl.name == best_name), None)
    if family is None:
        raise SystemExit(f"Unknown best_level={best_name!r}")
    theta = None
    for r in data.get("results", []):
        if r["level"] == best_name:
            theta = np.asarray(r["theta"], dtype=np.float64)
            break
    if theta is None:
        raise SystemExit(f"theta missing for level {best_name}")
    return family, theta, data


def _load_phi_field(phi_dir: Path, fi: int, n_rows: int, n_cols: int) -> Optional[np.ndarray]:
    candidates = [
        phi_dir / f"field_{fi:06d}.npy",
        phi_dir / f"phi_{fi:06d}.npy",
        phi_dir / "field" / f"field_{fi:06d}.npy",
    ]
    for path in candidates:
        if path.exists():
            arr = np.load(path).astype(np.float64)
            if arr.ndim == 2 and arr.shape == (n_rows, n_cols):
                return arr
            if arr.ndim == 2:
                # Pixel-resolution → pool by max over CTU blocks.
                ctu_h = max(1, arr.shape[0] // n_rows)
                ctu_w = max(1, arr.shape[1] // n_cols)
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


def _write_delta_map(delta: np.ndarray, output_path: Path, frame_idx: int,
                     delta_min: int = -8, delta_max: int = 4) -> None:
    """Mirror phase1.export.qp_exporter.export_delta_qp_vtm exactly."""
    clipped = np.clip(np.rint(delta), delta_min, delta_max).astype(np.int32)
    n_rows, n_cols = clipped.shape
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# frame={frame_idx} rows={n_rows} cols={n_cols} type=delta "
                f"delta_min={delta_min} delta_max={delta_max}\n")
        for row in range(n_rows):
            line = " ".join(f"{clipped[row, col]:+d}" for col in range(n_cols))
            f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply fitted Phase 3 formula → dQP maps")
    parser.add_argument("--theta", required=True, help="fit_results.json")
    parser.add_argument("--phi-ipf-dir", required=True,
                        help="Phase 1 directory with field_*.npy and qp_vtm_delta/")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write qp_*.txt (delta) maps")
    parser.add_argument("--qp-list", nargs="+", type=int, default=[27, 32, 37, 42])
    parser.add_argument("--n-frames", type=int, default=50)
    parser.add_argument("--ctu-rows", type=int, default=9)
    parser.add_argument("--ctu-cols", type=int, default=15)
    parser.add_argument("--delta-min", type=int, default=-8)
    parser.add_argument("--delta-max", type=int, default=4)
    parser.add_argument("--per-qp", action="store_true",
                        help="Write one subdir per Q_base. Otherwise the formula is "
                             "evaluated at Q_base=32 only (legacy compatibility).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    family, theta, meta = _load_theta(Path(args.theta).expanduser().resolve())
    logger.info("Loaded family=%s  dim=%d  theta=%s",
                family.name, family.dim, theta.tolist())

    phi_dir = Path(args.phi_ipf_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = args.ctu_rows, args.ctu_cols

    qp_targets = args.qp_list if args.per_qp else [32]
    for qp in qp_targets:
        if args.per_qp:
            sub = out_dir / f"qp_vtm_delta_q{qp}"
            sub.mkdir(parents=True, exist_ok=True)
        else:
            sub = out_dir / "qp_vtm_delta"
            sub.mkdir(parents=True, exist_ok=True)

        n_written = 0
        for fi in range(args.n_frames):
            phi = _load_phi_field(phi_dir, fi, n_rows, n_cols)
            if phi is None:
                logger.warning("frame %d: no Φ field found under %s — writing zeros",
                               fi, phi_dir)
                phi = np.zeros((n_rows, n_cols), dtype=np.float64)
            phi_norm = _percentile_norm(phi)
            phi_flat = phi_norm.flatten()

            features = np.column_stack([
                phi_flat,                     # IDX_PHI_MAX
                phi_flat,                     # IDX_PHI_SUM (same in MVP)
                phi_flat,                     # IDX_PHI_L2
                phi_flat,                     # IDX_PHI_L4
                np.full_like(phi_flat, qp),   # IDX_Q_BASE
                np.zeros_like(phi_flat),      # placeholder columns
                np.zeros_like(phi_flat),
            ])

            delta = family.predict(features, theta).reshape(n_rows, n_cols)
            _write_delta_map(delta, sub / f"qp_{fi:06d}.txt", fi,
                             args.delta_min, args.delta_max)
            n_written += 1

        logger.info("Q_base=%s → wrote %d files to %s", qp, n_written, sub)

    # Save a thin metadata file alongside.
    meta_out = out_dir / "phase3_apply_metadata.json"
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump({
            "best_level": family.name,
            "theta": theta.tolist(),
            "fit_summary": meta,
            "n_frames": args.n_frames,
            "qp_targets": qp_targets,
            "delta_clip": [args.delta_min, args.delta_max],
            "per_qp_outputs": bool(args.per_qp),
            "phi_ipf_dir": str(phi_dir),
        }, f, indent=2)
    logger.info("Wrote metadata %s", meta_out)


if __name__ == "__main__":
    main()
