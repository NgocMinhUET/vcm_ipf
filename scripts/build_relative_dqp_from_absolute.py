"""Convert Phase 1 absolute QP maps -> relative dQP maps for fair comparison.

Why this script exists
----------------------
Phase 1 generated absolute QP maps for M1/M3/M5/M6 (e.g.
``~/Minh/ipf/phase1_outputs/multi_seq_<seq>/<method>/qp_vtm/qp_NNNNNN.txt``)
where every CTU has an absolute integer QP value clustered around the
generation centre (~33-35). At pilot encoding QP_base ∈ {27, 32, 37, 42},
the *absolute* maps mean every CTU receives a near-constant QP regardless
of the desired Q_base — this destroys the differential ROI allocation
(the very motivation of pilot_v1's calibration mismatch). M4-LiteQP-CNN
(pilot_v8b) sidesteps this by using *relative* dQP per Q_base.

For a fair head-to-head with M4 (pilot_v10_bench), every adaptive method
must use the SAME representation. This script converts absolute maps to
the standardised relative form:

    dQP_relative(c) = round(QP_abs(c) - mean(QP_abs))

then clips to [-8, +4] (VVC legal range) and writes a
``qp_vtm_delta/qp_NNNNNN.txt`` file with ``type=delta`` header. The
phase 2 encoder wrapper composes the final QP at run-time::

    QP_final(c) = clip(Q_base + dQP_relative(c), 1, 51)

Optionally (``--rate-neutral``) we apply a single scalar shift ``s`` so
that ``Σ_c K_c · 2^{-(δ_c + s)/6} = Σ_c K_c`` — the same closed-form
projection used by the analytic A+ pipeline. This keeps the BD-Rate-Task
comparison apples-to-apples (no method gets a free rate boost).

Usage
-----
::

    PYTHONPATH=src python -m phase2.scripts.build_relative_dqp_from_absolute \\
        --input-dir ~/Minh/ipf/phase1_outputs/multi_seq_MOT17-04-DPM/M5/qp_vtm \\
        --output-dir ~/Minh/ipf/phase1_outputs/multi_seq_MOT17-04-DPM/M5/qp_vtm_delta \\
        --delta-min -8 --delta-max 4

For batch mode across all (sequence, method) pairs, use the
``run_build_relative_dqp.sh`` helper or wrap this in a shell loop.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger("phase2.scripts.build_relative_dqp_from_absolute")


# ---------------------------------------------------------------------------
# File parsing — mirror phase1.export.qp_exporter format
# ---------------------------------------------------------------------------

def _parse_absolute_qp_file(path: Path) -> Tuple[np.ndarray, dict]:
    """Read a Phase 1 absolute-QP file. Returns (grid, header_meta)."""
    text = path.read_text(encoding="utf-8").splitlines()
    if not text:
        raise ValueError(f"empty file: {path}")
    header = text[0].lstrip("# ").strip()
    meta = {}
    for tok in header.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            meta[k] = v
    if meta.get("type", "absolute") != "absolute":
        raise ValueError(
            f"{path} is not an absolute-QP file (type={meta.get('type')})"
        )
    n_rows = int(meta["rows"]); n_cols = int(meta["cols"])
    grid = np.zeros((n_rows, n_cols), dtype=np.int32)
    for r, line in enumerate(text[1:1 + n_rows]):
        toks = line.split()
        if len(toks) != n_cols:
            raise ValueError(
                f"{path}:{r+2}: expected {n_cols} values, got {len(toks)}"
            )
        for c, t in enumerate(toks):
            grid[r, c] = int(t)
    return grid, meta


def _write_delta_qp_file(
    delta: np.ndarray,
    path: Path,
    frame_idx: int,
    delta_min: int,
    delta_max: int,
) -> None:
    clipped = np.clip(np.rint(delta), delta_min, delta_max).astype(np.int32)
    n_rows, n_cols = clipped.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            f"# frame={frame_idx} rows={n_rows} cols={n_cols} "
            f"type=delta delta_min={delta_min} delta_max={delta_max}\n"
        )
        for r in range(n_rows):
            fh.write(" ".join(f"{clipped[r, c]:+d}" for c in range(n_cols)))
            fh.write("\n")


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _frame_idx_from_path(path: Path) -> int:
    stem = path.stem  # e.g. qp_000045
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def _absolute_to_relative(
    grid_abs: np.ndarray,
    rate_neutral: bool,
    K_grid: np.ndarray | None,
) -> np.ndarray:
    """Centre the absolute grid on its mean (or, if K is given, K-weighted mean).

    With ``rate_neutral`` we additionally apply the closed-form scalar
    shift ``s = 6 · log_2(Σ K · 2^{-δ/6} / Σ K)`` so that the resulting
    delta map has zero R-λ rate excess in expectation.
    """
    grid = grid_abs.astype(np.float64)
    if K_grid is None:
        centre = float(grid.mean())
        delta = grid - centre
    else:
        K = np.asarray(K_grid, dtype=np.float64)
        K = np.maximum(K, 1e-9)
        centre = float(np.sum(K * grid) / np.sum(K))
        delta = grid - centre

    if rate_neutral and K_grid is not None:
        K = np.asarray(K_grid, dtype=np.float64)
        denom = float(np.sum(K) + 1e-9)
        weighted = float(np.sum(K * np.power(2.0, -delta / 6.0)))
        ratio = max(weighted / denom, 1e-12)
        s = 6.0 * np.log2(ratio)
        delta = delta + s

    return delta


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    rate_neutral: bool = False,
    K_npz: Path | None = None,
    delta_min: int = -8,
    delta_max: int = 4,
) -> int:
    """Convert every absolute-QP file in ``input_dir`` to relative dQP."""
    in_files = sorted(input_dir.glob("qp_*.txt"))
    if not in_files:
        raise FileNotFoundError(f"No qp_*.txt files in {input_dir}")

    K_per_frame = None
    if K_npz is not None and K_npz.exists():
        loaded = np.load(K_npz)
        if "K_per_frame" in loaded.files:
            K_per_frame = loaded["K_per_frame"]
        elif "K" in loaded.files:
            K_per_frame = loaded["K"]
        if K_per_frame is not None:
            logger.info("Loaded K from %s, shape %s", K_npz, K_per_frame.shape)

    n_written = 0
    for in_path in in_files:
        grid_abs, _ = _parse_absolute_qp_file(in_path)
        fi = _frame_idx_from_path(in_path)
        K_grid = (K_per_frame[fi] if (K_per_frame is not None
                                       and fi < len(K_per_frame))
                  else None)
        delta = _absolute_to_relative(grid_abs, rate_neutral, K_grid)
        out_path = output_dir / in_path.name
        _write_delta_qp_file(
            delta, out_path, fi,
            delta_min=delta_min, delta_max=delta_max,
        )
        n_written += 1

    logger.info("Wrote %d delta-QP files to %s", n_written, output_dir)
    return n_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert absolute QP maps -> relative dQP maps"
    )
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Phase 1 dir containing qp_NNNNNN.txt absolute files")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Destination dir for qp_NNNNNN.txt delta files")
    parser.add_argument("--rate-neutral", action="store_true",
                        help="Apply closed-form rate-neutral projection (requires --K-npz)")
    parser.add_argument("--K-npz", type=Path, default=None,
                        help="Optional rate_surrogate.npz with K_per_frame")
    parser.add_argument("--delta-min", type=int, default=-8)
    parser.add_argument("--delta-max", type=int, default=4)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if args.rate_neutral and args.K_npz is None:
        raise SystemExit("--rate-neutral requires --K-npz with K_per_frame")

    convert_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        rate_neutral=args.rate_neutral,
        K_npz=args.K_npz,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
    )


if __name__ == "__main__":
    main()
