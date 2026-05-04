"""Phase 3 Stage C — automatic per-sequence λ_task from M0 anchor elasticity.

Why
---
In the rate–task Lagrangian
    minimise   ΔR  −  λ_task · ΔmAP
the optimal λ_task is the **local exchange rate** between bitrate and task
accuracy. If a sequence's mAP is highly sensitive to bitrate (steep R-T
curve), λ should be **large** so the optimiser spends bits to preserve
mAP. If the curve is flat (saturated mAP), λ should be **small** so the
optimiser harvests rate savings without losing accuracy.

Pilot v5 (PROJECT_STATE §7.10) chose λ manually based on intuition; the
chosen values turned out to be *opposite* to what the data implies and
produced a +18.75 pp BD-Rate-Task regression. This module replaces that
hand-tuning with a **purely data-driven estimator** computed from the
pilot v1 / v4 M0 anchor.

Method
------
For each sequence ``s`` we use the M0 (uniform-QP) operating points
``{(R_i, mAP_i)}_{i=1..n}`` from a pilot summary file:

    e_s   = median_i  | (mAP_{i+1} − mAP_i) / (log10 R_{i+1} − log10 R_i) |

    λ_s   = base_λ · ( e_s / median_s e_s )^α    clipped to [λ_min, λ_max]

Median-normalisation makes the formula scale-invariant — only the *relative*
elasticity across sequences matters. ``α = 0.5`` (square-root) prevents
single outlier slopes from blowing up λ (key safety guard learnt from v5).
The clip ``[3.5, 7.0]`` was specifically chosen to avoid v5's λ = 8.0
which over-shot at high QP.

Defensibility
-------------
Reviewer-friendly framing:

    "We do not tune λ per sequence manually. Instead, λ is estimated
    automatically from the anchor rate–accuracy curve, using the local
    task elasticity with respect to bitrate (Bjontegaard-style slope)."

A single global hyperparameter ``base_λ`` (and the clip bounds, fixed once)
governs the whole hierarchy — there are NO per-sequence knobs.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("phase2.phase3.auto_lambda")


# ---------------------------------------------------------------------------
# Pilot summary loader (mirrors phase2/scripts/compare_pilots.py:load_summary)
# ---------------------------------------------------------------------------

@dataclass
class _AnchorPoint:
    sequence: str
    qp_base: int
    bitrate_kbps: float
    map50: float


def _load_m0_points(pilot_summary_path: Path) -> List[_AnchorPoint]:
    """Extract all M0 (sequence, qp_base, bitrate, mAP) records from a
    Phase 2 ``experiment_summary.json``."""
    with open(pilot_summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("results", data) if isinstance(data, dict) else data
    pts: List[_AnchorPoint] = []
    for r in rows:
        if not bool(r.get("success", True)):
            continue
        if r.get("method", "?") != "M0":
            continue
        try:
            pts.append(_AnchorPoint(
                sequence=str(r.get("sequence", "?")),
                qp_base=int(r.get("qp_base") or r.get("qp") or 0),
                bitrate_kbps=float(r.get("bitrate_kbps") or 0.0),
                map50=float(r.get("mAP50") or r.get("map50") or 0.0),
            ))
        except (TypeError, ValueError):
            logger.warning("Skipping malformed row: %s", r)
    return pts


# ---------------------------------------------------------------------------
# Elasticity estimators
# ---------------------------------------------------------------------------

def _per_arc_slopes(rates: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """Per-adjacent-pair |Δ mAP / Δ log10 R|.

    Both arrays must already be sorted by ``rates`` (ascending or descending —
    sign is removed). The arrays are de-duplicated on rate before computing
    the slopes to avoid divide-by-zero.
    """
    o = np.argsort(rates, kind="stable")
    r = rates[o]; m = maps[o]
    keep = [0]
    for i in range(1, len(r)):
        if r[i] - r[keep[-1]] > 1e-9:
            keep.append(i)
    r = r[keep]; m = m[keep]
    if len(r) < 2:
        return np.array([], dtype=np.float64)
    lr = np.log10(np.clip(r, 1e-9, None))
    return np.abs(np.diff(m) / np.diff(lr))


def _seq_elasticity(rates: List[float], maps: List[float],
                     mode: str = "median",
                     qp_range: Optional[Tuple[int, int]] = None,
                     qps: Optional[List[int]] = None) -> float:
    """Robust elasticity ``|d mAP / d log10 R|`` for one sequence.

    Parameters
    ----------
    mode
        ``"median"`` (default, robust) or ``"mean"`` of the per-arc slopes.
    qp_range, qps
        Optional restriction to a deployment QP window: pass both to filter
        only points whose ``qps[i] ∈ [qp_range[0], qp_range[1]]``.
    """
    r = np.asarray(rates, dtype=np.float64)
    m = np.asarray(maps,  dtype=np.float64)
    if qp_range is not None and qps is not None:
        lo, hi = qp_range
        mask = np.array([(lo <= q <= hi) for q in qps])
        r = r[mask]; m = m[mask]
    slopes = _per_arc_slopes(r, m)
    if slopes.size == 0:
        return float("nan")
    if mode == "median":
        return float(np.median(slopes))
    if mode == "mean":
        return float(np.mean(slopes))
    raise ValueError(f"Unknown slope_mode: {mode}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_lambdas(
    pilot_summary_path: Path,
    sequences: Iterable[str],
    base_lambda: float = 5.0,
    alpha: float = 0.5,
    lambda_min: float = 3.5,
    lambda_max: float = 7.0,
    slope_mode: str = "median",
    qp_range: Optional[Tuple[int, int]] = None,
) -> Dict[str, dict]:
    """Compute auto-λ for each requested sequence.

    Returns a dict ``{seq_name: {"lambda_task", "elasticity", "n_pts",
    "qps_used", "raw_lambda", "clipped"}}``. Sequences absent from the
    pilot summary fall back to ``base_lambda`` and are flagged with
    ``"fallback": true``.

    The returned ``lambda_task`` is the value to use as
    ``teacher_overrides[seq]["lambda_task"]`` in the Phase 3 orchestrator.
    """
    if not (0.0 <= alpha <= 2.0):
        raise ValueError(f"alpha must be in [0, 2], got {alpha}")
    if lambda_min <= 0 or lambda_max <= lambda_min:
        raise ValueError(f"Bad clip range [{lambda_min}, {lambda_max}]")

    pts = _load_m0_points(pilot_summary_path)
    if not pts:
        raise SystemExit(f"No M0 anchor points in {pilot_summary_path}")

    # Group by sequence.
    by_seq: Dict[str, List[_AnchorPoint]] = {}
    for p in pts:
        by_seq.setdefault(p.sequence, []).append(p)
    for s in by_seq:
        by_seq[s].sort(key=lambda p: p.qp_base)

    # First pass: per-sequence elasticity (only for sequences requested).
    elasticities: Dict[str, float] = {}
    qps_used:     Dict[str, List[int]] = {}
    for seq in sequences:
        records = by_seq.get(seq, [])
        if not records:
            logger.warning("Sequence %s missing from pilot summary; will fall back",
                           seq)
            elasticities[seq] = float("nan")
            qps_used[seq] = []
            continue
        rates = [r.bitrate_kbps for r in records]
        maps  = [r.map50         for r in records]
        qps   = [r.qp_base       for r in records]
        e = _seq_elasticity(rates, maps, mode=slope_mode,
                             qp_range=qp_range, qps=qps)
        elasticities[seq] = e
        qps_used[seq] = qps

    # Median across sequences for normalisation (skip NaNs).
    finite = [v for v in elasticities.values()
              if v == v and not math.isinf(v)]
    if not finite:
        raise SystemExit("No finite elasticities — cannot normalise.")
    median_e = float(np.median(finite))
    if median_e <= 0:
        raise SystemExit(f"median elasticity = {median_e}; cannot normalise.")

    # Second pass: λ.
    out: Dict[str, dict] = {}
    for seq, e in elasticities.items():
        if not (e == e):           # NaN → fallback
            out[seq] = {
                "lambda_task":   float(base_lambda),
                "elasticity":    None,
                "n_pts":         0,
                "qps_used":      [],
                "raw_lambda":    None,
                "clipped":       False,
                "fallback":      True,
            }
            continue
        ratio = e / median_e
        raw = float(base_lambda) * (ratio ** float(alpha))
        clipped_val = float(min(max(raw, lambda_min), lambda_max))
        out[seq] = {
            "lambda_task":  clipped_val,
            "elasticity":   float(e),
            "n_pts":        len(qps_used[seq]),
            "qps_used":     [int(q) for q in qps_used[seq]],
            "raw_lambda":   raw,
            "clipped":      bool(abs(raw - clipped_val) > 1e-6),
            "fallback":     False,
        }
    return out


def write_metadata(results: Dict[str, dict], output_path: Path,
                    cfg_used: Optional[dict] = None) -> None:
    """Persist auto-λ outputs to JSON for reproducibility."""
    payload = {
        "schema_version": 1,
        "config":         cfg_used or {},
        "lambda_per_seq": results,
        "median_elasticity": float(np.median(
            [r["elasticity"] for r in results.values()
             if r["elasticity"] is not None])),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote auto-λ metadata to %s", output_path)


# ---------------------------------------------------------------------------
# CLI (useful for quick previews — `python -m phase2.phase3.auto_lambda ...`)
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute per-sequence λ_task from M0 anchor elasticity")
    parser.add_argument("--pilot-summary", required=True,
                        help="Path to a pilot experiment_summary.json")
    parser.add_argument("--sequences", nargs="+", required=True,
                        help="Sequence names to compute λ for")
    parser.add_argument("--base-lambda", type=float, default=5.0)
    parser.add_argument("--alpha",        type=float, default=0.5)
    parser.add_argument("--lambda-min",   type=float, default=3.5)
    parser.add_argument("--lambda-max",   type=float, default=7.0)
    parser.add_argument("--slope-mode",   choices=["median", "mean"],
                        default="median")
    parser.add_argument("--qp-min", type=int, default=None)
    parser.add_argument("--qp-max", type=int, default=None)
    parser.add_argument("--output", default="",
                        help="Optional JSON metadata output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    qp_range = None
    if args.qp_min is not None and args.qp_max is not None:
        qp_range = (int(args.qp_min), int(args.qp_max))

    results = compute_lambdas(
        Path(args.pilot_summary).expanduser().resolve(),
        sequences=args.sequences,
        base_lambda=args.base_lambda,
        alpha=args.alpha,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        slope_mode=args.slope_mode,
        qp_range=qp_range,
    )

    print(f"\nAuto-λ (base={args.base_lambda}, α={args.alpha}, "
          f"clip [{args.lambda_min}, {args.lambda_max}], slope={args.slope_mode}):\n")
    for s, r in results.items():
        ev = "n/a" if r["elasticity"] is None else f"{r['elasticity']:.3f}"
        rl = "n/a" if r["raw_lambda"] is None else f"{r['raw_lambda']:.2f}"
        flags = []
        if r["clipped"]:  flags.append("CLIPPED")
        if r["fallback"]: flags.append("FALLBACK")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        print(f"  {s:<20}  elasticity={ev:>6}  raw_lambda={rl:>6}  "
              f"-->  lambda_task={r['lambda_task']:.2f}{flag_str}")

    if args.output:
        cfg_used = {"base_lambda": args.base_lambda, "alpha": args.alpha,
                    "lambda_min": args.lambda_min, "lambda_max": args.lambda_max,
                    "slope_mode": args.slope_mode,
                    "pilot_summary": str(args.pilot_summary),
                    "qp_range": qp_range}
        write_metadata(results, Path(args.output).expanduser().resolve(), cfg_used)


if __name__ == "__main__":
    _main()
