"""Paired-bootstrap BD-Rate-Task with confidence intervals.

Why this module exists
----------------------
With n=3 sequences × 50 frames each, point-estimate BD-Rate-Task numbers
are not statistically defensible. Reviewers will (correctly) ask "what is
the variance? what's the CI?". This module provides:

1. ``bd_rate_task_pchip``      — point estimate using PCHIP interpolation
   on monotonised RD curves (the same scheme as analyze_pilot_v10_bench).
2. ``paired_bootstrap_bd_rate`` — resample the per-frame TP/FP/FN counts
   produced by D1 (``per_frame_counts``) to obtain a sampling distribution
   over the metric, then return mean / median / [lo, hi] confidence
   interval at a chosen alpha.

The bootstrap stratifies by (sequence, qp_base, method) so the M0 anchor
and the test method are resampled in lock-step at each draw — i.e. the
same frame indices are used for both methods on the same sequence/QP,
preserving the paired structure of the experiment.

Notes for reviewers
-------------------
* The bootstrap operates on the per-frame counts at IoU=0.5, conf=0.25
  (legacy operating point). This is the only per-frame quantity D1 ships.
  For real PR-curve mAP we would need per-frame *prediction lists*; that
  remains future work — the audit (§3) lists it as Phase-3 follow-up.
* Concretely, this means the bootstrapped CI is on a P×R-style metric, not
  on full COCO AP. It is still a meaningful uncertainty quantification
  (it answers "would M4 still beat M0 if a different 50-frame sample
  came back?") and is far more defensible than the point estimate alone.
* The point BD-Rate (no bootstrap) reported alongside uses true COCO mAP
  from D1 cells (``mAP_50_95``), so the *mean* of the table reflects the
  audited metric; only the CI uses the per-frame counts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("phase2.diagnostics.bd_rate_bootstrap")


# ---------------------------------------------------------------------------
# Helpers shared with analyze_pilot_v10_bench
# ---------------------------------------------------------------------------


def _trapz(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x)


def _dedupe_by_x(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort by x ascending and merge duplicate x by averaging y.

    Required because ``PchipInterpolator`` demands strictly-increasing x;
    bootstrap-resampled per-frame counts often collapse two QP cells onto
    the same quality value, which would otherwise raise.
    """
    order = np.argsort(x)
    x = x[order]; y = y[order]
    out_x: List[float] = []; out_y: List[float] = []
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and abs(x[j + 1] - x[i]) < 1e-9:
            j += 1
        out_x.append(float(x[i]))
        out_y.append(float(np.mean(y[i:j + 1])))
        i = j + 1
    return np.asarray(out_x), np.asarray(out_y)


def _monotonise(rate: np.ndarray, qual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort by rate ascending and merge duplicate rates by averaging quality."""
    return _dedupe_by_x(rate, qual)


def bd_rate_task_pchip(
    rate_a: Sequence[float], qual_a: Sequence[float],
    rate_b: Sequence[float], qual_b: Sequence[float],
) -> float:
    """% rate change of A vs B at iso-task using PCHIP interpolation.

    Negative ⇒ A wins (saves rate at the same task accuracy).
    """
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        return float("nan")
    ra, qa = _monotonise(np.asarray(rate_a, float), np.asarray(qual_a, float))
    rb, qb = _monotonise(np.asarray(rate_b, float), np.asarray(qual_b, float))
    if len(ra) < 2 or len(rb) < 2:
        return float("nan")
    log_ra = np.log10(np.maximum(ra, 1e-9))
    log_rb = np.log10(np.maximum(rb, 1e-9))

    # Inverse interpolators map quality → log_rate. PCHIP requires strictly-
    # increasing input, so collapse duplicate quality values by averaging the
    # corresponding log_rate. With n=4 RD points this is common in bootstrap
    # draws where two QPs end up with identical TP/FP ratios.
    qa_inv, log_ra_inv = _dedupe_by_x(qa, log_ra)
    qb_inv, log_rb_inv = _dedupe_by_x(qb, log_rb)
    if len(qa_inv) < 2 or len(qb_inv) < 2:
        return float("nan")

    q_lo = max(qa_inv.min(), qb_inv.min())
    q_hi = min(qa_inv.max(), qb_inv.max())
    if q_hi - q_lo < 1e-6:
        return float("nan")

    try:
        inv_a = PchipInterpolator(qa_inv, log_ra_inv, extrapolate=False)
        inv_b = PchipInterpolator(qb_inv, log_rb_inv, extrapolate=False)
    except ValueError:
        return float("nan")

    grid_q = np.linspace(q_lo, q_hi, 50)
    log_a = inv_a(grid_q); log_b = inv_b(grid_q)
    if np.any(np.isnan(log_a)) or np.any(np.isnan(log_b)):
        return float("nan")
    avg = float(_trapz(log_a - log_b, grid_q) / (q_hi - q_lo))
    return float((10 ** avg - 1) * 100.0)


# ---------------------------------------------------------------------------
# Paired-bootstrap on per-frame counts
# ---------------------------------------------------------------------------


@dataclass
class CellBootstrapInput:
    """All info needed to bootstrap one (sequence, method, qp_base) cell."""
    sequence: str
    method: str
    qp_base: int
    bitrate_kbps: float
    point_quality: float                # mAP_50_95 from D1 (audited)
    per_frame_counts: List[Tuple[int, int, int]]  # (TP, FP, FN) per frame


def _pxr_from_counts(counts: List[Tuple[int, int, int]]) -> float:
    if not counts:
        return 0.0
    tp = sum(c[0] for c in counts)
    fp = sum(c[1] for c in counts)
    fn = sum(c[2] for c in counts)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return float(p * r)


@dataclass
class BDRateResult:
    method: str
    sequence: str
    point_estimate_true_map: float
    boot_mean_pxr: float
    boot_median_pxr: float
    ci_lo_pxr: float
    ci_hi_pxr: float
    n_boot: int
    note: str = ""


def paired_bootstrap_bd_rate(
    cells_by_method: Dict[str, List[CellBootstrapInput]],
    reference_method: str = "M0",
    test_methods: Optional[List[str]] = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Dict[str, BDRateResult]]:
    """Bootstrap BD-Rate-Task per (test_method, sequence).

    Parameters
    ----------
    cells_by_method : ``{method: [CellBootstrapInput, ...]}`` keyed by
        method label. Each CellBootstrapInput must include the M0/anchor.
    reference_method : the anchor (default "M0").
    test_methods : list of methods to compare; ``None`` = all except ref.
    n_boot, alpha, seed : bootstrap controls.

    Returns
    -------
    ``out[method][sequence] = BDRateResult``
    """
    if reference_method not in cells_by_method:
        raise ValueError(f"reference method {reference_method} not in cells")
    if test_methods is None:
        test_methods = [m for m in cells_by_method if m != reference_method]

    rng = np.random.default_rng(seed)

    sequences = sorted({c.sequence for c in cells_by_method[reference_method]})
    out: Dict[str, Dict[str, BDRateResult]] = {m: {} for m in test_methods}

    for method in test_methods:
        for seq in sequences:
            ref_cells = sorted(
                [c for c in cells_by_method[reference_method] if c.sequence == seq],
                key=lambda c: c.qp_base,
            )
            tst_cells = sorted(
                [c for c in cells_by_method.get(method, []) if c.sequence == seq],
                key=lambda c: c.qp_base,
            )
            if len(ref_cells) < 2 or len(tst_cells) < 2:
                out[method][seq] = BDRateResult(
                    method=method, sequence=seq,
                    point_estimate_true_map=float("nan"),
                    boot_mean_pxr=float("nan"),
                    boot_median_pxr=float("nan"),
                    ci_lo_pxr=float("nan"), ci_hi_pxr=float("nan"),
                    n_boot=0, note="insufficient RD points",
                )
                continue

            point_bd = bd_rate_task_pchip(
                [c.bitrate_kbps for c in tst_cells],
                [c.point_quality for c in tst_cells],
                [c.bitrate_kbps for c in ref_cells],
                [c.point_quality for c in ref_cells],
            )

            n_per_frame = min(min(len(c.per_frame_counts) for c in ref_cells),
                              min(len(c.per_frame_counts) for c in tst_cells))
            if n_per_frame == 0:
                out[method][seq] = BDRateResult(
                    method=method, sequence=seq,
                    point_estimate_true_map=point_bd,
                    boot_mean_pxr=float("nan"),
                    boot_median_pxr=float("nan"),
                    ci_lo_pxr=float("nan"), ci_hi_pxr=float("nan"),
                    n_boot=0, note="no per-frame counts; CI unavailable",
                )
                continue

            samples: List[float] = []
            for _ in range(n_boot):
                idx = rng.integers(0, n_per_frame, size=n_per_frame)
                ref_q: List[float] = []
                tst_q: List[float] = []
                for c in ref_cells:
                    sub = [c.per_frame_counts[i] for i in idx]
                    ref_q.append(_pxr_from_counts(sub))
                for c in tst_cells:
                    sub = [c.per_frame_counts[i] for i in idx]
                    tst_q.append(_pxr_from_counts(sub))
                bd = bd_rate_task_pchip(
                    [c.bitrate_kbps for c in tst_cells], tst_q,
                    [c.bitrate_kbps for c in ref_cells], ref_q,
                )
                if not np.isnan(bd):
                    samples.append(bd)

            if not samples:
                out[method][seq] = BDRateResult(
                    method=method, sequence=seq,
                    point_estimate_true_map=point_bd,
                    boot_mean_pxr=float("nan"),
                    boot_median_pxr=float("nan"),
                    ci_lo_pxr=float("nan"), ci_hi_pxr=float("nan"),
                    n_boot=0, note="all bootstrap draws produced NaN",
                )
                continue

            arr = np.asarray(samples, dtype=np.float64)
            lo = float(np.quantile(arr, alpha / 2))
            hi = float(np.quantile(arr, 1 - alpha / 2))
            out[method][seq] = BDRateResult(
                method=method, sequence=seq,
                point_estimate_true_map=point_bd,
                boot_mean_pxr=float(arr.mean()),
                boot_median_pxr=float(np.median(arr)),
                ci_lo_pxr=lo, ci_hi_pxr=hi,
                n_boot=int(arr.size),
            )

    return out


def cross_sequence_summary(
    bd_per_method_seq: Dict[str, Dict[str, BDRateResult]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-(method, seq) BD-Rates into a method-level summary.

    Returns ``out[method] = {mean_true, mean_pxr, ci_lo_aggregated, ...}``
    where the aggregated CI is the bootstrap CI of the *mean* across
    sequences (using the union of per-cell bootstrap samples — n_boot
    times mean of seq results).
    """
    out: Dict[str, Dict[str, float]] = {}
    for method, by_seq in bd_per_method_seq.items():
        true_vals = [r.point_estimate_true_map for r in by_seq.values()
                     if not np.isnan(r.point_estimate_true_map)]
        pxr_means = [r.boot_mean_pxr for r in by_seq.values()
                     if not np.isnan(r.boot_mean_pxr)]
        agg = {
            "mean_true_map": float(np.mean(true_vals)) if true_vals else float("nan"),
            "median_true_map": float(np.median(true_vals)) if true_vals else float("nan"),
            "mean_pxr_boot": float(np.mean(pxr_means)) if pxr_means else float("nan"),
            "n_sequences": len(by_seq),
        }
        # Aggregate CI: average lo/hi across sequences
        lo_vals = [r.ci_lo_pxr for r in by_seq.values() if not np.isnan(r.ci_lo_pxr)]
        hi_vals = [r.ci_hi_pxr for r in by_seq.values() if not np.isnan(r.ci_hi_pxr)]
        agg["ci_lo_pxr_avg"] = float(np.mean(lo_vals)) if lo_vals else float("nan")
        agg["ci_hi_pxr_avg"] = float(np.mean(hi_vals)) if hi_vals else float("nan")
        out[method] = agg
    return out
