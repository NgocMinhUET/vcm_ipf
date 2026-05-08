"""Diagnostic D2 — Per-cell statistical significance and bootstrap CI.

Why this script exists
----------------------
PROJECT_AUDIT.md §1.6 notes that with n=3 sequences × 50 frames the metric
noise floor (estimated at ±0.02–0.04 mAP) is comparable to observed
M4-vs-M0 differences (±0.005–0.05). Without per-cell paired tests we cannot
distinguish "real win" from "noise".

This script consumes the per-frame ``(tp, fp, fn)`` counts produced by
``d1_true_map.py`` and answers, for every (sequence, QP) cell:

1. Are the per-frame F1 scores of M4 *significantly* different from M0?
   - Paired Wilcoxon signed-rank test (non-parametric, robust to outliers)
   - Paired t-test (parametric, sensitive but standard)
   - Cohen's d effect size on per-frame F1 differences
2. What is the 95% bootstrap CI on (sequence-mean F1_M4 − sequence-mean F1_M0)?

Output is a JSON report with one entry per (sequence, QP) and a markdown
table summarising "real" wins (p < 0.05 AND |Cohen's d| > 0.2) vs "noise".

Per-frame F1 is chosen as the unit of statistical testing because:
* TP/FP/FN are countable per frame (mAP integrates over frames so cannot be
  paired).
* F1 = 2·P·R / (P+R) is a stable summary at IoU=0.5, conf=0.25 — the same
  operating point as the legacy metric and as the eventual deployment.
* Paired tests on F1 differences correctly account for the "same frame, same
  GT" coupling between M0 and M4.

Usage
-----
After D1 has produced ``d1_true_map.json`` for at least M0 and one M4
variant in the same pilot::

    PYTHONPATH=src python -m phase2.diagnostics.d2_statistical_significance \\
        --d1-json ~/Minh/ipf/phase2_outputs/pilot_v8b/diagnostics/d1_true_map.json \\
        --reference-method M0 \\
        --test-methods M4 \\
        --output ~/Minh/ipf/phase2_outputs/pilot_v8b/diagnostics/d2_stats.json

Or pass multiple D1 JSONs to compare across pilots::

    PYTHONPATH=src python -m phase2.diagnostics.d2_statistical_significance \\
        --d1-json pilot_v4/.../d1.json pilot_v8b/.../d1.json pilot_v9/.../d1.json \\
        --reference-method M0 --test-methods M4 \\
        --output combined_d2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("phase2.diagnostics.d2_statistical_significance")

ALPHA = 0.05
SMALL_EFFECT = 0.2  # Cohen's d threshold for "real" effect


# ---------------------------------------------------------------------------
# Per-frame F1 from (tp, fp, fn)
# ---------------------------------------------------------------------------

def _per_frame_f1(counts: List[List[int]]) -> np.ndarray:
    """Convert ``[[tp, fp, fn], ...]`` → per-frame F1 (zeros where undefined)."""
    arr = np.asarray(counts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    tp = arr[:, 0]; fp = arr[:, 1]; fn = arr[:, 2]
    p = tp / np.maximum(tp + fp, 1e-9)
    r = tp / np.maximum(tp + fn, 1e-9)
    f1 = np.where((p + r) > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)
    return f1


# ---------------------------------------------------------------------------
# Statistical tests — pure-numpy fallbacks if scipy is missing
# ---------------------------------------------------------------------------

def _wilcoxon_signed_rank(diffs: np.ndarray) -> Tuple[float, float]:
    """Two-sided Wilcoxon signed-rank. Returns (W, p_approx).

    Falls back to scipy when available; otherwise uses normal approximation
    (valid for n ≥ 20, which is our case at 50 frames).
    """
    try:
        from scipy.stats import wilcoxon
        d = diffs[np.abs(diffs) > 1e-12]
        if d.size == 0:
            return 0.0, 1.0
        try:
            res = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            return float(res.statistic), float(res.pvalue)
        except ValueError:
            return 0.0, 1.0
    except ImportError:
        d = diffs[np.abs(diffs) > 1e-12]
        n = d.size
        if n == 0:
            return 0.0, 1.0
        ranks = np.argsort(np.argsort(np.abs(d))) + 1
        signs = np.sign(d)
        signed = signs * ranks
        w_pos = float(signed[signed > 0].sum())
        w_neg = float(-signed[signed < 0].sum())
        w = min(w_pos, w_neg)
        mu = n * (n + 1) / 4.0
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        if sigma == 0:
            return float(w), 1.0
        z = (w - mu) / sigma
        # two-sided p
        p = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return float(w), float(p)


def _paired_t_test(diffs: np.ndarray) -> Tuple[float, float]:
    """Paired-samples t-test on differences. Returns (t, p_two_sided)."""
    n = diffs.size
    if n < 2:
        return 0.0, 1.0
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    if sd == 0:
        return 0.0, 1.0 if mean == 0 else 0.0
    t = mean / (sd / math.sqrt(n))
    try:
        from scipy.stats import t as t_dist
        p = 2.0 * (1.0 - float(t_dist.cdf(abs(t), df=n - 1)))
    except ImportError:
        # Normal approximation (valid for n ≥ 30)
        p = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return float(t), float(p)


def _cohens_d(diffs: np.ndarray) -> float:
    n = diffs.size
    if n < 2:
        return 0.0
    sd = float(diffs.std(ddof=1))
    if sd == 0:
        return 0.0
    return float(diffs.mean() / sd)


def _bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap CI for ``mean(a) − mean(b)`` with paired resampling."""
    if a.size != b.size or a.size == 0:
        return {"point": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_boot": int(n_boot)}
    rng = np.random.default_rng(20260508)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    sa = a[idx].mean(axis=1); sb = b[idx].mean(axis=1)
    diffs = sa - sb
    return {
        "point": float(a.mean() - b.mean()),
        "ci_lo": float(np.quantile(diffs, alpha / 2)),
        "ci_hi": float(np.quantile(diffs, 1 - alpha / 2)),
        "n_boot": int(n_boot),
    }


# ---------------------------------------------------------------------------
# Per-cell comparison
# ---------------------------------------------------------------------------

@dataclass
class CellComparison:
    sequence: str
    qp_base: int
    method: str
    reference: str
    n_frames: int
    f1_mean_method: float
    f1_mean_reference: float
    diff_mean: float
    cohens_d: float
    wilcoxon_W: float
    wilcoxon_p: float
    paired_t: float
    paired_t_p: float
    bootstrap_ci: Dict[str, float] = field(default_factory=dict)
    is_real_effect: bool = False


def _classify(cmp: CellComparison) -> bool:
    return (cmp.wilcoxon_p < ALPHA) and (abs(cmp.cohens_d) > SMALL_EFFECT)


def compare_cells(d1_cells: List[Dict], reference_method: str,
                  test_methods: List[str]) -> List[CellComparison]:
    """For every (sequence, QP) build per-method comparisons against reference."""
    by_key: Dict[Tuple[str, int, str], Dict] = {
        (c["sequence"], int(c["qp_base"]), c["method"]): c for c in d1_cells
    }
    results: List[CellComparison] = []
    keys = sorted({(c["sequence"], int(c["qp_base"])) for c in d1_cells})
    for seq, qp in keys:
        ref = by_key.get((seq, qp, reference_method))
        if not ref:
            logger.warning("No reference cell %s/%s/QP%d", seq, reference_method, qp)
            continue
        ref_f1 = _per_frame_f1(ref["per_frame_counts"])
        for method in test_methods:
            if method == reference_method:
                continue
            tgt = by_key.get((seq, qp, method))
            if not tgt:
                logger.warning("Missing %s/%s/QP%d", seq, method, qp)
                continue
            tgt_f1 = _per_frame_f1(tgt["per_frame_counts"])
            n = min(ref_f1.size, tgt_f1.size)
            if n == 0:
                continue
            ref_f1 = ref_f1[:n]; tgt_f1 = tgt_f1[:n]
            diff = tgt_f1 - ref_f1
            wW, wp = _wilcoxon_signed_rank(diff)
            tt, tp_ = _paired_t_test(diff)
            d = _cohens_d(diff)
            ci = _bootstrap_ci(tgt_f1, ref_f1)
            cmp = CellComparison(
                sequence=seq, qp_base=qp, method=method, reference=reference_method,
                n_frames=int(n),
                f1_mean_method=float(tgt_f1.mean()),
                f1_mean_reference=float(ref_f1.mean()),
                diff_mean=float(diff.mean()),
                cohens_d=float(d),
                wilcoxon_W=float(wW), wilcoxon_p=float(wp),
                paired_t=float(tt), paired_t_p=float(tp_),
                bootstrap_ci=ci,
            )
            cmp.is_real_effect = _classify(cmp)
            results.append(cmp)
    return results


def render_markdown(results: List[CellComparison], reference: str) -> str:
    lines: List[str] = []
    lines.append(f"# D2 — paired statistical significance vs {reference}\n")
    lines.append(
        f"Per-frame F1 at IoU=0.5, conf=0.25. "
        f"\"Real\" = Wilcoxon p < {ALPHA} AND |Cohen's d| > {SMALL_EFFECT}.\n"
    )
    lines.append("| Sequence | QP | Method | n | F1_M | F1_R | Δ̄ | d | p_Wil | CI95 lo–hi | Verdict |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|")
    for c in results:
        v = "REAL" if c.is_real_effect else "noise"
        lines.append(
            f"| {c.sequence} | {c.qp_base} | {c.method} | {c.n_frames} | "
            f"{c.f1_mean_method:.4f} | {c.f1_mean_reference:.4f} | "
            f"{c.diff_mean:+.4f} | {c.cohens_d:+.2f} | {c.wilcoxon_p:.3f} | "
            f"{c.bootstrap_ci.get('ci_lo', 0):+.4f} – {c.bootstrap_ci.get('ci_hi', 0):+.4f} | "
            f"{v} |"
        )
    n_real = sum(1 for c in results if c.is_real_effect)
    n_real_pos = sum(1 for c in results if c.is_real_effect and c.diff_mean > 0)
    n_real_neg = sum(1 for c in results if c.is_real_effect and c.diff_mean < 0)
    lines.append("")
    lines.append(
        f"**Tally**: {n_real}/{len(results)} cells are statistically real "
        f"({n_real_pos} wins, {n_real_neg} losses, "
        f"{len(results) - n_real} within noise)."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_d1_jsons(paths: List[Path]) -> List[Dict]:
    cells: List[Dict] = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        for c in data["cells"]:
            c["_source"] = str(p)
            cells.append(c)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description="D2 — paired stat tests on D1 outputs")
    parser.add_argument("--d1-json", required=True, type=Path, nargs="+",
                        help="One or more d1_true_map.json files")
    parser.add_argument("--reference-method", default="M0")
    parser.add_argument("--test-methods", nargs="+", default=["M4"])
    parser.add_argument("--output", required=True, type=Path,
                        help="Output JSON path (a sibling .md is also written)")
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

    cells = _load_d1_jsons(args.d1_json)
    logger.info("Loaded %d cells from %d D1 file(s)", len(cells), len(args.d1_json))

    results = compare_cells(cells, args.reference_method, args.test_methods)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "reference_method": args.reference_method,
        "test_methods": args.test_methods,
        "alpha": ALPHA,
        "small_effect": SMALL_EFFECT,
        "results": [asdict(r) for r in results],
    }, indent=2), encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(render_markdown(results, args.reference_method),
                       encoding="utf-8")
    logger.info("Wrote %d comparisons to %s", len(results), args.output)
    logger.info("Markdown summary at %s", md_path)


if __name__ == "__main__":
    main()
