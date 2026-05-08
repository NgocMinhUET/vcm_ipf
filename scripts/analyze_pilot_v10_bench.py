"""Analyse pilot_v10_bench: 6-method head-to-head against the audited metric.

Reads the D1 + D2 outputs produced by ``run_phase1_diagnostics.py`` for
the pilot_v10_bench directory and produces:

1. A per-(sequence, QP) table of ``(method, true mAP_50, p_vs_M0,
   Cohen's d, verdict)``.
2. A pairwise win matrix on per-frame F1 (which method beats which,
   counting REAL effects only).
3. A BD-Rate-Task table per sequence using PCHIP interpolation against
   each method's RD points.
4. The "best method" per sequence and a global "is M4 best?" verdict.

Usage::

    PYTHONPATH=src python phase2/scripts/analyze_pilot_v10_bench.py \\
        --d1-json ~/Minh/ipf/phase2_outputs/pilot_v10_bench/diagnostics/d1_true_map.json \\
        --d2-json ~/Minh/ipf/phase2_outputs/pilot_v10_bench/diagnostics/d2_stats.json \\
        --experiment-summary ~/Minh/ipf/phase2_outputs/pilot_v10_bench/experiment_summary.json \\
        --output ~/Minh/ipf/phase2_outputs/pilot_v10_bench/diagnostics/v10_verdict.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger("phase2.scripts.analyze_pilot_v10_bench")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trapz(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x)


def _monotonise_pchip(rate: np.ndarray, qual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort by rate ascending and remove duplicates so PCHIP can interpolate."""
    order = np.argsort(rate)
    r = rate[order]; q = qual[order]
    # collapse duplicate rates by averaging quality
    dedup_r = []
    dedup_q = []
    i = 0
    while i < len(r):
        j = i
        while j + 1 < len(r) and abs(r[j + 1] - r[i]) < 1e-9:
            j += 1
        dedup_r.append(float(r[i]))
        dedup_q.append(float(np.mean(q[i:j + 1])))
        i = j + 1
    return np.asarray(dedup_r), np.asarray(dedup_q)


def _bd_rate_task(rate_a: np.ndarray, mapa: np.ndarray,
                  rate_b: np.ndarray, mapb: np.ndarray) -> float:
    """BD-Rate-Task: % rate change of A relative to B at iso-task using PCHIP.

    Negative = A uses fewer bits at the same task accuracy than B (A wins).
    """
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        return float("nan")

    # interpolate quality(rate) for each
    ra, qa = _monotonise_pchip(np.asarray(rate_a, dtype=float),
                                np.asarray(mapa, dtype=float))
    rb, qb = _monotonise_pchip(np.asarray(rate_b, dtype=float),
                                np.asarray(mapb, dtype=float))
    if len(ra) < 2 or len(rb) < 2:
        return float("nan")

    log_ra = np.log10(np.maximum(ra, 1e-9))
    log_rb = np.log10(np.maximum(rb, 1e-9))

    pa = PchipInterpolator(log_ra, qa, extrapolate=False)
    pb = PchipInterpolator(log_rb, qb, extrapolate=False)

    q_lo = max(qa.min(), qb.min())
    q_hi = min(qa.max(), qb.max())
    if q_hi - q_lo < 1e-6:
        return float("nan")

    # Find log10(rate) at common qualities, integrate
    # Invert: solve qa(log_r) = q for log_r
    grid_q = np.linspace(q_lo, q_hi, 50)
    log_ra_q = np.empty_like(grid_q); log_rb_q = np.empty_like(grid_q)
    # Approximate inverse via interpolation on the original data
    inv_a = PchipInterpolator(qa[np.argsort(qa)], log_ra[np.argsort(qa)],
                               extrapolate=False)
    inv_b = PchipInterpolator(qb[np.argsort(qb)], log_rb[np.argsort(qb)],
                               extrapolate=False)
    for i, q in enumerate(grid_q):
        log_ra_q[i] = float(inv_a(q))
        log_rb_q[i] = float(inv_b(q))
    if np.any(np.isnan(log_ra_q)) or np.any(np.isnan(log_rb_q)):
        return float("nan")

    avg_diff = float(_trapz(log_ra_q - log_rb_q, grid_q) / (q_hi - q_lo))
    return float((10 ** avg_diff - 1) * 100.0)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def _index_d1(cells: List[dict]) -> Dict[Tuple[str, str, int], dict]:
    return {(c["sequence"], c["method"], int(c["qp_base"])): c for c in cells}


def _index_d2(comparisons: List[dict]) -> Dict[Tuple[str, str, int], dict]:
    return {
        (c["sequence"], c["method"], int(c["qp_base"])): c for c in comparisons
    }


def _index_summary(summary: dict) -> Dict[Tuple[str, str, int], dict]:
    out: Dict[Tuple[str, str, int], dict] = {}
    for entry in summary.get("results", []):
        key = (entry["sequence"], entry["method"], int(entry["qp_base"]))
        out[key] = entry
    return out


def per_method_bd_rate(
    summary_idx: Dict, d1_idx: Dict, methods: List[str], reference: str,
    sequences: List[str],
) -> Dict[str, Dict[str, float]]:
    """For each (method, seq), compute BD-Rate-Task vs reference using true mAP."""
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for seq in sequences:
        ref_rates = []
        ref_maps = []
        # Build reference RD curve
        for entry in sorted(summary_idx.values(),
                            key=lambda e: (e["sequence"], e["method"], int(e["qp_base"]))):
            if entry["sequence"] != seq or entry["method"] != reference:
                continue
            qp = int(entry["qp_base"])
            rate = float(entry.get("bitrate_kbps", entry.get("encode", {}).get("bitrate_kbps", 0)))
            cell = d1_idx.get((seq, reference, qp))
            if cell is None or rate <= 0:
                continue
            ref_rates.append(rate)
            ref_maps.append(float(cell["mAP_50_95"]))

        for method in methods:
            if method == reference:
                continue
            m_rates = []; m_maps = []
            for entry in sorted(summary_idx.values()):
                if entry["sequence"] != seq or entry["method"] != method:
                    continue
                qp = int(entry["qp_base"])
                rate = float(entry.get("bitrate_kbps",
                                        entry.get("encode", {}).get("bitrate_kbps", 0)))
                cell = d1_idx.get((seq, method, qp))
                if cell is None or rate <= 0:
                    continue
                m_rates.append(rate); m_maps.append(float(cell["mAP_50_95"]))
            if len(m_rates) >= 2 and len(ref_rates) >= 2:
                bd = _bd_rate_task(np.asarray(m_rates), np.asarray(m_maps),
                                   np.asarray(ref_rates), np.asarray(ref_maps))
                out[method][seq] = bd
            else:
                out[method][seq] = float("nan")
    return out


def render_verdict(
    methods: List[str],
    sequences: List[str],
    bd_table: Dict[str, Dict[str, float]],
    d2_idx: Dict,
) -> str:
    """Markdown report with BD-Rate per (method, seq) and a winner ranking."""
    lines: List[str] = []
    lines.append("# pilot_v10_bench — comprehensive verdict\n")
    lines.append("All numbers use TRUE COCO mAP_50_95 (D1) and paired Wilcoxon (D2).")
    lines.append("")
    lines.append("## BD-Rate-Task (% vs M0, lower is better)\n")
    header = "| Method | " + " | ".join(sequences) + " | Mean |"
    sep = "|---|" + "|".join(["---:"] * (len(sequences) + 1)) + "|"
    lines.append(header); lines.append(sep)
    for m in methods:
        if m == "M0":
            continue
        row = [m]
        vals = []
        for s in sequences:
            v = bd_table.get(m, {}).get(s, float("nan"))
            row.append(f"{v:+.2f}%" if not np.isnan(v) else "n/a")
            if not np.isnan(v):
                vals.append(v)
        mean = float(np.mean(vals)) if vals else float("nan")
        row.append(f"{mean:+.2f}%" if vals else "n/a")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Per-cell paired statistical wins (D2)\n")
    lines.append("| Sequence | QP | Method | Δ̄F1 | d | p_Wil | Verdict |")
    lines.append("|---|---:|---|---:|---:|---:|---|")
    for key in sorted(d2_idx.keys()):
        seq, m, qp = key
        if m == "M0":
            continue
        c = d2_idx[key]
        v = "REAL " + ("WIN" if c["diff_mean"] > 0 else "LOSS") if c.get("is_real_effect") else "noise"
        lines.append(
            f"| {seq} | {qp} | {m} | {c['diff_mean']:+.4f} | "
            f"{c['cohens_d']:+.2f} | {c['wilcoxon_p']:.3f} | {v} |"
        )

    # Ranking
    means: List[Tuple[str, float]] = []
    for m, sd in bd_table.items():
        vs = [v for v in sd.values() if not np.isnan(v)]
        if vs:
            means.append((m, float(np.mean(vs))))
    means.sort(key=lambda t: t[1])
    lines.append("")
    lines.append("## Ranking (mean BD-Rate-Task across sequences)\n")
    for i, (m, v) in enumerate(means, 1):
        lines.append(f"{i}. **{m}**: {v:+.2f}%")
    if means:
        best = means[0][0]
        lines.append("")
        if best == "M4":
            lines.append("**Verdict**: M4 (LiteQP-CNN) is the best on average.")
        else:
            lines.append(
                f"**Verdict**: {best} is the best on average; M4 is "
                f"{'rank ' + str(next((i+1 for i,(m,_) in enumerate(means) if m=='M4'), '?'))}."
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse pilot_v10_bench")
    parser.add_argument("--d1-json", required=True, type=Path)
    parser.add_argument("--d2-json", required=True, type=Path)
    parser.add_argument("--experiment-summary", required=True, type=Path)
    parser.add_argument("--reference", default="M0")
    parser.add_argument("--output", required=True, type=Path)
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

    d1 = json.loads(args.d1_json.read_text(encoding="utf-8"))
    d2 = json.loads(args.d2_json.read_text(encoding="utf-8"))
    summary = json.loads(args.experiment_summary.read_text(encoding="utf-8"))

    d1_idx = _index_d1(d1["cells"])
    d2_idx = _index_d2(d2["results"])
    summary_idx = _index_summary(summary)

    methods = sorted({c["method"] for c in d1["cells"]})
    sequences = sorted({c["sequence"] for c in d1["cells"]})

    bd_table = per_method_bd_rate(
        summary_idx, d1_idx, methods, args.reference, sequences
    )
    md = render_verdict(methods, sequences, bd_table, d2_idx)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    logger.info("Wrote verdict to %s", args.output)
    print("\n" + md)


if __name__ == "__main__":
    main()
