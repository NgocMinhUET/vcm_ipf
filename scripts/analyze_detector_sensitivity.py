"""Diagnose whether the detector or the QP range is the headroom bottleneck.

Reads ``d1_true_map.json`` files produced by ``scripts/reevaluate_pilots.py``
and computes per-(sequence, method) mAP-vs-QP slopes plus the maximum
ROI-allocation headroom each setup permits. Output is a Markdown table
that tells you whether ROI-aware QP allocation can possibly work in the
current regime.

Why this matters
----------------
If ``mAP_50_95(M0)`` barely changes across QP=27..42, then no ROI-aware
allocation method (including M4 LiteQP) can produce gain larger than
the metric's noise floor. The diagnostic compares two regimes:

* **Slope ≤ 0.002 mAP/QP**  → saturated regime, no headroom
* **Slope ≥ 0.005 mAP/QP**  → useful regime, ROI methods can compete

Headroom is computed as::

    max_gain ≈ slope × |Δ_QP_max| × roi_area_fraction

where ``Δ_QP_max`` is the typical δ amplitude from M4 (taken from the
config, default 4) and ``roi_area_fraction`` is the share of CTUs that
fall in the high-saliency region (default 0.3 — coarse estimate).

Usage on the server::

    PYTHONPATH=src python scripts/analyze_detector_sensitivity.py \\
        --d1-jsons \\
            ~/Minh/ipf/phase2_outputs/pilot_v8b/diagnostics/d1_true_map.json \\
            ~/Minh/ipf/phase2_outputs/pilot_v8b/diagnostics/d1_true_map_yolov8m.json \\
        --detector-labels yolov8n yolov8m \\
        --output ~/Minh/ipf/phase2_outputs/diagnostics/detector_sensitivity.md

Outputs a Markdown report comparing per-detector slopes and concludes
whether Path A (stronger detector) opens the door to publishable gains.
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

logger = logging.getLogger("phase2.scripts.analyze_detector_sensitivity")


def _load_d1_cells(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("cells", [])


def _slope_mAP_per_QP(qps: List[int], maps: List[float]) -> Tuple[float, float]:
    """Return (slope, R²) of a linear fit mAP = a + b·QP.

    Slope is in units of *absolute mAP change per +1 QP*. We use
    least-squares so non-monotonic curves don't crash, and report R² so
    the caller can flag noisy fits (R² < 0.5 means the curve is too
    irregular for the slope to be meaningful).
    """
    if len(qps) < 2:
        return float("nan"), float("nan")
    x = np.asarray(qps, dtype=float)
    y = np.asarray(maps, dtype=float)
    a, b = np.polyfit(x, y, 1)  # y = a*x + b
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return float(a), float(r2)


def _summarise(cells: List[dict]) -> Dict[Tuple[str, str], dict]:
    """Group cells by (sequence, method) and compute slope per group."""
    by_key: Dict[Tuple[str, str], List[Tuple[int, float]]] = defaultdict(list)
    for c in cells:
        by_key[(c["sequence"], c["method"])].append(
            (int(c["qp_base"]), float(c["mAP_50_95"]))
        )
    out: Dict[Tuple[str, str], dict] = {}
    for key, pts in by_key.items():
        pts.sort(key=lambda x: x[0])
        qps = [p[0] for p in pts]
        maps = [p[1] for p in pts]
        slope, r2 = _slope_mAP_per_QP(qps, maps)
        out[key] = {
            "qps": qps,
            "mAP_50_95": maps,
            "slope_per_QP": slope,
            "abs_slope_per_QP": abs(slope),
            "r2_linear": r2,
            "mAP_at_lowest_QP": maps[0] if maps else float("nan"),
            "mAP_at_highest_QP": maps[-1] if maps else float("nan"),
            "mAP_drop": (maps[0] - maps[-1]) if maps else float("nan"),
        }
    return out


def _headroom(slope: float, delta_qp: int, roi_fraction: float) -> float:
    """Theoretical maximum ROI-aware gain in absolute mAP."""
    if not np.isfinite(slope):
        return float("nan")
    return abs(slope) * delta_qp * roi_fraction


def _verdict(
    abs_slope: float, headroom: float, noise_floor: float,
) -> str:
    if not np.isfinite(abs_slope) or not np.isfinite(headroom):
        return "n/a"
    if headroom >= 2.0 * noise_floor:
        return "USEFUL — ROI-aware QP allocation has clear headroom"
    if headroom >= noise_floor:
        return "MARGINAL — gain comparable to noise; need more frames"
    return "SATURATED — no headroom; ROI methods cannot beat M0 here"


def render_report(
    summaries_by_detector: Dict[str, Dict[Tuple[str, str], dict]],
    delta_qp: int, roi_fraction: float, noise_floor: float,
) -> str:
    lines: List[str] = []
    lines.append("# Detector-sensitivity diagnosis\n")
    lines.append(
        "Each detector's `mAP_50_95(QP)` curve is fit linearly. Slope "
        "tells us how much real AP changes per +1 QP. ROI-aware QP "
        "allocation is *only* useful when the slope is large enough that "
        "redistributing ±Δ QP across the high-saliency fraction can move "
        "AP by more than the metric's noise floor.\n"
    )
    lines.append(f"Assumptions used in the headroom column:\n")
    lines.append(f"* `Δ_QP_max = ±{delta_qp}` (max signed δ from M4 maps)")
    lines.append(f"* `ROI fraction ≈ {roi_fraction:.2f}` (share of high-φ CTUs)")
    lines.append(f"* `noise floor ≈ {noise_floor:.3f}` (≈1·SE of mAP for n=50 frames)\n")

    sequences = sorted({k[0] for s in summaries_by_detector.values() for k in s})
    for det, summary in summaries_by_detector.items():
        lines.append(f"## Detector: {det}\n")
        lines.append("| Sequence | Method | mAP@QP_lo | mAP@QP_hi | drop | "
                     "abs slope/QP | R² | headroom | verdict |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for seq in sequences:
            for method in ("M0", "M4"):
                key = (seq, method)
                if key not in summary:
                    continue
                row = summary[key]
                slope = row["abs_slope_per_QP"]
                head = _headroom(slope, delta_qp, roi_fraction)
                verdict = _verdict(slope, head, noise_floor)
                lines.append(
                    f"| {seq} | {method} | {row['mAP_at_lowest_QP']:.4f} | "
                    f"{row['mAP_at_highest_QP']:.4f} | {row['mAP_drop']:+.4f} | "
                    f"{slope:.4f} | {row['r2_linear']:+.2f} | "
                    f"{head:.4f} | {verdict} |"
                )
        # Aggregate
        m0_slopes = [v["abs_slope_per_QP"]
                     for k, v in summary.items() if k[1] == "M0"
                     and np.isfinite(v["abs_slope_per_QP"])]
        if m0_slopes:
            mean_slope = float(np.mean(m0_slopes))
            mean_head = _headroom(mean_slope, delta_qp, roi_fraction)
            global_verdict = _verdict(mean_slope, mean_head, noise_floor)
            lines.append("")
            lines.append(f"**M0 mean abs slope = {mean_slope:.4f} mAP/QP, "
                          f"mean headroom = {mean_head:.4f}, "
                          f"verdict: {global_verdict}**")
            lines.append("")

    # Global comparison if multiple detectors present
    if len(summaries_by_detector) > 1:
        lines.append("## Cross-detector slope comparison (M0 only)\n")
        lines.append("| Sequence | " + " | ".join(summaries_by_detector) + " |")
        lines.append("|---|" + "|".join([":---:"] * len(summaries_by_detector)) + "|")
        for seq in sequences:
            row = [seq]
            for det, summary in summaries_by_detector.items():
                key = (seq, "M0")
                if key in summary:
                    row.append(f"{summary[key]['abs_slope_per_QP']:.4f}")
                else:
                    row.append("n/a")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("### Path-A decision\n")
        # Pick the strongest-detector mean slope and compare
        best = max(summaries_by_detector.items(),
                   key=lambda kv: np.mean([
                       v["abs_slope_per_QP"]
                       for k, v in kv[1].items()
                       if k[1] == "M0" and np.isfinite(v["abs_slope_per_QP"])
                   ]) if kv[1] else 0.0)
        bn = best[0]
        m0 = [v["abs_slope_per_QP"]
              for k, v in best[1].items() if k[1] == "M0"
              and np.isfinite(v["abs_slope_per_QP"])]
        slope_best = float(np.mean(m0)) if m0 else float("nan")
        head_best = _headroom(slope_best, delta_qp, roi_fraction)
        lines.append(f"Strongest detector: **{bn}**, mean M0 slope = "
                      f"{slope_best:.4f}, headroom = {head_best:.4f}.")
        if head_best >= 2 * noise_floor:
            lines.append("→ **Proceed**: re-run M4 vs M0 evaluation under "
                         f"`{bn}` to obtain publishable BD-Rate-Task numbers.")
        elif head_best >= noise_floor:
            lines.append("→ **Marginal**: stronger detector helps but the "
                         "noise floor still dominates. Increase n_frames "
                         "or move to higher-QP regime (Path B).")
        else:
            lines.append("→ **Path A insufficient**: detector swap does not "
                         "open headroom. Move to Path B (QP grid 37–52) or "
                         "Path D (more sequences) before any further model "
                         "work.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detector-sensitivity diagnosis (Path A decision).",
    )
    ap.add_argument("--d1-jsons", nargs="+", required=True, type=Path,
                    help="One or more d1_true_map*.json files.")
    ap.add_argument("--detector-labels", nargs="+", required=True,
                    help="Label per d1 JSON, in the same order. Example: "
                         "yolov8n yolov8m.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--delta-qp", type=int, default=4,
                    help="Max signed δ used by M4 (default 4).")
    ap.add_argument("--roi-fraction", type=float, default=0.30,
                    help="Fraction of CTUs in the high-saliency band "
                         "(coarse estimate; default 0.30).")
    ap.add_argument("--noise-floor", type=float, default=0.020,
                    help="Approximate 1-SE of mAP_50_95 with n=50 frames "
                         "(default 0.020).")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if len(args.d1_jsons) != len(args.detector_labels):
        raise SystemExit("--d1-jsons and --detector-labels must have the same length")

    summaries: Dict[str, Dict[Tuple[str, str], dict]] = {}
    for path, label in zip(args.d1_jsons, args.detector_labels):
        cells = _load_d1_cells(path)
        summaries[label] = _summarise(cells)
        logger.info("%s — %d cells loaded from %s", label, len(cells), path)

    md = render_report(summaries, args.delta_qp, args.roi_fraction, args.noise_floor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    logger.info("Wrote sensitivity report to %s", args.output)


if __name__ == "__main__":
    main()
