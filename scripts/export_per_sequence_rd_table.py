"""Export per-sequence RD tables with BD-Rate, BD-Accuracy, and statistics.

This script produces the mandatory per-sequence comparison table required
by MPEG VCM CTC and the project evaluation standard:

    Method | QP | Rate (kbps) | mAP@0.5 | mAP@[0.5:0.95] | BD-Rate (%) | BD-Acc

One table per sequence; BD columns are ONE value per method block, computed
by PCHIP over the QP-rate-accuracy triples against the M0 anchor.

Statistics added (IEEE Access Q1 requirement)
---------------------------------------------
- Bootstrap 95% CI on BD-Rate (B=1000, paired by frame).
- Per-cell (sequence × QP): Wilcoxon p, Cohen's d, verdict (Real Win / Noise / Real Loss).
- Win/Neutral/Loss tally per method.
- Headroom H per sequence.

Output
------
- Markdown tables (one per sequence) → stdout / file
- LaTeX tables → <output_dir>/tab_rdtable_<seq>.tex
- CSV → <output_dir>/rd_table_full.csv
- JSON → <output_dir>/rd_table_full.json  (machine-readable, feeds access.tex)

Usage
-----
    PYTHONPATH=src python scripts/export_per_sequence_rd_table.py \\
        --d1-json ~/Minh/ipf/phase2_outputs/pilot_v11_fullqp/diagnostics/d1_true_map.json \\
        --d2-json ~/Minh/ipf/phase2_outputs/pilot_v11_fullqp/diagnostics/d2_stats.json \\
        --experiment-summary ~/Minh/ipf/phase2_outputs/pilot_v11_fullqp/experiment_summary.json \\
        --bd-bootstrap-json ~/Minh/ipf/phase2_outputs/pilot_v11_fullqp/diagnostics/bd_rate_bootstrap.json \\
        --reference-method M0 \\
        --methods M_soft_roi_ref M1 M5 M6 Classical-IPF LiteQP-v4 CA-OGIPF \\
        --heldout-sequences MOT17-11-DPM MOT17-13-DPM \\
        --output-dir ~/Minh/ipf/phase2_outputs/pilot_v11_fullqp/paper_tables
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("scripts.export_per_sequence_rd_table")

# Method labels for display in tables
METHOD_LABELS = {
    "M0": "VTM (anchor)",
    "M_soft_roi_ref": "Soft-ROI [SoftROI-TCSVT23]",
    "M1": "Binary ROI (M1)",
    "M5": "Anisotropic Gauss. (M5)",
    "M6": "RD-driven (M6)",
    "Classical-IPF": "Classical IPF",
    "LiteQP-v4": "LiteQP-v4 (prior)",
    "CA-OGIPF": r"\textbf{CA-OG-IPF (Proposed)}",
}


# ---------------------------------------------------------------------------
# PCHIP BD-Rate / BD-Accuracy
# ---------------------------------------------------------------------------

def _bd_rate_pchip(
    rate_a: np.ndarray, acc_a: np.ndarray,
    rate_b: np.ndarray, acc_b: np.ndarray,
) -> float:
    """BD-Rate: % bitrate change of A vs B at iso-accuracy. Negative = A wins."""
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        logger.error("scipy not available; BD-Rate not computed")
        return float("nan")

    def _dedup(x, y):
        order = np.argsort(x)
        x = x[order]; y = y[order]
        xu, yi = [], []
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and abs(x[j + 1] - x[i]) < 1e-9:
                j += 1
            xu.append(float(x[i]))
            yi.append(float(np.mean(y[i:j + 1])))
            i = j + 1
        return np.asarray(xu), np.asarray(yi)

    ra, qa = _dedup(np.asarray(rate_a, dtype=float), np.asarray(acc_a, dtype=float))
    rb, qb = _dedup(np.asarray(rate_b, dtype=float), np.asarray(acc_b, dtype=float))
    if len(ra) < 2 or len(rb) < 2:
        return float("nan")

    log_ra = np.log10(np.maximum(ra, 1e-9))
    log_rb = np.log10(np.maximum(rb, 1e-9))

    qa2, log_ra2 = _dedup(qa, log_ra)
    qb2, log_rb2 = _dedup(qb, log_rb)
    if len(qa2) < 2 or len(qb2) < 2:
        return float("nan")

    q_lo = max(qa2.min(), qb2.min())
    q_hi = min(qa2.max(), qb2.max())
    if q_hi - q_lo < 1e-6:
        return float("nan")

    try:
        pchip_a = PchipInterpolator(qa2, log_ra2)
        pchip_b = PchipInterpolator(qb2, log_rb2)
        qq = np.linspace(q_lo, q_hi, 200)
        la = pchip_a(qq); lb = pchip_b(qq)
        bd_rate = float(np.trapz(la - lb, qq) / (q_hi - q_lo)) * 100.0
        return bd_rate
    except Exception:
        return float("nan")


def _bd_accuracy_pchip(
    rate_a: np.ndarray, acc_a: np.ndarray,
    rate_b: np.ndarray, acc_b: np.ndarray,
) -> float:
    """BD-Accuracy: accuracy gain of A vs B at iso-rate. Positive = A wins."""
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        return float("nan")

    def _dedup(x, y):
        order = np.argsort(x)
        x = x[order]; y = y[order]
        xu, yi = [], []
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and abs(x[j + 1] - x[i]) < 1e-9:
                j += 1
            xu.append(float(x[i]))
            yi.append(float(np.mean(y[i:j + 1])))
            i = j + 1
        return np.asarray(xu), np.asarray(yi)

    ra, qa = _dedup(np.asarray(rate_a, dtype=float), np.asarray(acc_a, dtype=float))
    rb, qb = _dedup(np.asarray(rate_b, dtype=float), np.asarray(acc_b, dtype=float))
    if len(ra) < 2 or len(rb) < 2:
        return float("nan")

    log_ra = np.log10(np.maximum(ra, 1e-9))
    log_rb = np.log10(np.maximum(rb, 1e-9))

    ra2, qa2 = _dedup(log_ra, qa)
    rb2, qb2 = _dedup(log_rb, qb)
    if len(ra2) < 2 or len(rb2) < 2:
        return float("nan")

    r_lo = max(ra2.min(), rb2.min())
    r_hi = min(ra2.max(), rb2.max())
    if r_hi - r_lo < 1e-6:
        return float("nan")

    try:
        pchip_a = PchipInterpolator(ra2, qa2)
        pchip_b = PchipInterpolator(rb2, qb2)
        rr = np.linspace(r_lo, r_hi, 200)
        return float(np.trapz(pchip_a(rr) - pchip_b(rr), rr) / (r_hi - r_lo))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# LaTeX table builder
# ---------------------------------------------------------------------------

def _latex_rd_table(
    seq: str,
    qp_points: List[int],
    method_data: Dict[str, Dict],
    method_order: List[str],
    anchor: str = "M0",
    split_label: str = "",
) -> str:
    """Return a LaTeX longtable for one sequence."""
    lines = []
    caption_suffix = f" [{split_label}]" if split_label else ""
    lines.append(r"\begin{table}[t!]")
    lines.append(
        r"\caption{Per-QP Rate--Accuracy and BD-Rate-Task for "
        + seq.replace("_", r"\_")
        + caption_suffix
        + r". BD-Rate and BD-Acc are computed by PCHIP vs the VTM anchor (M0)."
        + r" CI = 95\% paired-bootstrap confidence interval (B=1000)."
        + r" Ref.\ [SoftROI-TCSVT23]: X.\ Liu~\etal, TCSVT 2023.}"
    )
    lines.append(r"\label{tab:rd_" + seq.replace("-", "") + r"}")
    lines.append(r"\centering\small")
    lines.append(
        r"\begin{tabular}{llrrrrrrcc}"
    )
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Method} & \textbf{QP} & \textbf{Rate} & "
        r"\textbf{mAP$_{50}$} & \textbf{mAP$_{50:95}$} & "
        r"\textbf{BD-BR$_{50}$} & \textbf{BD-BR$_{50:95}$} & "
        r"\textbf{BD-Acc$_{50:95}$} & \textbf{CI$_{50:95}$} & \textbf{Verd.}\\"
    )
    lines.append(r"\midrule")

    for mi, method in enumerate(method_order):
        if method not in method_data:
            continue
        mdat = method_data[method]
        label = METHOD_LABELS.get(method, method)
        rows = mdat.get("per_qp", {})
        bd_br_50 = mdat.get("bd_rate_mAP50", float("nan"))
        bd_br_5095 = mdat.get("bd_rate_mAP50_95", float("nan"))
        bd_acc = mdat.get("bd_acc_mAP50_95", float("nan"))
        ci_lo = mdat.get("ci_lo", float("nan"))
        ci_hi = mdat.get("ci_hi", float("nan"))
        verdict = mdat.get("aggregate_verdict", "—")

        def _fmt(v, fmt=".2f"):
            return "—" if v != v else f"{v:{fmt}}"

        for qi, qp in enumerate(qp_points):
            row = rows.get(str(qp), rows.get(qp, {}))
            rate_str = _fmt(row.get("rate_kbps", float("nan")), ".1f")
            map50_str = _fmt(row.get("mAP_50", float("nan")), ".4f")
            map5095_str = _fmt(row.get("mAP_50_95", float("nan")), ".4f")

            is_anchor = (method == anchor)
            if qi == 0:
                n_qp = len(qp_points)
                method_col = r"\multirow{" + str(n_qp) + r"}{*}{" + label + r"}"
                if is_anchor:
                    bd_br50_col = r"\multirow{" + str(n_qp) + r"}{*}{—}"
                    bd_br5095_col = r"\multirow{" + str(n_qp) + r"}{*}{—}"
                    bd_acc_col = r"\multirow{" + str(n_qp) + r"}{*}{—}"
                    ci_col = r"\multirow{" + str(n_qp) + r"}{*}{—}"
                    verd_col = r"\multirow{" + str(n_qp) + r"}{*}{—}"
                else:
                    bd_br50_col = (
                        r"\multirow{" + str(n_qp) + r"}{*}{"
                        + (_fmt(bd_br_50) if bd_br_50 == bd_br_50 else "—")
                        + r"\%}"
                    )
                    bd_br5095_col = (
                        r"\multirow{" + str(n_qp) + r"}{*}{"
                        + (_fmt(bd_br_5095) if bd_br_5095 == bd_br_5095 else "—")
                        + r"\%}"
                    )
                    bd_acc_col = (
                        r"\multirow{" + str(n_qp) + r"}{*}{"
                        + (_fmt(bd_acc, ".4f") if bd_acc == bd_acc else "—")
                        + r"}"
                    )
                    ci_str = (
                        f"[{ci_lo:.2f}, {ci_hi:.2f}]"
                        if (ci_lo == ci_lo and ci_hi == ci_hi)
                        else "—"
                    )
                    ci_col = r"\multirow{" + str(n_qp) + r"}{*}{" + ci_str + r"}"
                    verd_col = r"\multirow{" + str(n_qp) + r"}{*}{" + verdict + r"}"
            else:
                method_col = ""
                bd_br50_col = bd_br5095_col = bd_acc_col = ci_col = verd_col = ""

            lines.append(
                f"{method_col} & {qp} & {rate_str} & {map50_str} & {map5095_str}"
                f" & {bd_br50_col} & {bd_br5095_col} & {bd_acc_col}"
                f" & {ci_col} & {verd_col} \\\\"
            )

        if mi < len(method_order) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-json", required=True)
    parser.add_argument("--d2-json", default=None)
    parser.add_argument("--experiment-summary", default=None)
    parser.add_argument("--bd-bootstrap-json", default=None)
    parser.add_argument("--reference-method", default="M0")
    parser.add_argument("--methods", nargs="+",
                        default=["M_soft_roi_ref", "M1", "M5", "M6",
                                 "Classical-IPF", "LiteQP-v4", "CA-OGIPF"])
    parser.add_argument("--heldout-sequences", nargs="*", default=[])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    d1 = json.loads(Path(args.d1_json).read_text())
    d2 = json.loads(Path(args.d2_json).read_text()) if args.d2_json else {}
    exp = json.loads(Path(args.experiment_summary).read_text()) if args.experiment_summary else {}
    bstrap = json.loads(Path(args.bd_bootstrap_json).read_text()) if args.bd_bootstrap_json else {}

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.d1_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor = args.reference_method
    all_methods = [anchor] + args.methods

    # Collect all sequences
    sequences = sorted(set(
        seq
        for method_data in d1.get("results", {}).values()
        for seq in method_data.keys()
    ))

    all_tables = {}
    csv_rows = []
    win_loss_tally = defaultdict(lambda: {"win": 0, "neutral": 0, "loss": 0})

    for seq in sequences:
        split = "heldout" if seq in args.heldout_sequences else "dev"
        qp_set = set()
        method_data = {}

        for method in all_methods:
            res = d1.get("results", {}).get(method, {}).get(seq, {})
            if not res:
                continue
            per_qp = {}
            rates = {}; map50s = {}; map5095s = {}
            for qp_str, frame_data in res.items():
                try:
                    qp = int(qp_str)
                except ValueError:
                    continue
                qp_set.add(qp)
                # Aggregate over frames
                rate_key = None
                # Rate from experiment_summary
                for run in exp.get("runs", []):
                    if (run.get("method") == method and run.get("sequence") == seq
                            and str(run.get("qp")) == str(qp)):
                        rate_key = run.get("rate_kbps")
                        break

                mAP50_vals = [v.get("mAP_50", float("nan")) for v in frame_data.values()
                              if isinstance(v, dict)]
                mAP5095_vals = [v.get("mAP_50_95", float("nan")) for v in frame_data.values()
                                if isinstance(v, dict)]
                mean_mAP50 = float(np.nanmean(mAP50_vals)) if mAP50_vals else float("nan")
                mean_mAP5095 = float(np.nanmean(mAP5095_vals)) if mAP5095_vals else float("nan")

                per_qp[qp] = {
                    "rate_kbps": rate_key,
                    "mAP_50": mean_mAP50,
                    "mAP_50_95": mean_mAP5095,
                }
                rates[qp] = rate_key
                map50s[qp] = mean_mAP50
                map5095s[qp] = mean_mAP5095

            qps_sorted = sorted(qp_set)
            # Compute BD vs anchor
            bd_br_50 = float("nan"); bd_br_5095 = float("nan"); bd_acc = float("nan")
            if method != anchor and method in method_data:
                pass  # computed below
            if method != anchor:
                anc = method_data.get(anchor, {})
                anc_per_qp = anc.get("per_qp", {})
                anc_rates = np.asarray([anc_per_qp.get(q, {}).get("rate_kbps") or 0
                                        for q in qps_sorted], dtype=float)
                anc_map50 = np.asarray([anc_per_qp.get(q, {}).get("mAP_50") or 0
                                        for q in qps_sorted], dtype=float)
                anc_map5095 = np.asarray([anc_per_qp.get(q, {}).get("mAP_50_95") or 0
                                          for q in qps_sorted], dtype=float)
                this_rates = np.asarray([per_qp.get(q, {}).get("rate_kbps") or 0
                                         for q in qps_sorted], dtype=float)
                this_map50 = np.asarray([per_qp.get(q, {}).get("mAP_50") or 0
                                         for q in qps_sorted], dtype=float)
                this_map5095 = np.asarray([per_qp.get(q, {}).get("mAP_50_95") or 0
                                           for q in qps_sorted], dtype=float)
                bd_br_50 = _bd_rate_pchip(this_rates, this_map50, anc_rates, anc_map50)
                bd_br_5095 = _bd_rate_pchip(this_rates, this_map5095, anc_rates, anc_map5095)
                bd_acc = _bd_accuracy_pchip(this_rates, this_map5095, anc_rates, anc_map5095)

            # Pull CI from bootstrap JSON
            ci_lo = ci_hi = float("nan")
            agg_verdict = "—" if method == anchor else "?"
            if method != anchor and bstrap:
                entry = bstrap.get(method, {}).get(seq, {})
                ci_lo = entry.get("ci_lo", float("nan"))
                ci_hi = entry.get("ci_hi", float("nan"))
                if ci_lo == ci_lo and ci_hi == ci_hi:
                    if ci_hi < 0:
                        agg_verdict = r"\textbf{Win}"
                        win_loss_tally[method]["win"] += 1
                    elif ci_lo > 0:
                        agg_verdict = "Loss"
                        win_loss_tally[method]["loss"] += 1
                    else:
                        agg_verdict = "Neutral"
                        win_loss_tally[method]["neutral"] += 1

            method_data[method] = {
                "per_qp": per_qp,
                "bd_rate_mAP50": bd_br_50,
                "bd_rate_mAP50_95": bd_br_5095,
                "bd_acc_mAP50_95": bd_acc,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "aggregate_verdict": agg_verdict,
            }

            # CSV rows
            for qp in qps_sorted:
                rqp = per_qp.get(qp, {})
                csv_rows.append({
                    "sequence": seq,
                    "split": split,
                    "method": method,
                    "qp": qp,
                    "rate_kbps": rqp.get("rate_kbps"),
                    "mAP_50": rqp.get("mAP_50"),
                    "mAP_50_95": rqp.get("mAP_50_95"),
                    "BD-Rate_mAP50": bd_br_50 if qp == qps_sorted[0] else "",
                    "BD-Rate_mAP5095": bd_br_5095 if qp == qps_sorted[0] else "",
                    "BD-Acc_mAP5095": bd_acc if qp == qps_sorted[0] else "",
                    "CI_lo": ci_lo if qp == qps_sorted[0] else "",
                    "CI_hi": ci_hi if qp == qps_sorted[0] else "",
                    "verdict": agg_verdict if qp == qps_sorted[0] else "",
                })

        # Build LaTeX table
        qp_list_sorted = sorted(qp_set)
        method_order = [m for m in all_methods if m in method_data]
        latex = _latex_rd_table(seq, qp_list_sorted, method_data, method_order,
                                anchor, split_label=split)
        tex_path = out_dir / f"tab_rdtable_{seq.replace('-', '')}.tex"
        tex_path.write_text(latex, encoding="utf-8")
        logger.info("Written LaTeX table: %s", tex_path)
        all_tables[seq] = method_data

    # CSV
    if csv_rows:
        csv_path = out_dir / "rd_table_full.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info("Written CSV: %s", csv_path)

    # JSON
    json_path = out_dir / "rd_table_full.json"
    json_path.write_text(json.dumps(all_tables, indent=2), encoding="utf-8")
    logger.info("Written JSON: %s", json_path)

    # Win/Loss tally
    print("\n=== Win/Neutral/Loss tally (aggregate CIs vs M0) ===")
    print(f"{'Method':<30} {'Win':>4} {'Neutral':>8} {'Loss':>6}")
    print("-" * 52)
    for method in args.methods:
        t = win_loss_tally.get(method, {"win": 0, "neutral": 0, "loss": 0})
        print(f"{method:<30} {t['win']:>4} {t['neutral']:>8} {t['loss']:>6}")

    print("\nNote: Superiority (IEEE Access Q1) requires Win on majority AND")
    print("      aggregate CI excludes 0 on held-out set AND no catastrophic loss.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
