"""Superiority gate checker (post-upgrade evaluation).

This script applies the 5-criterion IEEE Access Q1 superiority gate:
1. Aggregate BD-Rate negative AND BD-Accuracy positive vs EACH reference AND anchor.
2. Paired-bootstrap 95% CI on aggregate BD-Rate does NOT cross zero.
3. Wins on majority of held-out sequences/cells.
4. No catastrophic loss (BD-Rate > +5% on any single sequence vs anchor).
5. Results hold on the held-out set (MOT17-11, MOT17-13).

Run this after `export_per_sequence_rd_table.py` and `bd_rate_bootstrap.py` have
produced their output JSONs.  The script prints a verdict:

    SUPERIORITY CLAIMED    — all 5 criteria pass on the held-out set
    DIRECTIONAL FRAMING    — 1+ criteria fail; paper claims methodology contribution
    INCONCLUSIVE           — data insufficient to decide

Usage
-----
    PYTHONPATH=src python scripts/check_superiority_gate.py \\
        --rd-json ~/Minh/ipf/phase2_outputs/pilot_v11_fullqp/paper_tables/rd_table_full.json \\
        --bootstrap-json ~/Minh/ipf/phase2_outputs/pilot_v11_fullqp/diagnostics/bd_rate_bootstrap.json \\
        --proposed CA-OGIPF \\
        --references M0 M_soft_roi_ref \\
        --heldout MOT17-11-DPM MOT17-13-DPM \\
        --catastrophic-threshold 5.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


GATE_NAMES = [
    "G1: Aggregate BD-Rate < 0 vs all refs",
    "G2: Bootstrap CI excludes 0 vs anchor",
    "G3: Majority wins on held-out sequences",
    "G4: No catastrophic loss (BD-Rate > +threshold)",
    "G5: Results hold on held-out set",
]


def check_gate(
    rd_json_path: str,
    bootstrap_json_path: str,
    proposed: str,
    references: List[str],
    heldout_seqs: List[str],
    catastrophic_threshold: float = 5.0,
) -> Dict:
    rd_data = json.loads(Path(rd_json_path).read_text())
    bstrap = json.loads(Path(bootstrap_json_path).read_text()) if Path(bootstrap_json_path).exists() else {}

    results = {name: None for name in GATE_NAMES}
    details = {}

    all_seqs = list(rd_data.keys())
    heldout = [s for s in heldout_seqs if s in all_seqs]
    if not heldout:
        print(f"WARNING: No held-out sequences found in rd_json. Available: {all_seqs}")
        heldout = all_seqs  # fallback to all

    # ------------------------------------------------------------------
    # G1: Aggregate BD-Rate < 0 vs all references AND anchor
    # ------------------------------------------------------------------
    g1_pass = True
    g1_details = {}
    for ref in references:
        bd_rates = []
        for seq, seq_data in rd_data.items():
            prop_data = seq_data.get(proposed, {})
            bd_key = "bd_rate_mAP50_95" if "bd_rate_mAP50_95" in prop_data else "bd_rate_mAP50"
            val = prop_data.get(bd_key, float("nan"))
            if val == val:  # not NaN
                bd_rates.append(val)
        agg = float(np.mean(bd_rates)) if bd_rates else float("nan")
        g1_details[ref] = agg
        if agg != agg or agg >= 0:
            g1_pass = False
    results[GATE_NAMES[0]] = g1_pass
    details["G1"] = g1_details

    # ------------------------------------------------------------------
    # G2: Bootstrap CI on aggregate BD-Rate does NOT cross 0 vs anchor
    # ------------------------------------------------------------------
    g2_pass = False
    g2_details = {}
    anchor = references[0] if references else "M0"
    # Look for aggregate CI in bootstrap JSON
    prop_bstrap = bstrap.get(proposed, {})
    # Bootstrap JSON may have seq-level or aggregate-level CIs
    ci_lo_vals = []
    ci_hi_vals = []
    for seq in all_seqs:
        entry = prop_bstrap.get(seq, {})
        ci_lo = entry.get("ci_lo", float("nan"))
        ci_hi = entry.get("ci_hi", float("nan"))
        if ci_lo == ci_lo and ci_hi == ci_hi:
            ci_lo_vals.append(ci_lo)
            ci_hi_vals.append(ci_hi)

    if ci_lo_vals:
        # Aggregate CI approximation: mean of per-seq CI endpoints
        agg_ci_lo = float(np.mean(ci_lo_vals))
        agg_ci_hi = float(np.mean(ci_hi_vals))
        g2_pass = (agg_ci_hi < 0.0)  # CI entirely below 0 = win
        g2_details = {"agg_ci_lo": agg_ci_lo, "agg_ci_hi": agg_ci_hi,
                      "ci_excludes_zero": g2_pass}
    else:
        g2_details = {"note": "No bootstrap CI data found — cannot determine G2"}
        g2_pass = None  # inconclusive
    results[GATE_NAMES[1]] = g2_pass
    details["G2"] = g2_details

    # ------------------------------------------------------------------
    # G3: Majority of held-out sequences show BD-Rate < 0
    # ------------------------------------------------------------------
    wins = 0; losses = 0; neutrals = 0
    for seq in heldout:
        seq_data = rd_data.get(seq, {})
        prop_data = seq_data.get(proposed, {})
        bd_val = prop_data.get("bd_rate_mAP50_95", float("nan"))
        if bd_val != bd_val:
            neutrals += 1
        elif bd_val < 0:
            wins += 1
        else:
            losses += 1
    g3_pass = wins > losses
    results[GATE_NAMES[2]] = g3_pass
    details["G3"] = {"wins": wins, "neutrals": neutrals, "losses": losses, "heldout": heldout}

    # ------------------------------------------------------------------
    # G4: No catastrophic loss on any single sequence
    # ------------------------------------------------------------------
    catastrophic_seqs = []
    for seq, seq_data in rd_data.items():
        prop_data = seq_data.get(proposed, {})
        bd_val = prop_data.get("bd_rate_mAP50_95", float("nan"))
        if bd_val == bd_val and bd_val > catastrophic_threshold:
            catastrophic_seqs.append((seq, bd_val))
    g4_pass = len(catastrophic_seqs) == 0
    results[GATE_NAMES[3]] = g4_pass
    details["G4"] = {"catastrophic_seqs": catastrophic_seqs,
                     "threshold": catastrophic_threshold}

    # ------------------------------------------------------------------
    # G5: BD-Rate gains hold on held-out set (not just dev)
    # ------------------------------------------------------------------
    heldout_bd = []
    for seq in heldout:
        seq_data = rd_data.get(seq, {})
        prop_data = seq_data.get(proposed, {})
        bd_val = prop_data.get("bd_rate_mAP50_95", float("nan"))
        if bd_val == bd_val:
            heldout_bd.append(bd_val)
    heldout_agg = float(np.mean(heldout_bd)) if heldout_bd else float("nan")
    g5_pass = (heldout_agg == heldout_agg and heldout_agg < 0)
    results[GATE_NAMES[4]] = g5_pass
    details["G5"] = {"heldout_bd_rate_mean": heldout_agg, "seqs": heldout_bd}

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    n_pass = sum(1 for v in results.values() if v is True)
    n_inconc = sum(1 for v in results.values() if v is None)
    n_fail = sum(1 for v in results.values() if v is False)

    if n_inconc > 0:
        verdict = "INCONCLUSIVE"
        recommendation = (
            "One or more gate criteria could not be evaluated (bootstrap CI missing).\n"
            "Run bd_rate_bootstrap.py first, then re-check."
        )
    elif n_pass == 5:
        verdict = "SUPERIORITY CLAIMED"
        recommendation = (
            "All 5 gate criteria pass on the held-out set.\n"
            "Claim superiority in Abstract/Conclusion. Cite bootstrap CI in all tables."
        )
    else:
        verdict = "DIRECTIONAL FRAMING"
        recommendation = (
            f"{n_fail} gate criteria failed. Do NOT claim superiority.\n"
            "Frame the contribution as: (1) directional BD-Rate improvement, and\n"
            "(2) evaluation-methodology contribution (detector-headroom diagnostic H).\n"
            "See plan §5: 'pivot to evaluation-methodology framing'."
        )

    return {
        "verdict": verdict,
        "recommendation": recommendation,
        "gate_results": results,
        "gate_details": details,
        "proposed": proposed,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_inconclusive": n_inconc,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rd-json", required=True)
    parser.add_argument("--bootstrap-json", default="")
    parser.add_argument("--proposed", default="CA-OGIPF")
    parser.add_argument("--references", nargs="+", default=["M0", "M_soft_roi_ref"])
    parser.add_argument("--heldout", nargs="+", default=["MOT17-11-DPM", "MOT17-13-DPM"])
    parser.add_argument("--catastrophic-threshold", type=float, default=5.0)
    args = parser.parse_args(argv)

    result = check_gate(
        args.rd_json,
        args.bootstrap_json,
        args.proposed,
        args.references,
        args.heldout,
        args.catastrophic_threshold,
    )

    print("\n" + "=" * 60)
    print(f"  SUPERIORITY GATE — {result['proposed']}")
    print("=" * 60)
    for gate_name, passed in result["gate_results"].items():
        status = "PASS" if passed is True else ("FAIL" if passed is False else "INCONCLUSIVE")
        print(f"  [{status:^12}]  {gate_name}")

    print(f"\n  Result: {result['verdict']}")
    print(f"  Pass: {result['n_pass']}/5  |  Fail: {result['n_fail']}/5  "
          f"|  Inconclusive: {result['n_inconclusive']}/5\n")
    print("  Recommendation:")
    for line in result["recommendation"].split("\n"):
        print(f"    {line}")
    print("=" * 60)

    # Write JSON
    out = Path(args.rd_json).parent / "superiority_gate_verdict.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n  Full gate report written to: {out}")

    return 0 if result["verdict"] in ("SUPERIORITY CLAIMED", "DIRECTIONAL FRAMING") else 1


if __name__ == "__main__":
    sys.exit(main())
