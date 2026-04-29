"""Compare two or more Phase 2 ``experiment_summary.json`` files.

For every (sequence, method) pair it produces:

* A per-QP table of {bitrate_kbps, mAP50, ΔRate %, ΔmAP} relative to M0 of
  the **same pilot** (M0 is the anchor in each pilot).
* The Bjontegaard BD-Rate-Task metric (Bjontegaard 2001 piecewise-cubic
  interpolation, with mAP as the quality axis instead of PSNR).
* A win/loss tally vs M0 (per-QP comparison: M4 wins iff bitrate ≤ M0
  AND mAP ≥ M0 within tolerances).
* A cross-pilot summary so you can see how pilot_v3 (oracle-direct) and
  pilot_v4 (LiteQP) stack up against pilot_v1.

Run with::

    python phase2/scripts/compare_pilots.py \
        ~/Minh/ipf/phase2_outputs/pilot_v1/experiment_summary.json \
        ~/Minh/ipf/phase2_outputs/pilot_v3/experiment_summary.json \
        ~/Minh/ipf/phase2_outputs/pilot_v4/experiment_summary.json

You can pass a label after each path with ``--labels``::

    python phase2/scripts/compare_pilots.py \
        a.json b.json c.json \
        --labels pilot_v1 pilot_v3 pilot_v4 \
        --output report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("phase2.compare_pilots")


# ---------------------------------------------------------------------------
# BD-Rate-Task computation (Bjontegaard 2001, piecewise-cubic on log-rate)
# ---------------------------------------------------------------------------

def bd_rate_task(rates_a: List[float], maps_a: List[float],
                 rates_b: List[float], maps_b: List[float]) -> Optional[float]:
    """Return BD-Rate using mAP as the quality axis. Negative = method B wins.

    Inputs
    ------
    rates_a, rates_b : list of bitrates (kbps), strictly positive.
    maps_a,  maps_b  : list of mAP values, in [0, 1] (or any monotone proxy).

    Implementation follows the Bjontegaard 2001 / VCEG-M33 reference:
    fit a cubic polynomial mAP → log10(rate) for each method, integrate
    the difference over the **overlap range** of mAP, and convert back
    to a percentage rate change.

    Returns ``None`` if the mAP overlap is empty or there are < 4 points.
    """
    if min(len(rates_a), len(maps_a), len(rates_b), len(maps_b)) < 4:
        return None
    ra = np.asarray(rates_a, dtype=np.float64)
    ma = np.asarray(maps_a,  dtype=np.float64)
    rb = np.asarray(rates_b, dtype=np.float64)
    mb = np.asarray(maps_b,  dtype=np.float64)

    if (ra <= 0).any() or (rb <= 0).any():
        return None

    # Sort by mAP (the "quality" axis).
    oa = np.argsort(ma); ma_s, ra_s = ma[oa], ra[oa]
    ob = np.argsort(mb); mb_s, rb_s = mb[ob], rb[ob]

    # Cubic polynomial fit:  log10(rate) = p0 + p1·m + p2·m² + p3·m³.
    pa = np.polyfit(ma_s, np.log10(ra_s), deg=3)
    pb = np.polyfit(mb_s, np.log10(rb_s), deg=3)

    lo = max(ma_s.min(), mb_s.min())
    hi = min(ma_s.max(), mb_s.max())
    if hi <= lo:
        return None

    Pa = np.polyint(pa)
    Pb = np.polyint(pb)
    int_a = np.polyval(Pa, hi) - np.polyval(Pa, lo)
    int_b = np.polyval(Pb, hi) - np.polyval(Pb, lo)
    avg_log_diff = (int_b - int_a) / (hi - lo)
    return float((10.0 ** avg_log_diff - 1.0) * 100.0)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class Run:
    sequence: str
    method: str
    qp_base: int
    bitrate_kbps: float
    map50: float
    psnr_y_full: float
    success: bool


def load_summary(path: Path) -> List[Run]:
    """Parse experiment_summary.json (Phase 2 format) into Run records."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("results", data) if isinstance(data, dict) else data
    out = []
    for r in rows:
        out.append(Run(
            sequence=r.get("sequence", "?"),
            method=r.get("method", "?"),
            qp_base=int(r.get("qp_base") or r.get("qp") or 0),
            bitrate_kbps=float(r.get("bitrate_kbps") or 0.0),
            map50=float(r.get("mAP50") or r.get("map50") or 0.0),
            psnr_y_full=float(r.get("psnr_y_full") or r.get("psnr_y_enc") or 0.0),
            success=bool(r.get("success", True)),
        ))
    return out


def index_runs(runs: List[Run]) -> Dict[Tuple[str, str], List[Run]]:
    """Group runs by (sequence, method); each group sorted by qp_base."""
    out: Dict[Tuple[str, str], List[Run]] = {}
    for r in runs:
        out.setdefault((r.sequence, r.method), []).append(r)
    for k in out:
        out[k].sort(key=lambda x: x.qp_base)
    return out


# ---------------------------------------------------------------------------
# Per-pilot analysis
# ---------------------------------------------------------------------------

def analyze_pilot(label: str, runs: List[Run],
                  rate_tol: float = 0.0,
                  map_tol: float = 0.0) -> dict:
    """Return a dict summarising one pilot.

    The "anchor" method is M0 (uniform QP). Every other method is compared
    against M0 at the same (sequence, qp_base).
    """
    by_key = index_runs(runs)
    sequences = sorted({k[0] for k in by_key})
    methods = sorted({k[1] for k in by_key})

    per_seq: List[dict] = []
    method_bd: Dict[str, List[float]] = {m: [] for m in methods if m != "M0"}
    win_count: Dict[str, Dict[str, int]] = {
        m: {"win": 0, "tie": 0, "loss": 0, "n": 0}
        for m in methods if m != "M0"
    }

    for seq in sequences:
        m0 = by_key.get((seq, "M0"), [])
        if not m0:
            logger.warning("[%s] %s missing M0 anchor", label, seq)
            continue
        m0_by_qp = {r.qp_base: r for r in m0}
        seq_entry = {"sequence": seq, "methods": {}}
        for method in methods:
            if method == "M0":
                continue
            others = by_key.get((seq, method), [])
            rows: List[dict] = []
            for r in others:
                anchor = m0_by_qp.get(r.qp_base)
                if anchor is None:
                    continue
                d_rate = (r.bitrate_kbps - anchor.bitrate_kbps) \
                         / max(anchor.bitrate_kbps, 1e-9) * 100.0
                d_map = r.map50 - anchor.map50
                # Win iff: bitrate ≤ anchor AND mAP ≥ anchor (within tolerance).
                if d_rate <= rate_tol and d_map >= -map_tol:
                    if d_rate <= -1e-3 or d_map >= 1e-3:
                        outcome = "win"
                    else:
                        outcome = "tie"
                elif d_rate >= -rate_tol and d_map <= map_tol:
                    outcome = "loss"
                else:
                    outcome = "tie"
                win_count[method][outcome] += 1
                win_count[method]["n"] += 1
                rows.append({
                    "qp_base": r.qp_base,
                    "method_rate_kbps": r.bitrate_kbps,
                    "anchor_rate_kbps": anchor.bitrate_kbps,
                    "delta_rate_pct":   d_rate,
                    "method_map50":  r.map50,
                    "anchor_map50":  anchor.map50,
                    "delta_map50":   d_map,
                    "outcome":        outcome,
                })
            # BD-Rate-Task vs M0 on this sequence
            ra = [m0_by_qp[r.qp_base].bitrate_kbps for r in others
                  if r.qp_base in m0_by_qp]
            ma = [m0_by_qp[r.qp_base].map50 for r in others
                  if r.qp_base in m0_by_qp]
            rb = [r.bitrate_kbps for r in others if r.qp_base in m0_by_qp]
            mb = [r.map50 for r in others if r.qp_base in m0_by_qp]
            bd = bd_rate_task(ra, ma, rb, mb)
            if bd is not None and not np.isnan(bd):
                method_bd[method].append(bd)
            seq_entry["methods"][method] = {
                "bd_rate_task_pct": bd,
                "per_qp": rows,
            }
        per_seq.append(seq_entry)

    summary = {
        "label": label,
        "n_sequences": len(sequences),
        "sequences":  sequences,
        "methods":    methods,
        "per_sequence": per_seq,
        "method_bd_avg": {
            m: (float(np.mean(v)) if v else None)
            for m, v in method_bd.items()
        },
        "win_loss": win_count,
    }
    return summary


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _row(seq: str, method: str, qp: int, rate: float, anc_rate: float,
          dr: float, mp: float, anc_mp: float, dm: float, out_: str) -> str:
    return (f"  {seq:<16} {method:<6} QP{qp:<3} "
            f"{rate:>7.1f} (vs {anc_rate:>7.1f})  ΔR={dr:+6.2f}%   "
            f"mAP={mp:.3f} (vs {anc_mp:.3f})  Δm={dm:+.3f}   {out_}")


def print_pilot_report(summary: dict) -> None:
    print("\n" + "=" * 92)
    print(f"PILOT: {summary['label']}    "
          f"({summary['n_sequences']} sequences, "
          f"methods = {', '.join(summary['methods'])})")
    print("=" * 92)
    for seq_entry in summary["per_sequence"]:
        print(f"\n[{seq_entry['sequence']}]")
        for method, payload in seq_entry["methods"].items():
            bd = payload["bd_rate_task_pct"]
            bd_str = f"{bd:+.2f} %" if bd is not None else "n/a"
            print(f"  ── {method} vs M0    BD-Rate-Task = {bd_str}")
            for r in payload["per_qp"]:
                print(_row(
                    seq_entry["sequence"], method, r["qp_base"],
                    r["method_rate_kbps"], r["anchor_rate_kbps"],
                    r["delta_rate_pct"],
                    r["method_map50"], r["anchor_map50"], r["delta_map50"],
                    r["outcome"].upper(),
                ))
    print("\n── Sequence-averaged BD-Rate-Task ──")
    for m, v in summary["method_bd_avg"].items():
        print(f"  {m:<6}  {('%+.2f %%' % v) if v is not None else 'n/a'}")
    print("\n── Win / Tie / Loss (per-QP, vs M0) ──")
    for m, w in summary["win_loss"].items():
        n = max(w["n"], 1)
        print(f"  {m:<6}  W={w['win']:>2}  T={w['tie']:>2}  L={w['loss']:>2}  "
              f"(win-rate {100*w['win']/n:.1f} %)")


def print_cross_pilot_summary(summaries: List[dict]) -> None:
    print("\n" + "=" * 92)
    print("CROSS-PILOT SUMMARY (sequence-averaged BD-Rate-Task, M4 vs M0)")
    print("=" * 92)
    methods_all = sorted({m for s in summaries for m in s["method_bd_avg"]})
    header = f"{'pilot':<14}" + "".join(f"{m:>14}" for m in methods_all)
    print(header)
    print("-" * len(header))
    for s in summaries:
        row = f"{s['label']:<14}"
        for m in methods_all:
            v = s["method_bd_avg"].get(m)
            row += f"{('%+.2f %%' % v) if v is not None else 'n/a':>14}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Phase 2 experiment summaries (BD-Rate-Task + per-QP).")
    parser.add_argument("paths", nargs="+",
                        help="One or more experiment_summary.json files.")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Optional labels (defaults to parent directory name)")
    parser.add_argument("--rate-tolerance", type=float, default=0.5,
                        help="Δ-rate %% tolerance for tie classification (default 0.5)")
    parser.add_argument("--map-tolerance",  type=float, default=0.005,
                        help="Δ-mAP   tolerance for tie classification (default 0.005)")
    parser.add_argument("--output", default="",
                        help="Optional JSON path for the structured report.")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    paths = [Path(p).expanduser().resolve() for p in args.paths]
    labels = args.labels or [p.parent.name for p in paths]
    if len(labels) != len(paths):
        raise SystemExit("--labels count must match number of paths")

    summaries: List[dict] = []
    for path, label in zip(paths, labels):
        if not path.exists():
            logger.error("Missing summary: %s", path)
            continue
        runs = load_summary(path)
        s = analyze_pilot(label, runs,
                           rate_tol=args.rate_tolerance,
                           map_tol=args.map_tolerance)
        s["source_path"] = str(path)
        summaries.append(s)
        print_pilot_report(s)

    if len(summaries) >= 2:
        print_cross_pilot_summary(summaries)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"pilots": summaries}, f, indent=2)
        logger.info("Wrote structured report to %s", args.output)


if __name__ == "__main__":
    main()
