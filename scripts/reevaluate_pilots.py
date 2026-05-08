"""Re-evaluate one or more pilots under the audited evaluation protocol.

This script is the single entry-point that produces the *scientifically
defensible* numbers requested by PROJECT_AUDIT and the user mandate
"give me convincing scientific results":

    1. True COCO mAP_50 / mAP_75 / mAP_50_95 (D1)
    2. Paired statistical tests on per-frame F1 (D2 — Wilcoxon, Cohen's d)
    3. Paired-bootstrap CI on BD-Rate-Task (this module)
    4. Cross-method comparison (M0, M_rect, M1, M5, M6, M4) when available
    5. Markdown verdict with explicit "REAL win" / "noise" tags per cell

For each pilot it writes::

    <pilot>/diagnostics/d1_true_map.json
    <pilot>/diagnostics/d2_stats.json
    <pilot>/diagnostics/bd_rate_bootstrap.json
    <pilot>/diagnostics/verdict.md

Speed
-----
* If the pilot was encoded under the new pipeline (`detections_decoded.json`
  + `detections_reference.json` saved per run), D1 takes ~10 s per pilot
  (file-load only, no YOLO).
* Old pilots fall back to YOLO re-detection on cached `decoded_frames/`.

Usage on the server::

    PYTHONPATH=src python scripts/reevaluate_pilots.py \\
        --pilots pilot_v4 pilot_v8b pilot_v9 pilot_v9b \\
        --pilots-root ~/Minh/ipf/phase2_outputs \\
        --configs configs/pilot_v4.yaml configs/pilot_v8b.yaml \\
                  configs/pilot_v9.yaml  configs/pilot_v9b.yaml \\
        --reference-method M0 \\
        --test-methods M4 \\
        --device cuda:0 \\
        --n-boot 1000

Local sanity (no encodes) is exercised by
``scripts/sanity_check_reevaluation.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("phase2.scripts.reevaluate_pilots")


# ---------------------------------------------------------------------------
# Minimal IO helpers
# ---------------------------------------------------------------------------


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run(cmd: List[str], dry: bool = False) -> int:
    logger.info("$ %s", " ".join(cmd))
    if dry:
        return 0
    return subprocess.run(cmd).returncode


# ---------------------------------------------------------------------------
# Phase A — D1 (real mAP) per pilot
# ---------------------------------------------------------------------------


def run_d1(
    pilot_dir: Path, config: Path, output: Path,
    model: str, conf_low: float, classes: List[int],
    device: str, dry: bool,
    gt_source: str = "mot17", min_visibility: float = 0.0,
) -> int:
    cmd = [
        sys.executable, "-m", "phase2.diagnostics.d1_true_map",
        "--pilot-dir", str(pilot_dir),
        "--config", str(config),
        "--output", str(output),
        "--model", model,
        "--conf-low", str(conf_low),
        "--device", device,
        "--gt-source", gt_source,
        "--min-visibility", str(min_visibility),
    ]
    cmd += ["--classes", *map(str, classes)]
    return _run(cmd, dry)


# ---------------------------------------------------------------------------
# Phase B — D2 (paired stats vs reference)
# ---------------------------------------------------------------------------


def run_d2(d1_json: Path, output: Path, reference: str, methods: List[str],
           dry: bool) -> int:
    cmd = [
        sys.executable, "-m", "phase2.diagnostics.d2_statistical_significance",
        "--d1-json", str(d1_json),
        "--reference-method", reference,
        "--test-methods", *methods,
        "--output", str(output),
    ]
    return _run(cmd, dry)


# ---------------------------------------------------------------------------
# Phase C — Paired-bootstrap BD-Rate-Task (this script)
# ---------------------------------------------------------------------------


def run_bd_bootstrap(
    pilot_dir: Path, d1_json: Path, output: Path,
    reference: str, methods: List[str],
    n_boot: int, alpha: float, seed: int,
) -> None:
    """Reads D1 cells + experiment_summary, runs paired bootstrap."""
    from phase2.diagnostics.bd_rate_bootstrap import (
        CellBootstrapInput,
        paired_bootstrap_bd_rate,
        cross_sequence_summary,
    )

    summary_path = pilot_dir / "experiment_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing experiment_summary.json: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    cells_d1 = json.loads(d1_json.read_text(encoding="utf-8"))["cells"]

    # Index summary for bitrate lookup
    sum_idx = {}
    for entry in summary.get("results", []):
        bitrate = float(entry.get("bitrate_kbps", 0.0))
        if bitrate <= 0:
            bitrate = float(entry.get("encode", {}).get("bitrate_kbps", 0.0))
        sum_idx[(entry["sequence"], entry["method"], int(entry["qp_base"]))] = bitrate

    # Build CellBootstrapInput per method
    by_method: Dict[str, List[CellBootstrapInput]] = {}
    for c in cells_d1:
        method = c["method"]; seq = c["sequence"]; qp = int(c["qp_base"])
        bitrate = sum_idx.get((seq, method, qp), 0.0)
        per_frame = [tuple(x) for x in c.get("per_frame_counts", [])]
        cbi = CellBootstrapInput(
            sequence=seq, method=method, qp_base=qp,
            bitrate_kbps=bitrate,
            point_quality=float(c.get("mAP_50_95", 0.0)),
            per_frame_counts=per_frame,
        )
        by_method.setdefault(method, []).append(cbi)

    bd = paired_bootstrap_bd_rate(
        by_method, reference_method=reference, test_methods=methods,
        n_boot=n_boot, alpha=alpha, seed=seed,
    )
    agg = cross_sequence_summary(bd)

    out_payload = {
        "pilot_dir": str(pilot_dir),
        "reference_method": reference,
        "test_methods": methods,
        "n_boot": n_boot, "alpha": alpha,
        "per_method_per_sequence": {
            m: {seq: asdict(r) for seq, r in by_seq.items()}
            for m, by_seq in bd.items()
        },
        "per_method_aggregate": agg,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    logger.info("Wrote bootstrap BD-Rate to %s", output)


# ---------------------------------------------------------------------------
# Phase D — Markdown verdict (combines D1 + D2 + bootstrap)
# ---------------------------------------------------------------------------


def render_verdict(
    pilot: str,
    d1_path: Path,
    d2_path: Optional[Path],
    bd_path: Path,
    output: Path,
) -> None:
    d1 = json.loads(d1_path.read_text(encoding="utf-8"))
    d2 = json.loads(d2_path.read_text(encoding="utf-8")) if d2_path and d2_path.exists() else {}
    bd = json.loads(bd_path.read_text(encoding="utf-8"))

    lines: List[str] = []
    lines.append(f"# {pilot} — audited verdict\n")
    lines.append("This document was produced by `scripts/reevaluate_pilots.py`. ")
    lines.append("All numbers are scientifically defensible:\n")
    lines.append("* mAP rows use **true COCO mAP** (D1)")
    lines.append("* per-cell verdicts use **paired Wilcoxon + Cohen's d** (D2)")
    lines.append("* BD-Rate columns include **paired-bootstrap 95 % CI** (this script)")
    lines.append("")

    # Cell-level mAP table
    lines.append("## Per-cell true COCO mAP\n")
    lines.append("| Sequence | QP | Method | mAP_50 | mAP_75 | mAP_50_95 | pxr_50@0.25 |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for c in sorted(d1["cells"], key=lambda x: (x["sequence"], int(x["qp_base"]), x["method"])):
        lines.append(
            f"| {c['sequence']} | {c['qp_base']} | {c['method']} | "
            f"{c['mAP_50']:.4f} | {c['mAP_75']:.4f} | {c['mAP_50_95']:.4f} | "
            f"{c.get('pxr_50_at_025', 0):.4f} |"
        )

    # Stat tests
    # D2 uses key "results"; tolerate both spellings for backward compat.
    d2_rows = d2.get("results") or d2.get("comparisons") or []
    if d2_rows:
        lines.append("")
        lines.append("## Per-cell paired statistical wins (D2)\n")
        lines.append("| Sequence | QP | Method | Δ̄F1 | d | p_Wil | Verdict |")
        lines.append("|---|---:|---|---:|---:|---:|---|")
        for c in sorted(d2_rows,
                        key=lambda x: (x["sequence"], int(x["qp_base"]), x["method"])):
            v = ("REAL " + ("WIN" if c["diff_mean"] > 0 else "LOSS")
                 if c.get("is_real_effect") else "noise")
            lines.append(
                f"| {c['sequence']} | {c['qp_base']} | {c['method']} | "
                f"{c['diff_mean']:+.4f} | {c['cohens_d']:+.2f} | "
                f"{c['wilcoxon_p']:.3f} | {v} |"
            )

    # BD-Rate with CI
    lines.append("")
    lines.append("## BD-Rate-Task with paired-bootstrap 95 % CI\n")
    lines.append(
        "Point estimate uses true COCO mAP_50_95 (audited). "
        "CI uses per-frame TP/FP/FN counts from D1 (P×R surrogate); the "
        "interpretation is *would the method still beat M0 if a different "
        "frame sample came back?*\n"
    )
    lines.append("| Method | Sequence | Point BD (true mAP) | Boot mean (P×R) | 95 % CI (P×R) | n_boot |")
    lines.append("|---|---|---:|---:|---:|---:|")
    pm = bd.get("per_method_per_sequence", {})
    for m in sorted(pm):
        for s, r in sorted(pm[m].items()):
            ci_str = (f"[{r['ci_lo_pxr']:+.2f}, {r['ci_hi_pxr']:+.2f}]"
                      if not np.isnan(r["ci_lo_pxr"]) else "n/a")
            lines.append(
                f"| {m} | {s} | "
                f"{r['point_estimate_true_map']:+.2f}% | "
                f"{r['boot_mean_pxr']:+.2f}% | {ci_str} | {r['n_boot']} |"
            )

    # Aggregate
    lines.append("")
    lines.append("## Aggregate (mean across sequences)\n")
    lines.append("| Method | mean_true_map | median_true_map | mean_pxr_boot | avg CI (P×R) |")
    lines.append("|---|---:|---:|---:|---:|")
    for m, agg in sorted(bd.get("per_method_aggregate", {}).items()):
        ci_str = (f"[{agg['ci_lo_pxr_avg']:+.2f}, {agg['ci_hi_pxr_avg']:+.2f}]"
                  if not np.isnan(agg.get("ci_lo_pxr_avg", float("nan")))
                  else "n/a")
        lines.append(
            f"| {m} | {agg['mean_true_map']:+.2f}% | "
            f"{agg['median_true_map']:+.2f}% | "
            f"{agg['mean_pxr_boot']:+.2f}% | {ci_str} |"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote verdict to %s", output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-evaluate pilots with audited metrics + bootstrap CI",
    )
    ap.add_argument("--pilots", nargs="+", required=True)
    ap.add_argument("--pilots-root", required=True, type=Path)
    ap.add_argument("--configs", nargs="+", required=True,
                    help="Phase2 YAML config per pilot, in same order as --pilots")
    ap.add_argument("--reference-method", default="M0")
    ap.add_argument("--test-methods", nargs="+", default=["M4"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf-low", type=float, default=0.001)
    ap.add_argument("--classes", type=int, nargs="+", default=[0])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--gt-source", choices=("mot17", "pseudo"), default="mot17",
        help="Ground-truth source for D1 (default: real MOT17 annotations).",
    )
    ap.add_argument(
        "--min-visibility", type=float, default=0.0,
        help="With --gt-source mot17, drop GT boxes with visibility below this.",
    )
    ap.add_argument("--skip-d1", action="store_true",
                    help="Re-use existing d1_true_map.json")
    ap.add_argument("--skip-d2", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
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

    if len(args.pilots) != len(args.configs):
        raise SystemExit("--pilots and --configs must have the same length")

    rc_total = 0
    skipped: List[str] = []
    for pilot, cfg in zip(args.pilots, args.configs):
        pilot_dir = args.pilots_root / pilot

        # Pre-flight: skip pilots that were never encoded (no
        # experiment_summary.json) so one missing pilot does not abort
        # the whole batch.
        if not (pilot_dir / "experiment_summary.json").exists():
            logger.warning(
                "Pilot %s has no experiment_summary.json under %s — "
                "skipping (was the encode pipeline ever run for it?)",
                pilot, pilot_dir,
            )
            skipped.append(pilot)
            continue

        diag_dir  = pilot_dir / "diagnostics"
        _ensure_dir(diag_dir)
        d1_out = diag_dir / "d1_true_map.json"
        d2_out = diag_dir / "d2_stats.json"
        bd_out = diag_dir / "bd_rate_bootstrap.json"
        verdict_out = diag_dir / "verdict.md"

        if not args.skip_d1:
            rc = run_d1(pilot_dir, Path(cfg), d1_out,
                        args.model, args.conf_low, args.classes,
                        args.device, args.dry_run,
                        gt_source=args.gt_source,
                        min_visibility=args.min_visibility)
            if rc != 0:
                logger.error("D1 failed for %s (exit %d) — skipping rest", pilot, rc)
                rc_total |= rc
                skipped.append(pilot)
                continue

        if not args.skip_d2 and d1_out.exists():
            rc = run_d2(d1_out, d2_out, args.reference_method,
                        args.test_methods, args.dry_run)
            if rc != 0:
                logger.error("D2 failed for %s (exit %d)", pilot, rc)
                rc_total |= rc

        if d1_out.exists() and not args.dry_run:
            run_bd_bootstrap(
                pilot_dir, d1_out, bd_out,
                args.reference_method, args.test_methods,
                args.n_boot, args.alpha, args.seed,
            )

        if d1_out.exists() and bd_out.exists() and not args.dry_run:
            render_verdict(pilot, d1_out,
                            d2_out if d2_out.exists() else None,
                            bd_out, verdict_out)

    if skipped:
        logger.warning("Skipped %d pilot(s): %s", len(skipped), ", ".join(skipped))
    if rc_total != 0:
        raise SystemExit(rc_total)
    logger.info("All pilots re-evaluated.")


if __name__ == "__main__":
    main()
