"""Run all Phase 1 diagnostics for one or more pilots.

This is the convenience wrapper that produces:

* ``<pilot>/diagnostics/d1_true_map.json``       (per-cell true COCO mAP)
* ``<pilot>/diagnostics/d2_stats.json[.md]``     (paired tests vs M0)
* ``<pilot>/diagnostics/d3_phi.json[.md]``       (φ distribution)

A combined cross-pilot D2 report is also written if ``--combined`` is given.

Usage on the server::

    PYTHONPATH=src python scripts/run_phase1_diagnostics.py \\
        --pilots pilot_v4 pilot_v8b pilot_v9 \\
        --pilots-root ~/Minh/ipf/phase2_outputs \\
        --configs configs/pilot_v4.yaml configs/pilot_v8b.yaml configs/pilot_v9.yaml \\
        --saliency-dir ~/Minh/ipf/phase3_outputs/saliency \\
        --sequences MOT17-02-DPM MOT17-04-DPM MOT17-09-DPM \\
        --device cuda:0 \\
        --combined-output ~/Minh/ipf/phase2_outputs/diagnostics_summary

Set ``--skip-d1`` if D1 already ran (re-detection is the slow step).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger("phase2.scripts.run_phase1_diagnostics")


def _run(cmd: List[str], dry: bool) -> int:
    logger.info("$ %s", " ".join(cmd))
    if dry:
        return 0
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D1+D2+D3 for the listed pilots")
    parser.add_argument("--pilots", nargs="+", required=True,
                        help="Pilot directory names (relative to --pilots-root)")
    parser.add_argument("--pilots-root", required=True, type=Path)
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Phase2 YAML config per pilot, in the same order")
    parser.add_argument("--saliency-dir", type=Path, required=False,
                        help="Parent dir of per-sequence φ_oracle .npy files")
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--reference-method", default="M0")
    parser.add_argument("--test-methods", nargs="+", default=["M4"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf-low", type=float, default=0.001)
    parser.add_argument("--classes", type=int, nargs="+", default=[0])
    parser.add_argument("--combined-output", type=Path, default=None,
                        help="If set, write a combined D2 across all pilots")
    parser.add_argument("--skip-d1", action="store_true")
    parser.add_argument("--skip-d2", action="store_true")
    parser.add_argument("--skip-d3", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if len(args.pilots) != len(args.configs):
        raise SystemExit("--pilots and --configs must have the same length")

    py = sys.executable
    rc_all = 0
    d1_jsons: List[Path] = []

    for name, cfg in zip(args.pilots, args.configs):
        pilot_dir = args.pilots_root / name
        diag_dir = pilot_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        d1_out = diag_dir / "d1_true_map.json"
        d2_out = diag_dir / "d2_stats.json"
        d3_out = diag_dir / "d3_phi.json"

        if not args.skip_d1:
            cmd = [
                py, "-m", "phase2.diagnostics.d1_true_map",
                "--pilot-dir", str(pilot_dir),
                "--config", str(cfg),
                "--output", str(d1_out),
                "--model", args.model,
                "--conf-low", str(args.conf_low),
                "--device", args.device,
            ]
            cmd += ["--classes", *map(str, args.classes)]
            rc = _run(cmd, args.dry_run)
            if rc != 0:
                logger.error("D1 failed for %s (exit %d)", name, rc)
                rc_all |= rc
                continue
        d1_jsons.append(d1_out)

        if not args.skip_d2:
            cmd = [
                py, "-m", "phase2.diagnostics.d2_statistical_significance",
                "--d1-json", str(d1_out),
                "--reference-method", args.reference_method,
                "--test-methods", *args.test_methods,
                "--output", str(d2_out),
            ]
            rc = _run(cmd, args.dry_run)
            if rc != 0:
                logger.error("D2 failed for %s (exit %d)", name, rc)
                rc_all |= rc

        if not args.skip_d3 and args.saliency_dir is not None:
            cmd = [
                py, "-m", "phase2.diagnostics.d3_phi_distribution",
                "--saliency-dir", str(args.saliency_dir),
                "--sequences", *args.sequences,
                "--output", str(d3_out),
            ]
            rc = _run(cmd, args.dry_run)
            if rc != 0:
                logger.error("D3 failed for %s (exit %d)", name, rc)
                rc_all |= rc

    if args.combined_output and not args.skip_d2 and len(d1_jsons) >= 2:
        args.combined_output.mkdir(parents=True, exist_ok=True)
        out_json = args.combined_output / "d2_combined.json"
        cmd = [
            py, "-m", "phase2.diagnostics.d2_statistical_significance",
            "--d1-json", *map(str, d1_jsons),
            "--reference-method", args.reference_method,
            "--test-methods", *args.test_methods,
            "--output", str(out_json),
        ]
        rc = _run(cmd, args.dry_run)
        if rc != 0:
            logger.error("Combined D2 failed (exit %d)", rc)
            rc_all |= rc

    if rc_all != 0:
        raise SystemExit(rc_all)
    logger.info("All Phase 1 diagnostics finished.")


if __name__ == "__main__":
    main()
