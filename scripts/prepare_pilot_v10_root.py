"""Prepare the unified Phase-1-style root directory for pilot_v10_bench.

The pilot_v10_bench config compares 6 methods at the same regime:

    M0      uniform QP (no map needed)
    M_rect  hand-crafted ROI rectangle (NEW; script: build_rect_baseline.py)
    M1      Binary saliency (Phase 1; absolute -> relative conversion)
    M5      Anisotropic Gaussian (Phase 1; absolute -> relative conversion)
    M6      RD-driven Lagrangian (Phase 1; absolute -> relative conversion)
    M4      LiteQP-CNN-direct (Phase 3; already relative; symlink/copy)

The Phase 2 encoder pipeline reads QP maps from a single
``phase1_output_dir / <prefix><seq> / <method> / ...``. This script
populates that single root by:

    1. Calling ``build_rect_baseline.py`` for each sequence -> M_rect.
    2. Calling ``build_relative_dqp_from_absolute.py`` for M1/M5/M6.
    3. Copying (or symlinking) Phase 3 LiteQP-CNN maps for M4 from
       ``<phase3_root>/<phase3_prefix><seq>/M4/qp_vtm_delta_QP<base>/``
       into the unified root.

Result: a single tree under ``--unified-root`` that the standard
``encode_pipeline._find_qp_maps`` can consume.

Usage on the server::

    PYTHONPATH=src python scripts/prepare_pilot_v10_root.py \\
        --sequences MOT17-02-DPM MOT17-04-DPM MOT17-09-DPM \\
        --base-qps 27 32 37 42 \\
        --phase1-root ~/Minh/ipf/phase1_outputs \\
        --phase1-prefix multi_seq_ \\
        --phase3-root ~/Minh/ipf/phase3_outputs/learned \\
        --phase3-prefix liteqp_v5d_ \\
        --unified-root ~/Minh/ipf/pilot_v10_bench_root \\
        --frames-base ~/Minh/ipf/datasets/MOT17/MOT17/train \\
        --reference-boxes-dir ~/Minh/ipf/phase2_outputs/pilot_v8b/_reference_boxes \\
        --frames-cap 50 \\
        --width 1920 --height 1152
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger("phase2.scripts.prepare_pilot_v10_root")

LEGACY_METHODS = ["M1", "M5", "M6"]


def _run(cmd: List[str], dry: bool) -> int:
    logger.info("$ %s", " ".join(cmd))
    if dry:
        return 0
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def _link_or_copy(src: Path, dst: Path, mode: str = "copy") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == "symlink":
        try:
            dst.symlink_to(src, target_is_directory=src.is_dir())
            return
        except OSError as exc:
            logger.warning("symlink %s -> %s failed (%s); falling back to copy", dst, src, exc)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare unified pilot_v10 root")
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--base-qps", nargs="+", type=int, default=[27, 32, 37, 42])
    parser.add_argument("--phase1-root", required=True, type=Path,
                        help="Phase 1 outputs root (contains <prefix><seq>/<method>/qp_vtm/)")
    parser.add_argument("--phase1-prefix", default="multi_seq_")
    parser.add_argument("--phase3-root", required=True, type=Path,
                        help="Phase 3 LiteQP outputs root (contains <prefix><seq>/M4/qp_vtm_delta_QPxx/)")
    parser.add_argument("--phase3-prefix", default="liteqp_v5d_")
    parser.add_argument("--unified-root", required=True, type=Path,
                        help="Destination root that pilot_v10_bench.yaml points at")
    parser.add_argument("--unified-prefix", default="multi_seq_",
                        help="Prefix used inside unified root (matches yaml's phase1_run_prefix)")
    parser.add_argument("--frames-base", type=Path, default=None,
                        help="Optional: required only if M_rect needs detector run "
                             "(currently we read cached reference boxes, so this is unused)")
    parser.add_argument("--reference-boxes-dir", type=Path, required=True)
    parser.add_argument("--frames-cap", type=int, default=50)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--ctu-size", type=int, default=128)
    parser.add_argument("--rect-delta-roi", type=int, default=2)
    parser.add_argument("--rect-delta-bg", type=int, default=2)
    parser.add_argument("--rect-expansion", type=float, default=0.1)
    parser.add_argument("--m4-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--skip-mrect", action="store_true")
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--skip-m4", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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

    py = sys.executable
    rc_all = 0

    for seq in args.sequences:
        unified_seq = args.unified_root / f"{args.unified_prefix}{seq}"
        unified_seq.mkdir(parents=True, exist_ok=True)

        # ---- M_rect ------------------------------------------------------------
        if not args.skip_mrect:
            ref_json = args.reference_boxes_dir / f"{seq}.json"
            if not ref_json.exists():
                logger.error("[%s] reference-boxes JSON missing: %s — skip M_rect", seq, ref_json)
                rc_all |= 1
            else:
                out_dir = unified_seq / "M_rect" / "qp_vtm_delta"
                cmd = [
                    py, "scripts/build_rect_baseline.py",
                    "--reference-boxes", str(ref_json),
                    "--output-dir", str(out_dir),
                    "--width", str(args.width),
                    "--height", str(args.height),
                    "--ctu-size", str(args.ctu_size),
                    "--expansion", str(args.rect_expansion),
                    "--delta-roi", str(args.rect_delta_roi),
                    "--delta-bg", str(args.rect_delta_bg),
                    "--n-frames", str(args.frames_cap),
                ]
                rc_all |= _run(cmd, args.dry_run)

        # ---- Legacy M1, M5, M6 ------------------------------------------------
        if not args.skip_legacy:
            phase1_seq = args.phase1_root / f"{args.phase1_prefix}{seq}"
            for m in LEGACY_METHODS:
                in_dir = phase1_seq / m / "qp_vtm"
                if not in_dir.exists() or not any(in_dir.glob("qp_*.txt")):
                    logger.warning(
                        "[%s/%s] absolute QP dir missing or empty: %s — skip", seq, m, in_dir,
                    )
                    rc_all |= 1
                    continue
                out_dir = unified_seq / m / "qp_vtm_delta"
                cmd = [
                    py, "-m", "phase2.scripts.build_relative_dqp_from_absolute",
                    "--input-dir", str(in_dir),
                    "--output-dir", str(out_dir),
                ]
                # The script lives in scripts/, run as module path or direct.
                cmd = [
                    py, "scripts/build_relative_dqp_from_absolute.py",
                    "--input-dir", str(in_dir),
                    "--output-dir", str(out_dir),
                ]
                rc_all |= _run(cmd, args.dry_run)

        # ---- M4 (Phase 3 LiteQP-CNN-direct) -----------------------------------
        if not args.skip_m4:
            phase3_seq = args.phase3_root / f"{args.phase3_prefix}{seq}" / "M4"
            for qp in args.base_qps:
                src = phase3_seq / f"qp_vtm_delta_QP{qp}"
                if not src.exists():
                    logger.warning("[%s/M4/QP%d] source missing: %s", seq, qp, src)
                    rc_all |= 1
                    continue
                dst = unified_seq / "M4" / f"qp_vtm_delta_QP{qp}"
                if args.dry_run:
                    logger.info("$ link/copy %s -> %s", src, dst)
                else:
                    _link_or_copy(src, dst, mode=args.m4_mode)

    if rc_all != 0:
        logger.warning("Completed with errors (%d)", rc_all)
        sys.exit(rc_all)
    logger.info("Unified pilot_v10 root prepared at %s", args.unified_root)


if __name__ == "__main__":
    main()
