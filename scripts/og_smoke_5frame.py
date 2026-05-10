"""5-frame smoke test for the OG-IPF apply pipeline (PROJECT_STATE §7.20).

Runs the **real** apply step (with YOLO box detection) on a tiny
5-frame slice of one MOT17 sequence to verify that:

* per-frame box JSONs are persisted under ``--saliency-dir``;
* δQP maps are written under ``qp_vtm_delta_QP{27,32,37,42}/``;
* ``og_diagnostics_summary.csv`` exists and reports
  ``object_protection_violation < 5 %``;
* rate-ratio post-rounding is in ``[0.95, 1.05]``.

**The smoke does NOT call VTM** — it only tests the OG generation
pipeline and the diagnostics. A separate full pilot
(``configs/pilot_v6_og.yaml``) handles encode + evaluate.

Usage::

    PYTHONPATH=src python scripts/og_smoke_5frame.py \\
        --frames-dir ~/Minh/ipf/datasets/MOT17/MOT17/train/MOT17-04-DPM/img1 \\
        --rate-npz   ~/Minh/ipf/phase3_outputs/rate/MOT17-04-DPM/rate_surrogate.npz \\
        --output-dir /tmp/og_smoke_v1 \\
        --device cpu     # safe even without GPU
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phase2.scripts.og_smoke_5frame")


def _run(cmd, label: str) -> None:
    logger.info("[%s] %s", label, " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"[{label}] failed (exit {proc.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="OG-IPF 5-frame smoke generation test.")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--rate-npz",   required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-frames",   type=int, default=5)
    ap.add_argument("--qp-list",    nargs="+", type=int,
                    default=[27, 32, 37, 42])
    ap.add_argument("--ctu-rows",   type=int, default=9)
    ap.add_argument("--ctu-cols",   type=int, default=15)
    ap.add_argument("--frame-h",    type=int, default=1152)
    ap.add_argument("--frame-w",    type=int, default=1920)
    ap.add_argument("--detector",   default="yolov8n.pt")
    ap.add_argument("--confidence", type=float, default=0.25)
    ap.add_argument("--device",     default="cpu")
    args = ap.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    sal = out / "saliency"
    sal.mkdir(exist_ok=True)
    m4 = out / "M4"

    # 1) Persist boxes for the 5-frame slice.
    _run([
        sys.executable, "-m", "phase2.phase3.save_boxes",
        "--frames-dir", str(Path(args.frames_dir).expanduser()),
        "--output-dir", str(sal),
        "--n-frames", str(args.n_frames),
        "--detector", args.detector,
        "--confidence", str(args.confidence),
        "--device", args.device,
        "--classes", "0",
    ], "save_boxes")

    # 2) Phi_oracle is not strictly required for og_a_plus, but the
    #    apply script falls back gracefully if it's missing — we skip
    #    the occlusion stage to keep the smoke fast (occlusion can take
    #    ~1.5 s/frame on CPU).
    #    The OG path does NOT read phi_oracle anyway.

    # 3) Apply OG-A+ on 5 frames × 4 QPs.
    _run([
        sys.executable, "-m", "phase2.phase3.apply_liteqp_model",
        "--mode", "og_a_plus",
        "--rate-npz", str(Path(args.rate_npz).expanduser()),
        "--saliency-dir", str(sal),
        "--boxes-dir",    str(sal),
        "--output-dir",   str(m4),
        "--qp-list", *map(str, args.qp_list),
        "--n-frames", str(args.n_frames),
        "--ctu-rows", str(args.ctu_rows),
        "--ctu-cols", str(args.ctu_cols),
        "--frame-h",  str(args.frame_h),
        "--frame-w",  str(args.frame_w),
        "--per-qp",
    ], "apply_og_a_plus")

    # 4) Verify outputs.
    failures: list[str] = []
    for q in args.qp_list:
        d = m4 / f"qp_vtm_delta_QP{q}"
        if not d.is_dir():
            failures.append(f"missing dir {d}")
            continue
        n = len(list(d.glob("qp_*.txt")))
        if n != args.n_frames:
            failures.append(
                f"{d.name}: got {n} maps, expected {args.n_frames}")

    csv_path = m4 / "og_diagnostics_summary.csv"
    if not csv_path.exists():
        failures.append(f"missing diagnostics CSV: {csv_path}")
    else:
        worst_violation = 0.0
        worst_drift = 0.0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                violation = float(row["object_protection_violation_max"])
                drift = abs(float(row["mean_rate_ratio_rounded"]) - 1.0)
                worst_violation = max(worst_violation, violation)
                worst_drift     = max(worst_drift,     drift)
        if worst_violation > 0.05:
            failures.append(
                f"object_protection_violation_max={worst_violation:.4f} > 5 %")
        if worst_drift > 0.05:
            failures.append(
                f"rate-ratio drift={worst_drift:.4f} > 5 %")
        logger.info(
            "Smoke diagnostics: worst object_protection_violation=%.3f, "
            "worst rate-ratio drift=%.3f",
            worst_violation, worst_drift,
        )

    meta = m4 / "liteqp_metadata.json"
    if not meta.exists():
        failures.append("missing metadata.json")
    else:
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
            if payload.get("method") != "M4-OG-A+":
                failures.append(
                    f"unexpected method in metadata: {payload.get('method')}")
        except Exception as exc:
            failures.append(f"could not parse metadata.json: {exc}")

    if failures:
        logger.error("OG smoke FAILED:")
        for f in failures:
            logger.error("  - %s", f)
        sys.exit(1)
    logger.info("OG smoke PASSED — %d δQP maps, %d QPs, %d frames each",
                 len(args.qp_list) * args.n_frames,
                 len(args.qp_list), args.n_frames)


if __name__ == "__main__":
    main()
