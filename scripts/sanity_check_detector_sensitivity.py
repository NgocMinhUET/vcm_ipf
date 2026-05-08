"""Offline sanity check for analyze_detector_sensitivity.py.

Builds two synthetic d1_true_map.json files (one with shallow slope to
mimic yolov8n + QP[27,42], one with steep slope to mimic a stronger
detector + QP[27,42]) and verifies the analyser's verdict labels.

Run::

    PYTHONPATH=src python scripts/sanity_check_detector_sensitivity.py

Exit code 0 = all checks pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except AttributeError:
    pass

HERE = Path(__file__).resolve().parents[1]


def _make_d1(path: Path, *, shallow: bool) -> None:
    """Two sequences × M0+M4 × QP=27/32/37/42; mAP slope chosen by flag."""
    if shallow:
        # ~0.002 mAP per QP, like yolov8n on real GT (saturated regime)
        rates = {27: 0.30, 32: 0.295, 37: 0.290, 42: 0.275}
    else:
        # ~0.040 mAP per QP, well above the 2× noise-floor threshold
        # (a hypothetical strong detector that responds aggressively to
        # compression — gives "USEFUL" verdict).
        rates = {27: 0.85, 32: 0.65, 37: 0.45, 42: 0.25}

    cells = []
    for seq in ("S1", "S2"):
        for method in ("M0", "M4"):
            for qp, mAP in rates.items():
                m4_offset = -0.005 if method == "M4" else 0.0
                cells.append({
                    "sequence": seq, "method": method, "qp_base": qp,
                    "n_frames": 50,
                    "mAP_50": mAP * 1.5,
                    "mAP_75": mAP * 0.7,
                    "mAP_50_95": mAP + m4_offset,
                    "per_iou_AP": {},
                    "pxr_50_at_025": 0.1,
                    "per_frame_counts": [[5, 1, 1]] * 50,
                })
    path.write_text(json.dumps({"pilot_dir": "synth", "cells": cells}, indent=2),
                    encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        shallow = td_p / "d1_shallow.json"
        steep   = td_p / "d1_steep.json"
        _make_d1(shallow, shallow=True)
        _make_d1(steep,   shallow=False)
        out_md = td_p / "report.md"

        cmd = [
            sys.executable,
            str(HERE / "scripts" / "analyze_detector_sensitivity.py"),
            "--d1-jsons", str(shallow), str(steep),
            "--detector-labels", "yolov8n_synth", "yolov8m_synth",
            "--output", str(out_md),
            "--delta-qp", "4",
            "--roi-fraction", "0.30",
            "--noise-floor", "0.020",
        ]
        env_path_src = HERE / "src"
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(env_path_src)},
        )
        if proc.returncode != 0:
            print("[FAIL] analyser exited non-zero")
            print(proc.stdout); print(proc.stderr)
            sys.exit(1)

        md = out_md.read_text(encoding="utf-8")
        ok = True
        if "SATURATED" not in md:
            print("[FAIL] expected SATURATED verdict for shallow detector"); ok = False
        if "USEFUL" not in md:
            print("[FAIL] expected USEFUL verdict for steep detector"); ok = False
        if "Cross-detector slope comparison" not in md:
            print("[FAIL] expected cross-detector section"); ok = False
        if "Path-A decision" not in md:
            print("[FAIL] expected Path-A decision section"); ok = False

        if ok:
            print("[PASS] detector sensitivity analyser produces correct verdicts")
            print()
            print("=== Excerpt of generated report ===")
            for line in md.splitlines()[-20:]:
                print(line)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
