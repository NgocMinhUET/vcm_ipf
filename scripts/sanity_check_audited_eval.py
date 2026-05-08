"""Sanity check for the audited-evaluation stack (no GPU / no server).

Runs purely offline using synthetic detections to verify:

    1. ``compute_coco_map_from_detections`` returns sensible AP values
       on hand-crafted predictions vs GT.
    2. ``compute_pxr_at_legacy`` reproduces the legacy P×R operating point.
    3. ``save_detections`` / ``load_detections`` round-trip exactly.
    4. ``D1.evaluate_run`` correctly uses cached detection JSONs (no YOLO
       call) when both files exist.
    5. ``bd_rate_task_pchip`` returns a finite, signed value on a typical
       4-point RD curve.
    6. ``paired_bootstrap_bd_rate`` produces a valid CI on synthetic counts.

Run::

    PYTHONPATH=src python scripts/sanity_check_audited_eval.py

Exit code 0 = all checks pass.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List

# Ensure UTF-8 stdout on Windows so that Unicode logging from imported
# modules (e.g., ✓ / ✗ in some loggers) doesn't crash the script.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except AttributeError:
    pass

# Make the in-tree package importable when running from a fresh checkout.
HERE = Path(__file__).resolve().parents[1]
SRC = HERE / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from phase2.evaluation.task_accuracy import (  # noqa: E402
    DetectionResult,
    compute_coco_map_from_detections,
    compute_pxr_at_legacy,
    save_detections,
    load_detections,
)
from phase2.diagnostics.bd_rate_bootstrap import (  # noqa: E402
    CellBootstrapInput,
    bd_rate_task_pchip,
    paired_bootstrap_bd_rate,
)


def _make_detection(boxes, scores, classes, frame_idx=0):
    return DetectionResult(
        frame_idx=frame_idx,
        n_detections=len(boxes),
        boxes=[tuple(b) for b in boxes],
        scores=list(scores),
        classes=list(classes),
    )


def _check(name: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' — ' + detail) if detail else ''}")
    return cond


# ---------------------------------------------------------------------------
# 1) COCO mAP behaviour on hand-crafted data
# ---------------------------------------------------------------------------


def test_coco_perfect_match() -> bool:
    gt_box = [10, 10, 50, 50]
    gt = [_make_detection([gt_box], [1.0], [0])]
    pred = [_make_detection([gt_box], [0.9], [0])]
    out = compute_coco_map_from_detections(pred, gt, [0])
    return _check(
        "COCO mAP — perfect match should give AP=1.0",
        abs(out["mAP_50"] - 1.0) < 1e-6 and abs(out["mAP_50_95"] - 1.0) < 1e-6,
        f"mAP_50={out['mAP_50']:.4f}  mAP_50_95={out['mAP_50_95']:.4f}",
    )


def test_coco_no_overlap() -> bool:
    gt = [_make_detection([[10, 10, 50, 50]], [1.0], [0])]
    pred = [_make_detection([[200, 200, 240, 240]], [0.9], [0])]
    out = compute_coco_map_from_detections(pred, gt, [0])
    return _check(
        "COCO mAP — disjoint boxes should give AP≈0",
        out["mAP_50"] < 1e-6 and out["mAP_50_95"] < 1e-6,
        f"mAP_50={out['mAP_50']:.6f}",
    )


def test_coco_score_sorting_matters() -> bool:
    gt = [_make_detection([[10, 10, 50, 50], [60, 60, 100, 100]], [1.0, 1.0], [0, 0])]
    pred = [_make_detection(
        [[10, 10, 50, 50], [200, 200, 240, 240], [60, 60, 100, 100]],
        [0.9, 0.5, 0.2], [0, 0, 0],
    )]
    out = compute_coco_map_from_detections(pred, gt, [0])
    return _check(
        "COCO mAP — TP/FP interleaving still gives 2/3 of recall (>0.5)",
        out["mAP_50"] > 0.5,
        f"mAP_50={out['mAP_50']:.4f}",
    )


# ---------------------------------------------------------------------------
# 2) Legacy P×R reproduction
# ---------------------------------------------------------------------------


def test_pxr_reproduces_legacy() -> bool:
    gt = [_make_detection([[10, 10, 50, 50]], [1.0], [0])]
    pred = [_make_detection([[10, 10, 50, 50], [200, 200, 240, 240]],
                            [0.9, 0.6], [0, 0])]
    pxr, counts = compute_pxr_at_legacy(pred, gt, iou_threshold=0.5,
                                         score_threshold=0.25,
                                         classes_of_interest=[0])
    # 1 TP, 1 FP, 0 FN  → P = 0.5, R = 1.0, P×R = 0.5
    return _check(
        "Legacy P×R — P=0.5, R=1.0 → 0.5",
        abs(pxr - 0.5) < 1e-6 and counts == [(1, 1, 0)],
        f"pxr={pxr:.4f} counts={counts}",
    )


def test_pxr_threshold_filters() -> bool:
    gt = [_make_detection([[10, 10, 50, 50]], [1.0], [0])]
    pred = [_make_detection([[10, 10, 50, 50]], [0.10], [0])]   # below 0.25 conf
    pxr, counts = compute_pxr_at_legacy(pred, gt, iou_threshold=0.5,
                                         score_threshold=0.25,
                                         classes_of_interest=[0])
    return _check(
        "Legacy P×R — predictions below conf threshold are dropped",
        pxr == 0.0 and counts == [(0, 0, 1)],
        f"pxr={pxr:.4f} counts={counts}",
    )


# ---------------------------------------------------------------------------
# 3) save_detections / load_detections round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip() -> bool:
    dets = [
        _make_detection([[10, 10, 20, 20], [30, 30, 40, 40]],
                        [0.7, 0.4], [0, 1], frame_idx=0),
        _make_detection([], [], [], frame_idx=1),
        _make_detection([[5, 5, 7, 7]], [0.9], [0], frame_idx=2),
    ]
    with tempfile.TemporaryDirectory() as td:
        pth = Path(td) / "dets.json"
        save_detections(dets, str(pth))
        loaded = load_detections(str(pth))
    if len(loaded) != len(dets):
        return _check("save/load round-trip — same length", False,
                       f"saved={len(dets)} loaded={len(loaded)}")
    same = True
    for a, b in zip(dets, loaded):
        if a.frame_idx != b.frame_idx: same = False; break
        if a.boxes != b.boxes: same = False; break
        if list(a.scores) != list(b.scores): same = False; break
        if list(a.classes) != list(b.classes): same = False; break
    return _check("save/load round-trip — exact equality", same)


# ---------------------------------------------------------------------------
# 4) D1 uses cached detections instead of running YOLO
# ---------------------------------------------------------------------------


def test_d1_cached_detections_fastpath() -> bool:
    from phase2.diagnostics.d1_true_map import (
        evaluate_run, _load_cached_detections,
    )
    # Build cached detection JSONs that map to the new task_accuracy schema
    pred = [_make_detection([[10, 10, 50, 50]], [0.9], [0])]
    ref  = [_make_detection([[10, 10, 50, 50]], [0.95], [0])]
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        save_detections(pred, str(td_p / "pred.json"))
        save_detections(ref,  str(td_p / "ref.json"))
        # _load_cached_detections returns FrameDetections (different type than
        # DetectionResult). Just verify the load happens and shapes are right.
        loaded_pred = _load_cached_detections(td_p / "pred.json")
        loaded_ref  = _load_cached_detections(td_p / "ref.json")
        if loaded_pred is None or loaded_ref is None:
            return _check("D1 cached load — returns non-None", False)
        if loaded_pred[0].boxes.shape != (1, 4):
            return _check("D1 cached load — boxes shape", False,
                           f"got {loaded_pred[0].boxes.shape}")
        # evaluate_run should not crash even if frame dirs are empty,
        # because we passed cached_*_path that already exist.
        coco, pxr, per_frame = evaluate_run(
            decoded_frames_dir=td_p,    # unused
            reference_frames_dir=td_p,  # unused
            model_name="yolov8n.pt",
            conf_low=0.001,
            device="cpu",
            classes_of_interest=[0],
            n_frames=1,
            cached_pred_path=td_p / "pred.json",
            cached_ref_path=td_p / "ref.json",
        )
    return _check(
        "D1 fast-path — cached detections give AP=1.0",
        abs(coco["mAP_50"] - 1.0) < 1e-6 and pxr > 0.0,
        f"mAP_50={coco['mAP_50']:.4f} pxr={pxr:.4f}",
    )


# ---------------------------------------------------------------------------
# 5) bd_rate_task_pchip on a 4-point curve
# ---------------------------------------------------------------------------


def test_bd_rate_basic() -> bool:
    # M0: simple decreasing-rate, decreasing-mAP curve
    rate_b = [800, 400, 200, 100]
    qual_b = [0.85, 0.80, 0.70, 0.55]
    # M4: same mAP at 90% of the rate → BD-Rate-Task should be ≈ -10%
    rate_a = [r * 0.9 for r in rate_b]
    qual_a = qual_b
    bd = bd_rate_task_pchip(rate_a, qual_a, rate_b, qual_b)
    return _check(
        "bd_rate_task_pchip — uniform 10 % rate cut at iso-task ⇒ ≈ −10 %",
        not np.isnan(bd) and -12.0 < bd < -8.0,
        f"BD={bd:+.2f}%",
    )


def test_bd_rate_zero_when_equal() -> bool:
    rate = [800, 400, 200, 100]
    qual = [0.85, 0.80, 0.70, 0.55]
    bd = bd_rate_task_pchip(rate, qual, rate, qual)
    return _check(
        "bd_rate_task_pchip — equal curves ⇒ ≈ 0 %",
        not np.isnan(bd) and abs(bd) < 0.5,
        f"BD={bd:+.4f}%",
    )


def test_bd_rate_handles_duplicate_quality() -> bool:
    """Regression for the server crash on pilot_v4: PchipInterpolator raises
    ``x must be strictly increasing`` when bootstrap collapses two QPs to the
    same quality. Must return a finite value (or NaN cleanly), never crash.
    """
    # QP=37 and QP=42 share the same quality — a realistic bootstrap outcome.
    rate_b = [800, 400, 200, 100]
    qual_b = [0.85, 0.80, 0.55, 0.55]
    rate_a = [r * 0.95 for r in rate_b]
    qual_a = qual_b
    bd = bd_rate_task_pchip(rate_a, qual_a, rate_b, qual_b)
    return _check(
        "bd_rate_task_pchip — duplicate quality values do not crash",
        not np.isnan(bd) and abs(bd) < 50.0,
        f"BD={bd:+.2f}%",
    )


# ---------------------------------------------------------------------------
# 6) Paired bootstrap on synthetic counts
# ---------------------------------------------------------------------------


def _synthetic_cells(method: str, qp_offsets: List[int], rate_scale: float = 1.0) -> List[CellBootstrapInput]:
    """Make 4 RD points with 50 frames of (TP, FP, FN) counts."""
    cells: List[CellBootstrapInput] = []
    rng = np.random.default_rng(42 + (1 if method == "M4" else 0))
    base_rates = {27: 800, 32: 400, 37: 200, 42: 100}
    base_quals = {27: 0.85, 32: 0.80, 37: 0.70, 42: 0.55}
    for qp in [27, 32, 37, 42]:
        rate = base_rates[qp] * rate_scale
        # M4 has slightly better task quality at every QP
        qual = base_quals[qp] + (0.02 if method == "M4" else 0.0)
        # Build per-frame counts that approximately match `qual`
        per_frame = []
        for _ in range(50):
            tp = int(rng.binomial(20, qual))
            fp = int(rng.binomial(5, 0.5))
            fn = max(0, 20 - tp)
            per_frame.append((tp, fp, fn))
        cells.append(CellBootstrapInput(
            sequence="seq_synth", method=method, qp_base=qp,
            bitrate_kbps=rate,
            point_quality=qual,
            per_frame_counts=per_frame,
        ))
    return cells


def test_paired_bootstrap() -> bool:
    cells_by_method = {
        "M0": _synthetic_cells("M0", [0, 0, 0, 0], rate_scale=1.0),
        "M4": _synthetic_cells("M4", [0, 0, 0, 0], rate_scale=0.92),
    }
    bd = paired_bootstrap_bd_rate(
        cells_by_method, reference_method="M0", test_methods=["M4"],
        n_boot=200, alpha=0.05, seed=0,
    )
    r = bd["M4"]["seq_synth"]
    cond = (
        not np.isnan(r.point_estimate_true_map)
        and r.n_boot > 0
        and not np.isnan(r.ci_lo_pxr)
        and r.ci_lo_pxr <= r.boot_mean_pxr <= r.ci_hi_pxr
    )
    return _check(
        "paired_bootstrap_bd_rate — produces ordered CI",
        cond,
        f"point_true={r.point_estimate_true_map:+.2f}% "
        f"boot_mean_pxr={r.boot_mean_pxr:+.2f}% "
        f"CI=[{r.ci_lo_pxr:+.2f}, {r.ci_hi_pxr:+.2f}] n={r.n_boot}",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


TESTS = [
    test_coco_perfect_match,
    test_coco_no_overlap,
    test_coco_score_sorting_matters,
    test_pxr_reproduces_legacy,
    test_pxr_threshold_filters,
    test_save_load_roundtrip,
    test_d1_cached_detections_fastpath,
    test_bd_rate_basic,
    test_bd_rate_zero_when_equal,
    test_bd_rate_handles_duplicate_quality,
    test_paired_bootstrap,
]


def main() -> None:
    ok = 0
    failed: List[str] = []
    for t in TESTS:
        try:
            if t():
                ok += 1
            else:
                failed.append(t.__name__)
        except Exception:  # noqa: BLE001
            print(f"[FAIL] {t.__name__} — exception")
            traceback.print_exc()
            failed.append(t.__name__)
    print()
    print(f"PASSED: {ok} / {len(TESTS)}")
    if failed:
        print("FAILED tests:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
