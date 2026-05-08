"""Local sanity check for the Phase 1 diagnostic battery.

Validates D1 (true mAP), D2 (paired stats), D3 (φ distribution) without
requiring YOLOv8, server data, or VTM. All checks operate on synthetic
fixtures and exit non-zero on any failure.

Run::

    PYTHONPATH=src python scripts/sanity_check_diagnostics.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except AttributeError:
    pass

# Make the in-tree package importable when running from repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from phase2.diagnostics import d1_true_map as d1
from phase2.diagnostics import d2_statistical_significance as d2
from phase2.diagnostics import d3_phi_distribution as d3


N_TESTS = 0
N_FAIL = 0


def _ok(msg: str) -> None:
    global N_TESTS
    N_TESTS += 1
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    global N_TESTS, N_FAIL
    N_TESTS += 1; N_FAIL += 1
    print(f"  [FAIL] {msg}")


def _assert(cond: bool, msg: str) -> None:
    if cond:
        _ok(msg)
    else:
        _fail(msg)


# ---------------------------------------------------------------------------
# D1 — COCO mAP correctness on synthetic data
# ---------------------------------------------------------------------------

def test_d1_perfect_match() -> None:
    print("D1.1 — Perfect prediction → mAP = 1.0")
    gt = [d1.FrameDetections(
        boxes=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
        scores=np.array([1.0, 1.0], dtype=np.float32),
        classes=np.array([0, 0], dtype=np.int32),
    )]
    pred = [d1.FrameDetections(
        boxes=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
        scores=np.array([0.9, 0.8], dtype=np.float32),
        classes=np.array([0, 0], dtype=np.int32),
    )]
    res = d1.compute_coco_map(pred, gt, classes_of_interest=[0])
    _assert(abs(res["mAP_50"] - 1.0) < 1e-6,
            f"mAP_50 = {res['mAP_50']:.4f} (expected 1.0)")
    _assert(abs(res["mAP_50_95"] - 1.0) < 1e-6,
            f"mAP_50_95 = {res['mAP_50_95']:.4f} (expected 1.0)")


def test_d1_no_overlap() -> None:
    print("D1.2 — Zero overlap → mAP = 0")
    gt = [d1.FrameDetections(
        boxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
        scores=np.array([1.0], dtype=np.float32),
        classes=np.array([0], dtype=np.int32),
    )]
    pred = [d1.FrameDetections(
        boxes=np.array([[100, 100, 110, 110]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([0], dtype=np.int32),
    )]
    res = d1.compute_coco_map(pred, gt, classes_of_interest=[0])
    _assert(res["mAP_50"] == 0.0,
            f"mAP_50 = {res['mAP_50']:.4f} (expected 0)")


def test_d1_partial_recall() -> None:
    print("D1.3 — Half of GTs detected at high score → AP_50 ≈ 0.5")
    gt = [d1.FrameDetections(
        boxes=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
        scores=np.array([1.0, 1.0], dtype=np.float32),
        classes=np.array([0, 0], dtype=np.int32),
    )]
    pred = [d1.FrameDetections(
        boxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([0], dtype=np.int32),
    )]
    res = d1.compute_coco_map(pred, gt, classes_of_interest=[0])
    # One TP, one missed GT → recall = 0.5, precision = 1
    # 101-pt interpolation: precision is 1 for r ≤ 0.5, 0 for r > 0.5
    _assert(0.45 < res["mAP_50"] < 0.56,
            f"mAP_50 = {res['mAP_50']:.4f} (expected ~0.5)")


def test_d1_pxr_matches_legacy() -> None:
    print("D1.4 — pxr metric matches the legacy P×R formula")
    gt = [d1.FrameDetections(
        boxes=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
        scores=np.array([1.0, 1.0], dtype=np.float32),
        classes=np.array([0, 0], dtype=np.int32),
    )]
    pred = [d1.FrameDetections(
        boxes=np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32),
        scores=np.array([0.9, 0.5], dtype=np.float32),
        classes=np.array([0, 0], dtype=np.int32),
    )]
    pxr, per_frame = d1.compute_pxr_at(pred, gt, 0.5, 0.25, [0])
    # Frame: TP=1 (matched bbox 1), FP=1 (bbox at 50,50 unmatched), FN=1 (gt 2 missed)
    # P = 1/2 = 0.5, R = 1/2 = 0.5, P×R = 0.25
    _assert(abs(pxr - 0.25) < 1e-6, f"pxr = {pxr:.4f} (expected 0.25)")
    _assert(per_frame[0] == (1, 1, 1),
            f"per-frame counts = {per_frame[0]} (expected (1,1,1))")


def test_d1_low_conf_predictions_ignored_in_pxr() -> None:
    print("D1.5 — pxr score threshold filters low-conf predictions")
    gt = [d1.FrameDetections(
        boxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
        scores=np.array([1.0], dtype=np.float32),
        classes=np.array([0], dtype=np.int32),
    )]
    pred = [d1.FrameDetections(
        boxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
        scores=np.array([0.10], dtype=np.float32),  # below 0.25 threshold
        classes=np.array([0], dtype=np.int32),
    )]
    pxr, per_frame = d1.compute_pxr_at(pred, gt, 0.5, 0.25, [0])
    _assert(pxr == 0.0, f"low-conf pred kept in pxr: {pxr}")
    _assert(per_frame[0] == (0, 0, 1),
            f"counts after thresholding = {per_frame[0]} (expected (0,0,1))")


def test_d1_iou_matrix() -> None:
    print("D1.6 — _box_iou_matrix vectorisation matches a manual calc")
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [50, 50, 60, 60]],
                 dtype=np.float32)
    iou = d1._box_iou_matrix(a, b)
    # IoU(a, b[1]) = 25 / (100 + 100 - 25) = 25/175 ≈ 0.1429
    _assert(abs(iou[0, 0] - 1.0) < 1e-5, f"self-IoU != 1: {iou[0,0]}")
    _assert(abs(iou[0, 1] - 25.0 / 175.0) < 1e-5,
            f"manual IoU mismatch: {iou[0,1]:.4f}")
    _assert(iou[0, 2] == 0.0, f"non-overlap IoU != 0: {iou[0,2]}")


# ---------------------------------------------------------------------------
# D2 — Statistical tests on synthetic per-frame counts
# ---------------------------------------------------------------------------

def _make_d1_payload(rng: np.random.Generator, n_frames: int = 50) -> dict:
    """Synthesise a paired 2-method × 1-sequence × 1-QP D1 JSON.

    The pairing matters for D2: per-frame counts for M0 and the M4 variants
    must share a common per-frame "scene difficulty" so that paired tests
    can detect a *systematic* offset.
    """
    base_tp = rng.poisson(8, size=n_frames)
    base_fp = rng.poisson(2, size=n_frames)
    base_fn = rng.poisson(2, size=n_frames)
    cells = []
    for method, tp_offset in [("M0", 0), ("M4_real_win", 2), ("M4_noise", 0)]:
        per_frame = []
        for i in range(n_frames):
            jitter_tp = rng.poisson(0.3) if method == "M4_noise" else 0
            jitter_fp = rng.poisson(0.3) if method == "M4_noise" else 0
            tp = int(max(0, base_tp[i] + tp_offset + jitter_tp))
            fp = int(max(0, base_fp[i] + jitter_fp))
            # M4_real_win recovers some misses that M0 had
            fn = int(max(0, base_fn[i] - tp_offset))
            per_frame.append([tp, fp, fn])
        cells.append({
            "sequence": "TEST",
            "method": method,
            "qp_base": 32,
            "n_frames": n_frames,
            "mAP_50": 0.5,
            "mAP_75": 0.4,
            "mAP_50_95": 0.3,
            "per_iou_AP": {},
            "pxr_50_at_025": 0.5,
            "per_frame_counts": per_frame,
        })
    return {"cells": cells}


def test_d2_real_vs_noise() -> None:
    print("D2.1 — Real win is flagged, noise is not (synthetic Poisson data)")
    rng = np.random.default_rng(42)
    payload = _make_d1_payload(rng, n_frames=50)
    cmps = d2.compare_cells(payload["cells"], "M0", ["M4_real_win", "M4_noise"])
    by_method = {c.method: c for c in cmps}
    real = by_method["M4_real_win"]; noise = by_method["M4_noise"]
    _assert(real.is_real_effect,
            f"M4_real_win should be REAL (p={real.wilcoxon_p:.3f}, d={real.cohens_d:.2f})")
    _assert(not noise.is_real_effect,
            f"M4_noise should be noise (p={noise.wilcoxon_p:.3f}, d={noise.cohens_d:.2f})")


def test_d2_bootstrap_ci_contains_zero_for_noise() -> None:
    print("D2.2 — Bootstrap CI for noise comparison contains 0")
    rng = np.random.default_rng(7)
    payload = _make_d1_payload(rng, n_frames=50)
    cmps = d2.compare_cells(payload["cells"], "M0", ["M4_noise"])
    ci = cmps[0].bootstrap_ci
    _assert(ci["ci_lo"] < 0 < ci["ci_hi"] or abs(ci["point"]) < 0.01,
            f"CI for noise = [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}] "
            f"(point = {ci['point']:.4f})")


def test_d2_per_frame_f1() -> None:
    print("D2.3 — F1 helper gives 0 / 0.5 / 1 in the obvious cases")
    f = d2._per_frame_f1([[10, 0, 0], [5, 5, 5], [0, 5, 5]])
    expected = [1.0, 0.5, 0.0]
    for v, e in zip(f.tolist(), expected):
        _assert(abs(v - e) < 1e-6, f"F1 case: got {v}, expected {e}")


def test_d2_markdown_renders() -> None:
    print("D2.4 — render_markdown produces a non-empty table with header")
    rng = np.random.default_rng(0)
    payload = _make_d1_payload(rng, n_frames=50)
    cmps = d2.compare_cells(payload["cells"], "M0", ["M4_real_win", "M4_noise"])
    md = d2.render_markdown(cmps, "M0")
    has_header = "| Sequence |" in md and "Tally" in md
    _assert(has_header, "markdown rendering missing header/tally")


# ---------------------------------------------------------------------------
# D3 — φ distribution metrics on synthetic grids
# ---------------------------------------------------------------------------

def test_d3_uniform_grid_high_entropy() -> None:
    print("D3.1 — Uniform φ grid → entropy_ratio ≈ 1, headroom ≈ 0")
    grids = np.ones((10, 9, 15), dtype=np.float64)
    s = d3._compute_one("UNIFORM", grids)
    _assert(s.entropy_ratio > 0.99,
            f"uniform entropy_ratio = {s.entropy_ratio:.4f}")
    _assert(s.headroom_score < 0.20,
            f"uniform headroom = {s.headroom_score:.4f}")


def test_d3_delta_grid_low_entropy() -> None:
    print("D3.2 — Single-spike φ grid → entropy_ratio ≈ 0, headroom large")
    grids = np.zeros((10, 9, 15), dtype=np.float64)
    grids[:, 4, 7] = 100.0
    s = d3._compute_one("SPIKE", grids)
    _assert(s.entropy_ratio < 0.05,
            f"spike entropy_ratio = {s.entropy_ratio:.4f}")
    _assert(s.headroom_score > 0.65,
            f"spike headroom = {s.headroom_score:.4f}")


def test_d3_top_k_mass() -> None:
    print("D3.3 — top-K mass is between 0 and 1 and monotone in K")
    g = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    m1 = d3._top_k_mass(g, 1)
    m3 = d3._top_k_mass(g, 3)
    m_all = d3._top_k_mass(g, 9)
    _assert(0 < m1 < m3 < 1.0 and abs(m_all - 1.0) < 1e-9,
            f"top_k mass not monotone: {m1:.3f}, {m3:.3f}, {m_all:.3f}")


def test_d3_spearman_random_low() -> None:
    print("D3.4 — Spearman of independent random vectors is near zero")
    rng = np.random.default_rng(1)
    a = rng.normal(size=200); b = rng.normal(size=200)
    rho = d3._spearman(a, b)
    _assert(abs(rho) < 0.20, f"|rho| too high for independent: {rho:.3f}")


def test_d3_writes_files(tmp_root: Path) -> None:
    print("D3.5 — End-to-end run produces JSON + markdown")
    sal_root = tmp_root / "saliency"
    seq_dir = sal_root / "TEST"
    seq_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    for fi in range(8):
        g = rng.gamma(1.0, size=(9, 15))
        np.save(seq_dir / f"phi_oracle_{fi:06d}.npy", g)
    out = tmp_root / "d3.json"

    grids = d3._load_per_frame(seq_dir)
    s = d3._compute_one("TEST", grids)
    cross = d3._cross_sequence_correlation({"TEST": grids.mean(axis=0)})
    md = d3.render_markdown([s], cross)
    out.write_text(json.dumps({"per_sequence": [s.__dict__],
                               "cross_sequence_spearman": cross}, indent=2),
                   encoding="utf-8")
    out.with_suffix(".md").write_text(md, encoding="utf-8")

    _assert(out.exists() and out.with_suffix(".md").exists(),
            f"D3 output files written ({out.name}, {out.with_suffix('.md').name})")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Phase 1 diagnostics — local sanity check")
    print("=" * 70)

    print("\n--- D1 ---")
    test_d1_perfect_match()
    test_d1_no_overlap()
    test_d1_partial_recall()
    test_d1_pxr_matches_legacy()
    test_d1_low_conf_predictions_ignored_in_pxr()
    test_d1_iou_matrix()

    print("\n--- D2 ---")
    test_d2_real_vs_noise()
    test_d2_bootstrap_ci_contains_zero_for_noise()
    test_d2_per_frame_f1()
    test_d2_markdown_renders()

    print("\n--- D3 ---")
    test_d3_uniform_grid_high_entropy()
    test_d3_delta_grid_low_entropy()
    test_d3_top_k_mass()
    test_d3_spearman_random_low()
    with tempfile.TemporaryDirectory() as td:
        test_d3_writes_files(Path(td))

    print("\n" + "=" * 70)
    if N_FAIL == 0:
        print(f"All {N_TESTS} sanity checks passed.")
        sys.exit(0)
    else:
        print(f"{N_FAIL}/{N_TESTS} sanity checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
