"""Diagnostic D1 — Compute *true* COCO-style mAP for existing pilot encodes.

Why this script exists
----------------------
Audit finding §1.5 (PROJECT_AUDIT.md) shows that
``phase2/src/phase2/evaluation/task_accuracy.py`` reports a quantity it calls
``mAP50`` that is actually ``precision × recall`` evaluated at a single
confidence threshold (``conf = 0.25``) and a single IoU threshold (0.5).
This is **not** the COCO/Pascal-VOC Average Precision used in the literature.

This script provides a clean, side-by-side implementation of the COCO mAP
definition: for each IoU threshold ``t ∈ {0.50, 0.55, …, 0.95}``,

1. Run YOLOv8 with a very low confidence threshold (``conf = 0.001``) on
   both the decoded frames and the reference (uncompressed) frames.
2. Treat the reference detections as pseudo ground-truth (same convention
   as the existing pipeline).
3. Sort all decoded detections by confidence (descending), greedily match
   each to an unmatched GT box at IoU ≥ t. Each match is a TP; otherwise
   FP. Unmatched GTs contribute to FN.
4. Walk the sorted list, accumulating cumulative TP/FP. Compute the
   precision–recall curve.
5. AP_t = area under the precision-envelope curve (COCO uses 101-point
   interpolation; we use the same).

Outputs per (sequence, method, QP) cell:

* ``mAP_50``, ``mAP_75``, ``mAP_50_95`` (true COCO definitions)
* Old ``pxr_50`` (precision × recall at conf=0.25, IoU=0.5) for direct
  comparison with the legacy metric.
* Per-frame counts at IoU=0.5, conf=0.25 (TP, FP, FN) — consumed by D2 for
  paired statistical tests.

Usage
-----
On the server with cached ``recon.yuv`` files::

    PYTHONPATH=src python -m phase2.diagnostics.d1_true_map \\
        --pilot-dir ~/Minh/ipf/phase2_outputs/pilot_v8b \\
        --config configs/pilot_v8b.yaml \\
        --output ~/Minh/ipf/phase2_outputs/pilot_v8b/diagnostics/d1_true_map.json \\
        --conf-low 0.001 \\
        --classes 0 \\
        --device cuda:0

Local sanity check on a tiny smoke directory works the same way.

Cost
----
Re-detection: about the same as the original task evaluation step (≈ 1–2 s
per frame on GPU, ≈ 20 s per CPU). For pilot_v8b (24 runs × 50 frames ×
2 detector passes = 2 400 frames) ≈ 4–8 min on GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("phase2.diagnostics.d1_true_map")


# ---------------------------------------------------------------------------
# COCO-style AP computation
# ---------------------------------------------------------------------------

IOU_THRESHOLDS = np.round(np.arange(0.50, 1.00, 0.05), 2).tolist()  # 10 thresholds
COCO_RECALL_GRID = np.linspace(0.0, 1.0, 101)


@dataclass
class FrameDetections:
    """All detections on one frame (for one detector run)."""
    boxes: np.ndarray   # (N, 4) xyxy float32
    scores: np.ndarray  # (N,) float32
    classes: np.ndarray  # (N,) int32

    @classmethod
    def empty(cls) -> "FrameDetections":
        return cls(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            classes=np.zeros((0,), dtype=np.int32),
        )


def _box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorised IoU between two sets of xyxy boxes.

    Returns a matrix of shape ``(len(boxes_a), len(boxes_b))``.
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    a = boxes_a.astype(np.float32)
    b = boxes_b.astype(np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter_w = np.clip(x2 - x1, 0.0, None)
    inter_h = np.clip(y2 - y1, 0.0, None)
    inter = inter_w * inter_h
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def _per_class_ap(
    pred_scores: np.ndarray,
    pred_matched: np.ndarray,  # bool array, True if TP at this IoU threshold
    n_gt: int,
) -> float:
    """Compute one AP value (COCO 101-point interpolation).

    ``pred_scores`` and ``pred_matched`` must be aligned and refer to ALL
    predictions for the (class, IoU-threshold) pair, sorted by score
    descending. ``n_gt`` is the total number of GT boxes for the class.
    """
    if n_gt == 0:
        return 0.0  # no GT → AP undefined; COCO conv: 0
    if pred_scores.size == 0:
        return 0.0
    # Already sorted by score (caller's responsibility)
    tp_cum = np.cumsum(pred_matched.astype(np.float64))
    fp_cum = np.cumsum((~pred_matched).astype(np.float64))
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # Make precision monotonically decreasing along recall (envelope)
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    # 101-point interpolation
    ap = 0.0
    for r in COCO_RECALL_GRID:
        idx = np.searchsorted(recall, r, side="left")
        if idx < len(precision_envelope):
            ap += precision_envelope[idx]
    ap /= len(COCO_RECALL_GRID)
    return float(ap)


def _match_predictions_to_gt(
    pred: FrameDetections,
    gt: FrameDetections,
    iou_threshold: float,
    cls: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Match one frame's predictions to GT for a single class & IoU threshold.

    Returns ``(pred_matched, pred_scores_for_class, n_gt_for_class)`` where
    ``pred_matched`` aligns with ``pred_scores_for_class`` (same length) and
    is True for TPs.
    """
    pred_mask = pred.classes == cls
    gt_mask = gt.classes == cls
    pred_boxes = pred.boxes[pred_mask]
    pred_scores = pred.scores[pred_mask]
    gt_boxes = gt.boxes[gt_mask]
    n_gt = int(gt_mask.sum())

    if pred_boxes.shape[0] == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float32), n_gt

    if n_gt == 0:
        return (
            np.zeros((pred_boxes.shape[0],), dtype=bool),
            pred_scores.astype(np.float32),
            0,
        )

    iou = _box_iou_matrix(pred_boxes, gt_boxes)
    order = np.argsort(-pred_scores)  # descending
    pred_scores = pred_scores[order]
    iou = iou[order]
    matched = np.zeros((pred_boxes.shape[0],), dtype=bool)
    gt_used = np.zeros((n_gt,), dtype=bool)
    for i in range(iou.shape[0]):
        candidates = iou[i].copy()
        candidates[gt_used] = -1.0
        j = int(np.argmax(candidates))
        if candidates[j] >= iou_threshold:
            matched[i] = True
            gt_used[j] = True
    return matched, pred_scores.astype(np.float32), n_gt


def compute_coco_map(
    pred_dets: List[FrameDetections],
    gt_dets: List[FrameDetections],
    classes_of_interest: Optional[List[int]] = None,
) -> Dict[str, float]:
    """COCO-style mAP from per-frame predictions and GT.

    Returns a dict with ``mAP_50``, ``mAP_75``, ``mAP_50_95`` plus per-IoU
    ``AP_xx`` keys for inspection.
    """
    assert len(pred_dets) == len(gt_dets), (
        f"frame count mismatch: pred={len(pred_dets)} gt={len(gt_dets)}"
    )

    classes = classes_of_interest
    if classes is None:
        all_cls = set()
        for fd in pred_dets:
            all_cls.update(fd.classes.tolist())
        for fd in gt_dets:
            all_cls.update(fd.classes.tolist())
        classes = sorted(all_cls)

    if not classes:
        return {f"AP_{int(t*100):02d}": 0.0 for t in IOU_THRESHOLDS} | {
            "mAP_50": 0.0, "mAP_75": 0.0, "mAP_50_95": 0.0,
        }

    # For each (class, IoU) compute AP
    per_iou_aps: Dict[float, List[float]] = {t: [] for t in IOU_THRESHOLDS}
    for cls in classes:
        for iou_t in IOU_THRESHOLDS:
            all_matched: List[np.ndarray] = []
            all_scores: List[np.ndarray] = []
            n_gt_total = 0
            for pred_fd, gt_fd in zip(pred_dets, gt_dets):
                m, s, n_gt = _match_predictions_to_gt(pred_fd, gt_fd, iou_t, cls)
                all_matched.append(m)
                all_scores.append(s)
                n_gt_total += n_gt
            if not all_scores:
                per_iou_aps[iou_t].append(0.0)
                continue
            scores_cat = np.concatenate(all_scores) if all_scores else np.zeros((0,))
            matched_cat = np.concatenate(all_matched) if all_matched else np.zeros((0,), dtype=bool)
            order = np.argsort(-scores_cat)
            scores_cat = scores_cat[order]
            matched_cat = matched_cat[order]
            ap = _per_class_ap(scores_cat, matched_cat, n_gt_total)
            per_iou_aps[iou_t].append(ap)

    out: Dict[str, float] = {}
    for iou_t, aps in per_iou_aps.items():
        key = f"AP_{int(iou_t * 100):02d}"
        out[key] = float(np.mean(aps)) if aps else 0.0
    out["mAP_50"] = out["AP_50"]
    out["mAP_75"] = out["AP_75"]
    out["mAP_50_95"] = float(np.mean(list(out[f"AP_{int(t*100):02d}"]
                                          for t in IOU_THRESHOLDS)))
    return out


def compute_pxr_at(
    pred_dets: List[FrameDetections],
    gt_dets: List[FrameDetections],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
    classes_of_interest: Optional[List[int]] = None,
) -> Tuple[float, List[Tuple[int, int, int]]]:
    """Replicate the legacy ``precision × recall`` metric for direct comparison.

    Also return per-frame ``(tp, fp, fn)`` counts (consumed by D2).
    """
    classes = classes_of_interest
    if classes is None:
        all_cls = set()
        for fd in pred_dets:
            all_cls.update(fd.classes.tolist())
        for fd in gt_dets:
            all_cls.update(fd.classes.tolist())
        classes = sorted(all_cls)

    per_frame_counts: List[Tuple[int, int, int]] = []
    tp_total = fp_total = fn_total = 0
    for pred_fd, gt_fd in zip(pred_dets, gt_dets):
        f_tp = f_fp = f_fn = 0
        # Apply legacy score threshold
        keep = pred_fd.scores >= score_threshold
        pred_kept = FrameDetections(
            boxes=pred_fd.boxes[keep],
            scores=pred_fd.scores[keep],
            classes=pred_fd.classes[keep],
        )
        for cls in classes:
            m, _s, n_gt = _match_predictions_to_gt(
                pred_kept, gt_fd, iou_threshold, cls
            )
            tp = int(m.sum())
            fp = int((~m).sum()) if m.size else 0
            fn = int(n_gt - tp)
            f_tp += tp; f_fp += fp; f_fn += fn
        per_frame_counts.append((f_tp, f_fp, f_fn))
        tp_total += f_tp; fp_total += f_fp; fn_total += f_fn

    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    return float(precision * recall), per_frame_counts


# ---------------------------------------------------------------------------
# Detector wrapper (low-conf YOLOv8)
# ---------------------------------------------------------------------------

def _list_image_files(d: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    return sorted([p for p in d.iterdir() if p.suffix in exts])


def detect_directory(
    frames_dir: Path,
    model_name: str,
    conf_low: float,
    device: str,
    n_frames: int = 0,
) -> List[FrameDetections]:
    """Detect on every image in ``frames_dir`` with a very low conf threshold.

    Returns a list aligned to the alphabetical order of files.
    """
    from ultralytics import YOLO
    model = YOLO(model_name)
    files = _list_image_files(frames_dir)
    if n_frames > 0:
        files = files[:n_frames]
    out: List[FrameDetections] = []
    for fp in files:
        results = model(str(fp), conf=conf_low, device=device, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            out.append(FrameDetections.empty())
            continue
        b = results[0].boxes
        out.append(FrameDetections(
            boxes=b.xyxy.detach().cpu().numpy().astype(np.float32),
            scores=b.conf.detach().cpu().numpy().astype(np.float32),
            classes=b.cls.detach().cpu().numpy().astype(np.int32),
        ))
    return out


# ---------------------------------------------------------------------------
# Top-level driver — re-evaluate one pilot directory
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    sequence: str
    method: str
    qp_base: int
    n_frames: int
    mAP_50: float
    mAP_75: float
    mAP_50_95: float
    per_iou_AP: Dict[str, float] = field(default_factory=dict)
    pxr_50_at_025: float = 0.0
    per_frame_counts: List[List[int]] = field(default_factory=list)  # [[tp, fp, fn], ...]


def evaluate_run(
    decoded_frames_dir: Path,
    reference_frames_dir: Path,
    model_name: str,
    conf_low: float,
    device: str,
    classes_of_interest: Optional[List[int]],
    n_frames: int,
) -> Tuple[Dict[str, float], float, List[Tuple[int, int, int]]]:
    """Detect → compute true mAP + legacy P×R + per-frame counts."""
    pred_dets = detect_directory(decoded_frames_dir, model_name, conf_low, device, n_frames)
    gt_dets = detect_directory(reference_frames_dir, model_name, conf_low, device, n_frames)
    n_use = min(len(pred_dets), len(gt_dets))
    pred_dets = pred_dets[:n_use]; gt_dets = gt_dets[:n_use]
    coco = compute_coco_map(pred_dets, gt_dets, classes_of_interest)
    pxr, per_frame = compute_pxr_at(
        pred_dets, gt_dets,
        iou_threshold=0.5, score_threshold=0.25,
        classes_of_interest=classes_of_interest,
    )
    return coco, pxr, per_frame


def reevaluate_pilot(
    pilot_dir: Path,
    config_path: Path,
    output_path: Path,
    model_name: str = "yolov8n.pt",
    conf_low: float = 0.001,
    classes_of_interest: Optional[List[int]] = None,
    device: str = "cuda:0",
    only_methods: Optional[List[str]] = None,
    only_sequences: Optional[List[str]] = None,
) -> List[CellResult]:
    """Iterate every run in a pilot directory and re-score with true mAP."""
    summary_path = pilot_dir / "experiment_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"experiment_summary.json missing at {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # Lazily resolve sequence frames-dir from config
    from phase2.core.config import load_phase2_config
    cfg = load_phase2_config(str(config_path))
    seq_to_frames: Dict[str, Path] = {}
    seq_to_n_frames: Dict[str, int] = {}
    for s in cfg.sequences:
        if s.frames_dir:
            seq_to_frames[s.name] = Path(s.frames_dir).expanduser().resolve()
            seq_to_n_frames[s.name] = int(s.n_frames)

    cells: List[CellResult] = []
    for entry in summary["results"]:
        seq_name = entry["sequence"]
        method = entry["method"]
        qp_base = entry["qp_base"]
        run_id = entry["run_id"]
        if only_methods and method not in only_methods:
            continue
        if only_sequences and seq_name not in only_sequences:
            continue

        run_dir = pilot_dir / run_id
        decoded_dir = run_dir / "decoded_frames"
        if not decoded_dir.exists() or not any(decoded_dir.iterdir()):
            logger.warning("[%s] decoded_frames missing — skip (rerun encode_pipeline first)", run_id)
            continue
        ref_dir = seq_to_frames.get(seq_name)
        if ref_dir is None:
            logger.warning("[%s] no frames_dir in config for %s — skip", run_id, seq_name)
            continue
        n_frames = seq_to_n_frames.get(seq_name, 0)

        logger.info("[%s] true-mAP re-eval (n_frames=%d)", run_id, n_frames)
        coco, pxr, per_frame = evaluate_run(
            decoded_dir, ref_dir, model_name, conf_low, device,
            classes_of_interest, n_frames,
        )
        cells.append(CellResult(
            sequence=seq_name, method=method, qp_base=qp_base,
            n_frames=len(per_frame),
            mAP_50=coco["mAP_50"], mAP_75=coco["mAP_75"],
            mAP_50_95=coco["mAP_50_95"],
            per_iou_AP={k: v for k, v in coco.items() if k.startswith("AP_")},
            pxr_50_at_025=pxr,
            per_frame_counts=[list(c) for c in per_frame],
        ))
        logger.info(
            "  mAP_50=%.4f  mAP_75=%.4f  mAP_50_95=%.4f  pxr_50@025=%.4f",
            coco["mAP_50"], coco["mAP_75"], coco["mAP_50_95"], pxr,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        {"pilot_dir": str(pilot_dir), "cells": [asdict(c) for c in cells]},
        indent=2,
    ), encoding="utf-8")
    logger.info("Wrote %d cells to %s", len(cells), output_path)
    return cells


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="D1 — true COCO mAP re-evaluation")
    parser.add_argument("--pilot-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf-low", type=float, default=0.001,
                        help="Low confidence threshold for full PR-curve sweep")
    parser.add_argument("--classes", type=int, nargs="*", default=[0],
                        help="COCO class IDs to evaluate (default: 0=person)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--sequences", nargs="*", default=None)
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

    cls_of_interest = args.classes if args.classes else None
    reevaluate_pilot(
        pilot_dir=args.pilot_dir,
        config_path=args.config,
        output_path=args.output,
        model_name=args.model,
        conf_low=args.conf_low,
        classes_of_interest=cls_of_interest,
        device=args.device,
        only_methods=args.methods,
        only_sequences=args.sequences,
    )


if __name__ == "__main__":
    main()
