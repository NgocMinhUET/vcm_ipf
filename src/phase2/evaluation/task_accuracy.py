"""Task accuracy evaluation on decoded frames.

Runs YOLOv8 object detection on reconstructed (decoded) frames and
computes mAP metrics. This measures how well the encoded video preserves
machine vision task performance — the core VCM evaluation.

Scientific note (PROJECT_AUDIT §1.5)
------------------------------------
Pre-2026-05-08 versions of this module reported a quantity called
``mAP50`` that was actually ``precision × recall`` evaluated at a single
``confidence=0.25, IoU=0.5`` operating point. That number is **not** the
COCO Average Precision the literature uses (sorted PR-curve, 101-point
interpolation). Rankings between methods using the legacy quantity
were not statistically comparable to anything published.

This module now computes **real COCO-style AP**:

1. Detection runs at low confidence (default ``conf_low=0.001``) so the
   PR-curve has shape over the full recall range.
2. Per (class, IoU-threshold), predictions are sorted by confidence
   descending, greedily matched to GT at IoU ≥ t, accumulated into
   precision/recall arrays, then integrated under the precision-envelope
   on a 101-point recall grid (COCO convention).
3. Reported metrics:
    * ``mAP_50``        — AP at IoU=0.50, averaged over classes
    * ``mAP_75``        — AP at IoU=0.75
    * ``mAP_50_95``     — mean AP over IoU ∈ {0.50, 0.55, …, 0.95}

The legacy ``P × R`` quantity is retained as ``pxr_50`` purely so old
result tables remain interpretable.

Raw detections (boxes + scores + classes, both decoded and reference)
are saved to ``detections_decoded.json`` / ``detections_reference.json``
under each run directory. This allows offline re-scoring without re-
running the detector — the basis for `scripts/reevaluate_pilots.py`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("phase2.evaluation.task_accuracy")


# IoU sweep used for ``mAP_50_95`` (COCO convention).
IOU_THRESHOLDS = np.round(np.arange(0.50, 1.00, 0.05), 2).tolist()
COCO_RECALL_GRID = np.linspace(0.0, 1.0, 101)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    """Detections on a single frame from one detector pass."""
    frame_idx: int
    n_detections: int
    boxes: List[Tuple[float, float, float, float]]
    scores: List[float]
    classes: List[int]


@dataclass
class TaskMetrics:
    """Task accuracy metrics for one (sequence, method, QP) cell."""
    n_frames: int
    n_detections_total: int
    mean_detections_per_frame: float
    mean_confidence: float

    # Real COCO AP (the scientific metric)
    mAP50: float                 # AP@IoU=0.50
    mAP75: float                 # AP@IoU=0.75
    mAP50_95: float              # mean AP over IoU ∈ {0.50, ..., 0.95}

    # Legacy P×R quantity at conf=0.25, IoU=0.5 (kept for backward compat)
    pxr_50: float = 0.0

    # Raw counts at conf=0.25, IoU=0.5 (per-frame TP/FP/FN — needed for D2)
    per_frame_counts: List[List[int]] = field(default_factory=list)

    # Per-IoU AP breakdown (for inspection / diagnostics)
    per_iou_AP: Dict[str, float] = field(default_factory=dict)

    # Per-class API breakdown (for inspection)
    per_class_ap: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IoU + per-class AP primitives
# ---------------------------------------------------------------------------


def _box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised IoU between two sets of xyxy boxes; returns ``(len(a), len(b))``."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = a.astype(np.float32); b = b.astype(np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def _per_class_ap(scores: np.ndarray, matched: np.ndarray, n_gt: int) -> float:
    """COCO-style AP from a single (class, IoU) prediction list.

    ``scores`` and ``matched`` must already be sorted by score descending.
    """
    if n_gt == 0 or scores.size == 0:
        return 0.0
    tp_cum = np.cumsum(matched.astype(np.float64))
    fp_cum = np.cumsum((~matched).astype(np.float64))
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # Precision envelope (right-to-left max)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    ap = 0.0
    for r in COCO_RECALL_GRID:
        idx = np.searchsorted(recall, r, side="left")
        if idx < len(envelope):
            ap += envelope[idx]
    return float(ap / len(COCO_RECALL_GRID))


def _match_predictions_to_gt(
    pred_boxes: np.ndarray, pred_scores: np.ndarray,
    gt_boxes: np.ndarray, iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Match a single frame's predictions for one class to GT boxes.

    Returns ``(matched, scores_sorted, n_gt)`` where ``matched[i]`` is True
    if the i-th (score-sorted) prediction is a TP at this IoU threshold.
    """
    n_gt = int(gt_boxes.shape[0])
    if pred_boxes.shape[0] == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float32), n_gt
    if n_gt == 0:
        return (np.zeros((pred_boxes.shape[0],), dtype=bool),
                pred_scores.astype(np.float32), 0)
    iou = _box_iou_matrix(pred_boxes, gt_boxes)
    order = np.argsort(-pred_scores)
    pred_scores = pred_scores[order]
    iou = iou[order]
    matched = np.zeros((pred_boxes.shape[0],), dtype=bool)
    gt_used = np.zeros((n_gt,), dtype=bool)
    for i in range(iou.shape[0]):
        cand = iou[i].copy()
        cand[gt_used] = -1.0
        j = int(np.argmax(cand))
        if cand[j] >= iou_threshold:
            matched[i] = True
            gt_used[j] = True
    return matched, pred_scores.astype(np.float32), n_gt


# ---------------------------------------------------------------------------
# Public scoring functions (also used by reevaluate_pilots.py and D1)
# ---------------------------------------------------------------------------


def compute_coco_map_from_detections(
    pred_per_frame: List[DetectionResult],
    gt_per_frame: List[DetectionResult],
    classes_of_interest: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Real COCO mAP from two parallel lists of frame-level detections.

    Returns dict with ``mAP_50`` / ``mAP_75`` / ``mAP_50_95`` plus per-IoU
    ``AP_xx`` keys for inspection.
    """
    assert len(pred_per_frame) == len(gt_per_frame), (
        f"frame count mismatch: pred={len(pred_per_frame)} gt={len(gt_per_frame)}"
    )
    if classes_of_interest is None:
        seen = set()
        for fd in pred_per_frame:
            seen.update(fd.classes)
        for fd in gt_per_frame:
            seen.update(fd.classes)
        classes_of_interest = sorted(seen)

    out: Dict[str, float] = {}
    if not classes_of_interest:
        for t in IOU_THRESHOLDS:
            out[f"AP_{int(t*100):02d}"] = 0.0
        out["mAP_50"] = 0.0; out["mAP_75"] = 0.0; out["mAP_50_95"] = 0.0
        return out

    per_iou_aps: Dict[float, List[float]] = {t: [] for t in IOU_THRESHOLDS}
    for cls in classes_of_interest:
        for iou_t in IOU_THRESHOLDS:
            all_matched: List[np.ndarray] = []
            all_scores:  List[np.ndarray] = []
            n_gt_total = 0
            for pred, gt in zip(pred_per_frame, gt_per_frame):
                pb = np.asarray([b for b, c in zip(pred.boxes, pred.classes) if c == cls],
                                dtype=np.float32).reshape(-1, 4)
                ps = np.asarray([s for s, c in zip(pred.scores, pred.classes) if c == cls],
                                dtype=np.float32)
                gb = np.asarray([b for b, c in zip(gt.boxes, gt.classes) if c == cls],
                                dtype=np.float32).reshape(-1, 4)
                m, s, n_gt = _match_predictions_to_gt(pb, ps, gb, iou_t)
                all_matched.append(m); all_scores.append(s)
                n_gt_total += n_gt
            scores_cat = (np.concatenate(all_scores)
                          if all_scores else np.zeros((0,), dtype=np.float32))
            matched_cat = (np.concatenate(all_matched)
                           if all_matched else np.zeros((0,), dtype=bool))
            order = np.argsort(-scores_cat)
            scores_cat = scores_cat[order]
            matched_cat = matched_cat[order]
            ap = _per_class_ap(scores_cat, matched_cat, n_gt_total)
            per_iou_aps[iou_t].append(ap)

    for iou_t, aps in per_iou_aps.items():
        out[f"AP_{int(iou_t*100):02d}"] = float(np.mean(aps)) if aps else 0.0
    out["mAP_50"]    = out["AP_50"]
    out["mAP_75"]    = out["AP_75"]
    out["mAP_50_95"] = float(np.mean([out[f"AP_{int(t*100):02d}"] for t in IOU_THRESHOLDS]))
    return out


def compute_pxr_at_legacy(
    pred_per_frame: List[DetectionResult],
    gt_per_frame: List[DetectionResult],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
    classes_of_interest: Optional[List[int]] = None,
) -> Tuple[float, List[Tuple[int, int, int]]]:
    """Replicate the legacy ``precision × recall`` operating point.

    Returns ``(P × R, per_frame_counts)`` where each per-frame counts entry
    is ``(TP, FP, FN)`` aggregated over all classes of interest.
    """
    if classes_of_interest is None:
        seen = set()
        for fd in pred_per_frame:
            seen.update(fd.classes)
        for fd in gt_per_frame:
            seen.update(fd.classes)
        classes_of_interest = sorted(seen)

    per_frame_counts: List[Tuple[int, int, int]] = []
    tp_total = fp_total = fn_total = 0
    for pred, gt in zip(pred_per_frame, gt_per_frame):
        f_tp = f_fp = f_fn = 0
        for cls in classes_of_interest:
            pb = np.asarray([b for b, s, c in zip(pred.boxes, pred.scores, pred.classes)
                             if c == cls and s >= score_threshold],
                            dtype=np.float32).reshape(-1, 4)
            ps = np.asarray([s for s, c in zip(pred.scores, pred.classes)
                             if c == cls and s >= score_threshold],
                            dtype=np.float32)
            gb = np.asarray([b for b, c in zip(gt.boxes, gt.classes) if c == cls],
                            dtype=np.float32).reshape(-1, 4)
            m, _s, n_gt = _match_predictions_to_gt(pb, ps, gb, iou_threshold)
            tp = int(m.sum())
            fp = int((~m).sum()) if m.size else 0
            fn = int(n_gt - tp)
            f_tp += tp; f_fp += fp; f_fn += fn
        per_frame_counts.append((f_tp, f_fp, f_fn))
        tp_total += f_tp; fp_total += f_fp; fn_total += f_fn

    precision = tp_total / max(tp_total + fp_total, 1)
    recall    = tp_total / max(tp_total + fn_total, 1)
    return float(precision * recall), per_frame_counts


# ---------------------------------------------------------------------------
# YOLOv8 wrapper
# ---------------------------------------------------------------------------


class TaskEvaluator:
    """Detect → score (real AP + legacy P×R) → optional persist."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.25,
        device: str = "cuda:0",
        conf_low: float = 0.001,
        classes_of_interest: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        model_name, device : YOLO config.
        confidence : threshold for the legacy P×R operating point and for the
            user-visible mean-confidence statistic.
        conf_low : threshold used during real-AP detection. Must be ≤
            ``confidence`` and small enough that the PR curve covers the full
            recall range. Default 0.001 follows COCO convention.
        classes_of_interest : list of integer class IDs to evaluate. ``None``
            means "all classes seen in either prediction or GT". For VCM /
            MOT we typically pass ``[0]`` (person).
        """
        self.model_name = model_name
        self.confidence = float(confidence)
        self.device = device
        self.conf_low = float(conf_low)
        self.classes_of_interest = classes_of_interest
        self._model = None

    # ── detection ──────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
            logger.info("Loaded %s on %s", self.model_name, self.device)

    def detect_on_frames(
        self,
        frames_dir: str,
        n_frames: int = 0,
        conf: Optional[float] = None,
    ) -> List[DetectionResult]:
        """Detect on every image in a directory.

        Parameters
        ----------
        conf : if ``None``, uses ``self.conf_low`` (recommended for AP).
            Pass ``self.confidence`` to reproduce the legacy operating point.
        """
        self._load_model()
        fdir = Path(frames_dir).expanduser()

        patterns = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
        seen = set()
        files: List[Path] = []
        for pat in patterns:
            for p in fdir.glob(pat):
                key = p.name.lower()
                if key not in seen:
                    seen.add(key); files.append(p)
        files.sort(key=lambda p: p.name)
        if not files:
            logger.warning("detect_on_frames: no images in %s", fdir)
        if n_frames > 0:
            files = files[:n_frames]

        conf_use = self.conf_low if conf is None else float(conf)
        out: List[DetectionResult] = []
        for idx, fp in enumerate(files):
            results = self._model(
                str(fp), conf=conf_use, device=self.device, verbose=False,
            )
            boxes: List[Tuple[float, float, float, float]] = []
            scores: List[float] = []
            classes: List[int] = []
            if results and results[0].boxes is not None:
                bd = results[0].boxes
                for i in range(len(bd)):
                    boxes.append(tuple(bd.xyxy[i].cpu().numpy().astype(float).tolist()))
                    scores.append(float(bd.conf[i].cpu()))
                    classes.append(int(bd.cls[i].cpu()))
            out.append(DetectionResult(
                frame_idx=idx, n_detections=len(boxes),
                boxes=boxes, scores=scores, classes=classes,
            ))
        return out

    # ── scoring ────────────────────────────────────────────────────────────

    def compute_task_metrics(
        self,
        decoded_frames_dir: str,
        reference_frames_dir: Optional[str] = None,
        n_frames: int = 0,
        save_dir: Optional[str] = None,
    ) -> TaskMetrics:
        """Run detection, compute real AP + legacy P×R.

        If ``reference_frames_dir`` is None, AP cannot be computed (no GT)
        and only the detection statistics are returned.

        If ``save_dir`` is provided, raw detections are written to
        ``{save_dir}/detections_decoded.json`` and
        ``{save_dir}/detections_reference.json`` so we can re-score later
        without re-running YOLO.
        """
        decoded = self.detect_on_frames(decoded_frames_dir, n_frames, conf=self.conf_low)
        n_total = sum(d.n_detections for d in decoded)
        mean_dets = n_total / max(len(decoded), 1)
        mean_conf = (float(np.mean([s for d in decoded for s in d.scores]))
                     if any(d.scores for d in decoded) else 0.0)

        coco: Dict[str, float] = {}
        pxr_value = 0.0
        per_frame_counts: List[Tuple[int, int, int]] = []
        per_iou_AP: Dict[str, float] = {}

        if reference_frames_dir:
            reference = self.detect_on_frames(
                reference_frames_dir, n_frames, conf=self.conf_low,
            )
            n_use = min(len(decoded), len(reference))
            decoded_use = decoded[:n_use]; reference_use = reference[:n_use]
            coco = compute_coco_map_from_detections(
                decoded_use, reference_use, self.classes_of_interest,
            )
            pxr_value, per_frame_counts = compute_pxr_at_legacy(
                decoded_use, reference_use,
                iou_threshold=0.5, score_threshold=self.confidence,
                classes_of_interest=self.classes_of_interest,
            )
            per_iou_AP = {k: v for k, v in coco.items() if k.startswith("AP_")}

            if save_dir:
                save_pred_path = Path(save_dir).expanduser() / "detections_decoded.json"
                save_gt_path   = Path(save_dir).expanduser() / "detections_reference.json"
                save_detections(decoded_use,   str(save_pred_path))
                save_detections(reference_use, str(save_gt_path))
                logger.info("Saved raw detections to %s", save_dir)
        else:
            logger.warning("No reference frames; AP and P×R left at 0.0")

        return TaskMetrics(
            n_frames=len(decoded),
            n_detections_total=n_total,
            mean_detections_per_frame=mean_dets,
            mean_confidence=mean_conf,
            mAP50=float(coco.get("mAP_50", 0.0)),
            mAP75=float(coco.get("mAP_75", 0.0)),
            mAP50_95=float(coco.get("mAP_50_95", 0.0)),
            pxr_50=float(pxr_value),
            per_frame_counts=[list(c) for c in per_frame_counts],
            per_iou_AP=per_iou_AP,
        )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_detections(detections: List[DetectionResult], output_path: str) -> None:
    """Write a list of ``DetectionResult`` to JSON for offline re-scoring."""
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{
        "frame_idx":     d.frame_idx,
        "n_detections":  d.n_detections,
        "boxes":         [list(b) for b in d.boxes],
        "scores":        list(d.scores),
        "classes":       list(d.classes),
    } for d in detections]
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_detections(path: str) -> List[DetectionResult]:
    """Inverse of :func:`save_detections`."""
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    out: List[DetectionResult] = []
    for d in raw:
        out.append(DetectionResult(
            frame_idx=int(d["frame_idx"]),
            n_detections=int(d["n_detections"]),
            boxes=[tuple(b) for b in d.get("boxes", [])],
            scores=list(map(float, d.get("scores", []))),
            classes=list(map(int, d.get("classes", []))),
        ))
    return out
