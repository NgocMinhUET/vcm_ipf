"""Task accuracy evaluation on decoded frames.

Runs YOLOv8 object detection on reconstructed (decoded) frames and
computes mAP metrics. This measures how well the encoded video preserves
machine vision task performance — the core VCM evaluation.

The key insight: ROI-aware QP allocation (like IPF) should maintain
higher mAP at lower bitrates compared to uniform QP encoding.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("phase2.evaluation.task_accuracy")


@dataclass
class DetectionResult:
    """Detection results for a single frame."""
    frame_idx: int
    n_detections: int
    boxes: List[Tuple[float, float, float, float]]
    scores: List[float]
    classes: List[int]


@dataclass
class TaskMetrics:
    """Task accuracy metrics for a sequence."""
    n_frames: int
    n_detections_total: int
    mean_detections_per_frame: float
    mean_confidence: float
    mAP50: float
    mAP75: float
    mAP50_95: float
    per_class_ap: Dict[str, float] = field(default_factory=dict)


class TaskEvaluator:
    """Evaluates machine vision task accuracy on decoded frames."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.25,
        device: str = "cuda:0",
    ):
        self.model_name = model_name
        self.confidence = confidence
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy-load YOLO model."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
            logger.info("Loaded %s on %s", self.model_name, self.device)

    def detect_on_frames(
        self,
        frames_dir: str,
        n_frames: int = 0,
    ) -> List[DetectionResult]:
        """Run detection on a directory of PNG frames.

        Args:
            frames_dir: Directory with frames named NNNNNN.png.
            n_frames: Max frames to process (0 = all).

        Returns:
            List of DetectionResult, one per frame.
        """
        self._load_model()
        fdir = Path(frames_dir).expanduser()

        frame_files = sorted(fdir.glob("*.png"))
        if n_frames > 0:
            frame_files = frame_files[:n_frames]

        results_list = []
        for idx, fpath in enumerate(frame_files):
            results = self._model(
                str(fpath),
                conf=self.confidence,
                device=self.device,
                verbose=False,
            )

            boxes = []
            scores = []
            classes = []
            if results and results[0].boxes is not None:
                bdata = results[0].boxes
                for i in range(len(bdata)):
                    xyxy = bdata.xyxy[i].cpu().numpy()
                    boxes.append(tuple(xyxy.tolist()))
                    scores.append(float(bdata.conf[i].cpu()))
                    classes.append(int(bdata.cls[i].cpu()))

            results_list.append(DetectionResult(
                frame_idx=idx,
                n_detections=len(boxes),
                boxes=boxes,
                scores=scores,
                classes=classes,
            ))

        return results_list

    def compute_task_metrics(
        self,
        decoded_frames_dir: str,
        reference_frames_dir: Optional[str] = None,
        n_frames: int = 0,
    ) -> TaskMetrics:
        """Compute task accuracy metrics.

        For proper mAP computation with ground truth, we compare detections
        on decoded frames against detections on the original (uncompressed)
        reference frames. The reference detections serve as pseudo-GT.

        Args:
            decoded_frames_dir: Directory with decoded PNG frames.
            reference_frames_dir: Directory with original PNG frames (pseudo-GT).
            n_frames: Max frames to evaluate.

        Returns:
            TaskMetrics with mAP values.
        """
        decoded_dets = self.detect_on_frames(decoded_frames_dir, n_frames)

        n_total = sum(d.n_detections for d in decoded_dets)
        mean_dets = n_total / max(len(decoded_dets), 1)
        mean_conf = 0.0
        all_confs = [s for d in decoded_dets for s in d.scores]
        if all_confs:
            mean_conf = float(np.mean(all_confs))

        mAP50 = 0.0
        mAP75 = 0.0
        mAP50_95 = 0.0

        if reference_frames_dir:
            ref_dets = self.detect_on_frames(reference_frames_dir, n_frames)
            mAP50, mAP75, mAP50_95 = self._compute_map(decoded_dets, ref_dets)

        return TaskMetrics(
            n_frames=len(decoded_dets),
            n_detections_total=n_total,
            mean_detections_per_frame=mean_dets,
            mean_confidence=mean_conf,
            mAP50=mAP50,
            mAP75=mAP75,
            mAP50_95=mAP50_95,
        )

    def _compute_map(
        self,
        pred_dets: List[DetectionResult],
        gt_dets: List[DetectionResult],
    ) -> Tuple[float, float, float]:
        """Compute mAP at IoU 0.5, 0.75, and 0.5:0.95.

        Uses reference detections as pseudo ground truth.
        """
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        aps_per_thresh = []

        for iou_thresh in iou_thresholds:
            tp_total = 0
            fp_total = 0
            fn_total = 0

            for pred, gt in zip(pred_dets, gt_dets):
                if not gt.boxes:
                    fp_total += pred.n_detections
                    continue
                if not pred.boxes:
                    fn_total += gt.n_detections
                    continue

                gt_matched = [False] * len(gt.boxes)
                for pb in pred.boxes:
                    best_iou = 0.0
                    best_idx = -1
                    for j, gb in enumerate(gt.boxes):
                        iou = self._compute_iou(pb, gb)
                        if iou > best_iou and not gt_matched[j]:
                            best_iou = iou
                            best_idx = j
                    if best_iou >= iou_thresh and best_idx >= 0:
                        tp_total += 1
                        gt_matched[best_idx] = True
                    else:
                        fp_total += 1
                fn_total += sum(1 for m in gt_matched if not m)

            precision = tp_total / max(tp_total + fp_total, 1)
            recall = tp_total / max(tp_total + fn_total, 1)
            ap = precision * recall
            aps_per_thresh.append(ap)

        aps = np.array(aps_per_thresh)
        mAP50 = float(aps[0]) if len(aps) > 0 else 0.0
        mAP75 = float(aps[5]) if len(aps) > 5 else 0.0
        mAP50_95 = float(np.mean(aps))

        return mAP50, mAP75, mAP50_95

    @staticmethod
    def _compute_iou(
        box_a: Tuple[float, ...],
        box_b: Tuple[float, ...],
    ) -> float:
        """Compute IoU between two (x1, y1, x2, y2) boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        if union < 1e-6:
            return 0.0
        return inter / union


def save_detections(
    detections: List[DetectionResult],
    output_path: str,
) -> None:
    """Save detection results as JSON for reproducibility."""
    data = []
    for d in detections:
        data.append({
            "frame_idx": d.frame_idx,
            "n_detections": d.n_detections,
            "boxes": [list(b) for b in d.boxes],
            "scores": d.scores,
            "classes": d.classes,
        })
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
