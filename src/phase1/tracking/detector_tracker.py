"""Unified detector + tracker wrapper using Ultralytics.

Runs YOLOv8 detection with built-in ByteTrack/BoT-SORT tracking,
then converts results into ObjectState dataclasses.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from phase1.core.config import DetectorConfig, TrackerConfig, MassConfig
from phase1.core.constants import COCO_CLASS_NAMES, DEFAULT_CLASS_PRIORITY
from phase1.core.schemas import ObjectState
from phase1.utils.log import get_logger

logger = get_logger("tracking.detector_tracker")


class DetectorTracker:
    """Wraps Ultralytics YOLO model with built-in tracking.

    Initializes the model lazily on first call to allow config-only instantiation.
    """

    def __init__(
        self,
        det_cfg: DetectorConfig,
        trk_cfg: TrackerConfig,
        mass_cfg: MassConfig,
    ):
        self.det_cfg = det_cfg
        self.trk_cfg = trk_cfg
        self.mass_cfg = mass_cfg
        self._model = None
        self._track_ages: dict[int, int] = {}

    def _load_model(self) -> None:
        from ultralytics import YOLO

        logger.info("Loading model: %s on %s", self.det_cfg.model_name, self.det_cfg.device)
        self._model = YOLO(self.det_cfg.model_name)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> list[ObjectState]:
        """Run detection + tracking on a single BGR frame.

        Args:
            frame: BGR image (H, W, 3).
            frame_idx: 0-based frame index.

        Returns:
            List of ObjectState for all tracked objects in this frame.
        """
        if self._model is None:
            self._load_model()

        results = self._model.track(
            frame,
            persist=True,
            conf=self.det_cfg.confidence_threshold,
            iou=self.det_cfg.iou_threshold,
            imgsz=self.det_cfg.img_size,
            device=self.det_cfg.device,
            classes=self.det_cfg.classes,
            tracker=f"{self.trk_cfg.tracker_type}.yaml",
            verbose=False,
        )

        objects: list[ObjectState] = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].item())
                cls_id = int(boxes.cls[i].item())
                cls_name = COCO_CLASS_NAMES.get(cls_id, "unknown")

                w = float(x2 - x1)
                h = float(y2 - y1)
                cx = float(x1 + w / 2)
                cy = float(y1 + h / 2)

                # Track age management
                if track_id not in self._track_ages:
                    self._track_ages[track_id] = 0
                self._track_ages[track_id] += 1
                age_count = self._track_ages[track_id]

                # tau: ramp from 0 to 1 over warmup period
                tau = min(age_count / self.mass_cfg.age_warmup_frames, 1.0) \
                    if self.mass_cfg.use_track_age else 1.0

                # pi: class-dependent priority
                pi = self.mass_cfg.class_priorities.get(cls_name, DEFAULT_CLASS_PRIORITY) \
                    if self.mass_cfg.use_class_priority else 1.0

                obj = ObjectState(
                    track_id=track_id,
                    frame_idx=frame_idx,
                    x_center=cx,
                    y_center=cy,
                    width=w,
                    height=h,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    class_priority=pi,
                    track_age=tau,
                )
                objects.append(obj)

        logger.debug("Frame %d: %d objects tracked", frame_idx, len(objects))
        return objects

    def reset(self) -> None:
        """Reset tracker state for a new video."""
        self._track_ages.clear()
        if self._model is not None:
            self._model.predictor = None
        logger.info("Tracker state reset")
