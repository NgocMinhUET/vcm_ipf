"""Data schemas for the IPF pipeline.

All structured data exchanged between modules uses these dataclasses
to enforce type safety and provide serialization helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Object-level state (one object, one frame)
# ---------------------------------------------------------------------------

@dataclass
class ObjectState:
    """State vector for a single tracked object at frame t.

    Matches the paper notation:
        o_{j,t} = (x, y, w, h, c, pi, tau)

    Attributes:
        track_id:    Unique track identifier assigned by the tracker.
        frame_idx:   0-based frame index.
        x_center:    Bounding-box center x (pixels).
        y_center:    Bounding-box center y (pixels).
        width:       Bounding-box width (pixels).
        height:      Bounding-box height (pixels).
        confidence:  Detector confidence score in [0, 1].
        class_id:    COCO class ID (int).
        class_name:  Human-readable class label.
        class_priority: pi_j — task-dependent priority weight (default 1.0).
        track_age:   tau_{j,t} — track age reliability factor in [0, 1].
    """

    track_id: int
    frame_idx: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    class_id: int = 0
    class_name: str = "object"
    class_priority: float = 1.0
    track_age: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ObjectState:
        return cls(**d)


# ---------------------------------------------------------------------------
# Frame-level results
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Aggregated result for a single frame.

    Stores references to computed arrays (field map, QP map) and
    the list of object states visible in that frame.
    """

    frame_idx: int
    objects: list[ObjectState] = field(default_factory=list)
    n_objects: int = 0
    raw_field: Optional[np.ndarray] = None       # H x W float
    normalized_field: Optional[np.ndarray] = None # CTU_rows x CTU_cols float [0,1]
    raw_qp_map: Optional[np.ndarray] = None      # CTU_rows x CTU_cols float
    final_qp_map: Optional[np.ndarray] = None     # CTU_rows x CTU_cols int
    processing_time_ms: float = 0.0

    def summary_dict(self) -> dict:
        """Lightweight summary (no numpy arrays)."""
        qp_stats = {}
        if self.final_qp_map is not None:
            qp_stats = {
                "qp_min": int(np.min(self.final_qp_map)),
                "qp_max": int(np.max(self.final_qp_map)),
                "qp_mean": float(np.mean(self.final_qp_map)),
                "qp_std": float(np.std(self.final_qp_map)),
            }
        return {
            "frame_idx": self.frame_idx,
            "n_objects": self.n_objects,
            "processing_time_ms": round(self.processing_time_ms, 2),
            **qp_stats,
        }


# ---------------------------------------------------------------------------
# Run-level metadata
# ---------------------------------------------------------------------------

@dataclass
class RunMetadata:
    """Top-level metadata for a complete pipeline run."""

    run_id: str = ""
    video_path: str = ""
    config_path: str = ""
    total_frames: int = 0
    processed_frames: int = 0
    frame_width: int = 0
    frame_height: int = 0
    fps: float = 0.0
    ctu_rows: int = 0
    ctu_cols: int = 0
    ctu_size: int = 128
    total_time_s: float = 0.0
    avg_frame_time_ms: float = 0.0
    avg_objects_per_frame: float = 0.0
    status: str = "pending"

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> RunMetadata:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)
