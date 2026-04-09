"""Baseline QP map generation methods for controlled comparison.

Each method takes the same inputs (objects, frame dimensions, config)
and produces a normalized importance map in [0, 1] at CTU resolution.
The comparison pipeline applies the SAME QP mapping and bounded dynamics
to all methods, ensuring fair comparison.

Methods (from Experiment Matrix):
    M0: Uniform QP — VVC anchor, no ROI awareness
    M1: Binary ROI — hard mask, CTUs overlapping any bbox get importance=1
    M5: Gaussian heatmap — sigma proportional to object size
    M6: Exponential decay — importance = exp(-alpha * normalized_distance)
    M7: Distance transform — inverse distance from ROI with cutoff
    M8: Blurred ROI — binary ROI followed by Gaussian blur

All methods share:
    - Same QP mapping (applied externally in comparison pipeline)
    - Same bounded dynamics (applied externally)
    - Same CTU grid construction
    - Importance map output normalized to [0, 1]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from phase1.core.config import QPMappingConfig, CTUConfig, BoundedDynamicsConfig
from phase1.core.schemas import ObjectState
from phase1.field.importance_field import build_ctu_grid
from phase1.utils.log import get_logger

logger = get_logger("baselines.qp_methods")


def _avg_bbox_diagonal(objects: list[ObjectState]) -> float:
    """Average bounding box diagonal across all objects."""
    if not objects:
        return 100.0
    diags = [np.sqrt(o.width**2 + o.height**2) for o in objects]
    return float(np.mean(diags))


class BaseQPMethod(ABC):
    """Abstract base for all QP map generation methods."""

    method_id: str = "base"
    method_name: str = "Base Method"

    def __init__(
        self,
        qp_cfg: QPMappingConfig,
        ctu_cfg: CTUConfig,
        bd_cfg: BoundedDynamicsConfig,
    ):
        self.qp_cfg = qp_cfg
        self.ctu_cfg = ctu_cfg
        self.bd_cfg = bd_cfg

    @abstractmethod
    def compute_importance_map(
        self,
        objects: list[ObjectState],
        frame_h: int,
        frame_w: int,
    ) -> np.ndarray:
        """Compute normalized importance map in [0, 1] at CTU resolution.

        Returns:
            2D array (n_rows, n_cols) in [0, 1].
            1.0 = maximum importance (ROI center)
            0.0 = no importance (pure background)
        """
        ...

    def compute_qp_map(
        self,
        objects: list[ObjectState],
        frame_h: int,
        frame_w: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute final QP map from importance map (standalone use).

        For comparison pipeline, use compute_importance_map() directly
        with the shared QP mapping to ensure identical processing.

        Returns:
            (importance_map, qp_map) both at CTU resolution.
        """
        imp = self.compute_importance_map(objects, frame_h, frame_w)
        imp = np.clip(imp, 0.0, 1.0)

        cfg = self.qp_cfg
        qp_map = np.full_like(imp, float(cfg.qp_base))

        roi_mask = imp >= cfg.mu
        if np.any(roi_mask):
            strength = ((imp[roi_mask] - cfg.mu) / (1.0 - cfg.mu + 1e-12)) ** cfg.gamma_roi
            qp_map[roi_mask] = cfg.qp_base - cfg.delta_roi * strength

        bg_mask = ~roi_mask
        if np.any(bg_mask):
            strength = ((cfg.mu - imp[bg_mask]) / (cfg.mu + 1e-12)) ** cfg.gamma_bg
            qp_map[bg_mask] = cfg.qp_base + cfg.delta_bg * strength

        qp_map = np.clip(np.round(qp_map), self.bd_cfg.qp_min, self.bd_cfg.qp_max).astype(np.int32)
        return imp, qp_map


# =========================================================================
# M0: Uniform QP (VVC anchor)
# =========================================================================

class M0_UniformQP(BaseQPMethod):
    """M0: No ROI awareness. Uniform QP across all CTUs."""

    method_id = "M0"
    method_name = "Uniform QP (anchor)"

    def compute_importance_map(self, objects, frame_h, frame_w):
        _, _, nr, nc = build_ctu_grid(frame_h, frame_w, self.ctu_cfg.ctu_size)
        return np.zeros((nr, nc), dtype=np.float64)

    def compute_qp_map(self, objects, frame_h, frame_w):
        _, _, nr, nc = build_ctu_grid(frame_h, frame_w, self.ctu_cfg.ctu_size)
        imp = np.zeros((nr, nc), dtype=np.float64)
        qp = np.full((nr, nc), self.qp_cfg.qp_base, dtype=np.int32)
        return imp, qp


# =========================================================================
# M1: Binary ROI (hard mask with fixed delta QP)
# =========================================================================

class M1_BinaryROI(BaseQPMethod):
    """M1: Hard binary ROI mask.

    All CTUs overlapping any object bounding box get importance=1.0,
    producing a sharp ROI/background boundary.
    """

    method_id = "M1"
    method_name = "Binary ROI"

    def compute_importance_map(self, objects, frame_h, frame_w):
        _, _, nr, nc = build_ctu_grid(frame_h, frame_w, self.ctu_cfg.ctu_size)
        imp = np.zeros((nr, nc), dtype=np.float64)
        ctu = self.ctu_cfg.ctu_size

        for obj in objects:
            x1 = obj.x_center - obj.width / 2
            y1 = obj.y_center - obj.height / 2
            x2 = obj.x_center + obj.width / 2
            y2 = obj.y_center + obj.height / 2

            c_start = max(0, int(x1 // ctu))
            c_end = min(nc - 1, int(x2 // ctu))
            r_start = max(0, int(y1 // ctu))
            r_end = min(nr - 1, int(y2 // ctu))

            imp[r_start:r_end + 1, c_start:c_end + 1] = 1.0

        return imp


# =========================================================================
# M5: Gaussian Heatmap ROI
# =========================================================================

class M5_GaussianHeatmap(BaseQPMethod):
    """M5: Gaussian kernel centered at each object.

    sigma_x = sigma_factor * width, sigma_y = sigma_factor * height.
    Uses max-superposition across objects and normalizes to [0, 1].
    """

    method_id = "M5"
    method_name = "Gaussian Heatmap"

    def __init__(self, *args, sigma_factor: float = 0.75, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_factor = sigma_factor

    def compute_importance_map(self, objects, frame_h, frame_w):
        grid_x, grid_y, nr, nc = build_ctu_grid(frame_h, frame_w, self.ctu_cfg.ctu_size)
        imp = np.zeros((nr, nc), dtype=np.float64)

        for obj in objects:
            sigma_x = self.sigma_factor * obj.width + 1e-6
            sigma_y = self.sigma_factor * obj.height + 1e-6

            dx = (grid_x - obj.x_center) / sigma_x
            dy = (grid_y - obj.y_center) / sigma_y

            gauss = obj.confidence * np.exp(-0.5 * (dx**2 + dy**2))
            imp = np.maximum(imp, gauss)

        if imp.max() > 1e-8:
            imp = imp / imp.max()

        return imp


# =========================================================================
# M6: Exponential Distance Decay
# =========================================================================

class M6_ExponentialDecay(BaseQPMethod):
    """M6: importance = exp(-alpha * normalized_distance).

    Uses the same normalized distance as IPF (alpha_w * obj_width) for
    fair spatial comparison. alpha controls the decay steepness.
    Default alpha=0.8 gives ROI coverage comparable to other methods.
    """

    method_id = "M6"
    method_name = "Exponential Decay"

    def __init__(self, *args, alpha: float = 0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def compute_importance_map(self, objects, frame_h, frame_w):
        grid_x, grid_y, nr, nc = build_ctu_grid(frame_h, frame_w, self.ctu_cfg.ctu_size)
        imp = np.zeros((nr, nc), dtype=np.float64)

        for obj in objects:
            dx = (grid_x - obj.x_center) / (0.75 * obj.width + 1e-3)
            dy = (grid_y - obj.y_center) / (0.75 * obj.height + 1e-3)
            d = np.sqrt(dx**2 + dy**2)

            contrib = obj.confidence * np.exp(-self.alpha * d)
            imp = np.maximum(imp, contrib)

        if imp.max() > 1e-8:
            imp = imp / imp.max()

        return imp


# =========================================================================
# M7: Distance Transform Weighting (with cutoff)
# =========================================================================

class M7_DistanceTransform(BaseQPMethod):
    """M7: Inverse distance transform from binary ROI mask with cutoff.

    Steps:
        1. Create pixel-level binary ROI mask from bounding boxes
        2. Compute Euclidean distance transform (distance to nearest ROI pixel)
        3. Apply cutoff: importance = clip(1 - dist/cutoff, 0, 1)
           where cutoff = cutoff_factor * avg_bbox_diagonal
        4. Downsample to CTU grid via average pooling

    The cutoff prevents the distance transform from making every pixel
    "somewhat important", ensuring meaningful ROI/BG separation.
    """

    method_id = "M7"
    method_name = "Distance Transform"

    def __init__(self, *args, cutoff_factor: float = 2.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.cutoff_factor = cutoff_factor

    def compute_importance_map(self, objects, frame_h, frame_w):
        ctu = self.ctu_cfg.ctu_size
        nr = int(np.ceil(frame_h / ctu))
        nc = int(np.ceil(frame_w / ctu))

        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        for obj in objects:
            x1 = max(0, int(obj.x_center - obj.width / 2))
            y1 = max(0, int(obj.y_center - obj.height / 2))
            x2 = min(frame_w, int(obj.x_center + obj.width / 2))
            y2 = min(frame_h, int(obj.y_center + obj.height / 2))
            mask[y1:y2, x1:x2] = 1

        if mask.max() == 0:
            return np.zeros((nr, nc), dtype=np.float64)

        dist = distance_transform_edt(1 - mask)

        cutoff = self.cutoff_factor * _avg_bbox_diagonal(objects)
        importance_pixel = np.clip(1.0 - dist / (cutoff + 1e-8), 0.0, 1.0)

        imp = np.zeros((nr, nc), dtype=np.float64)
        for r in range(nr):
            for c in range(nc):
                y1 = r * ctu
                y2 = min((r + 1) * ctu, frame_h)
                x1 = c * ctu
                x2 = min((c + 1) * ctu, frame_w)
                imp[r, c] = importance_pixel[y1:y2, x1:x2].mean()

        return imp


# =========================================================================
# M8: Blurred ROI Mask (adaptive sigma)
# =========================================================================

class M8_BlurredROI(BaseQPMethod):
    """M8: Binary ROI mask followed by Gaussian blur and normalization.

    The blur sigma adapts to scene content:
        sigma = blur_factor * avg_bbox_diagonal
    This ensures the blur spread scales with object size, producing
    comparable ROI coverage across different video resolutions and scenes.
    """

    method_id = "M8"
    method_name = "Blurred ROI"

    def __init__(self, *args, blur_factor: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.blur_factor = blur_factor

    def compute_importance_map(self, objects, frame_h, frame_w):
        ctu = self.ctu_cfg.ctu_size
        nr = int(np.ceil(frame_h / ctu))
        nc = int(np.ceil(frame_w / ctu))

        mask = np.zeros((frame_h, frame_w), dtype=np.float64)
        for obj in objects:
            x1 = max(0, int(obj.x_center - obj.width / 2))
            y1 = max(0, int(obj.y_center - obj.height / 2))
            x2 = min(frame_w, int(obj.x_center + obj.width / 2))
            y2 = min(frame_h, int(obj.y_center + obj.height / 2))
            mask[y1:y2, x1:x2] = obj.confidence

        sigma = self.blur_factor * _avg_bbox_diagonal(objects)
        blurred = gaussian_filter(mask, sigma=sigma)

        if blurred.max() > 1e-8:
            blurred = blurred / blurred.max()

        imp = np.zeros((nr, nc), dtype=np.float64)
        for r in range(nr):
            for c in range(nc):
                y1 = r * ctu
                y2 = min((r + 1) * ctu, frame_h)
                x1 = c * ctu
                x2 = min((c + 1) * ctu, frame_w)
                imp[r, c] = blurred[y1:y2, x1:x2].mean()

        return imp


# =========================================================================
# Registry
# =========================================================================

METHOD_REGISTRY: dict[str, type[BaseQPMethod]] = {
    "M0": M0_UniformQP,
    "M1": M1_BinaryROI,
    "M5": M5_GaussianHeatmap,
    "M6": M6_ExponentialDecay,
    "M7": M7_DistanceTransform,
    "M8": M8_BlurredROI,
}


def create_method(
    method_id: str,
    qp_cfg: QPMappingConfig,
    ctu_cfg: CTUConfig,
    bd_cfg: BoundedDynamicsConfig,
    **kwargs,
) -> BaseQPMethod:
    """Factory function to create a QP method by ID."""
    if method_id not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method_id}. Available: {list(METHOD_REGISTRY.keys())}")
    cls = METHOD_REGISTRY[method_id]
    return cls(qp_cfg, ctu_cfg, bd_cfg, **kwargs)
