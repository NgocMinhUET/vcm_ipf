"""Baseline QP map generation methods for controlled comparison.

Each method takes the same inputs (objects, frame dimensions, config)
and produces a CTU-level QP map — enabling direct, fair comparison.

Methods (from Experiment Matrix):
    M0: Uniform QP — VVC anchor, no ROI awareness
    M1: Binary ROI — hard mask with fixed delta QP
    M5: Gaussian heatmap — sigma proportional to object size
    M6: Exponential decay — importance = exp(-alpha * distance)
    M7: Distance transform — normalized inverse distance from ROI mask
    M8: Blurred ROI — binary ROI followed by Gaussian blur

All methods share:
    - Same qp_base, delta_roi, delta_bg, qp_min, qp_max
    - Same CTU grid construction
    - Same final clamp to [qp_min, qp_max]
    - No temporal smoothing (applied separately in comparison pipeline)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from phase1.core.config import QPMappingConfig, CTUConfig, BoundedDynamicsConfig
from phase1.core.schemas import ObjectState
from phase1.field.importance_field import build_ctu_grid
from phase1.utils.log import get_logger

logger = get_logger("baselines.qp_methods")


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
        """Compute final QP map from importance map.

        Uses the same asymmetric mapping as IPF for fair comparison.

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
    """M1: Hard binary ROI mask. CTU is ROI if any object center falls in it."""

    method_id = "M1"
    method_name = "Binary ROI"

    def compute_importance_map(self, objects, frame_h, frame_w):
        _, _, nr, nc = build_ctu_grid(frame_h, frame_w, self.ctu_cfg.ctu_size)
        imp = np.zeros((nr, nc), dtype=np.float64)
        ctu = self.ctu_cfg.ctu_size

        for obj in objects:
            # Mark all CTUs that overlap with the bounding box
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
    """M5: Gaussian kernel centered at each object. Sigma proportional to object size.

    sigma_x = sigma_factor * width
    sigma_y = sigma_factor * height

    Matched parameter budget: sigma_factor is the single tunable parameter.
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
            imp = np.maximum(imp, gauss)  # max superposition for Gaussian

        # Normalize to [0, 1]
        if imp.max() > 1e-8:
            imp = imp / imp.max()

        return imp


# =========================================================================
# M6: Exponential Distance Decay
# =========================================================================

class M6_ExponentialDecay(BaseQPMethod):
    """M6: importance = exp(-alpha * normalized_distance).

    Matched parameter budget: alpha is the single tunable parameter.
    """

    method_id = "M6"
    method_name = "Exponential Decay"

    def __init__(self, *args, alpha: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def compute_importance_map(self, objects, frame_h, frame_w):
        grid_x, grid_y, nr, nc = build_ctu_grid(frame_h, frame_w, self.ctu_cfg.ctu_size)
        imp = np.zeros((nr, nc), dtype=np.float64)

        for obj in objects:
            # Normalized distance (same as IPF for fairness)
            dx = (grid_x - obj.x_center) / (0.75 * obj.width + 1e-3)
            dy = (grid_y - obj.y_center) / (0.75 * obj.height + 1e-3)
            d = np.sqrt(dx**2 + dy**2)

            contrib = obj.confidence * np.exp(-self.alpha * d)
            imp = np.maximum(imp, contrib)

        if imp.max() > 1e-8:
            imp = imp / imp.max()

        return imp


# =========================================================================
# M7: Distance Transform Weighting
# =========================================================================

class M7_DistanceTransform(BaseQPMethod):
    """M7: Normalized inverse distance transform from binary ROI mask.

    Steps:
        1. Create pixel-level binary ROI mask
        2. Compute distance transform (distance to nearest ROI pixel)
        3. Invert and normalize: importance = 1 - dist/max_dist
        4. Downsample to CTU grid
    """

    method_id = "M7"
    method_name = "Distance Transform"

    def compute_importance_map(self, objects, frame_h, frame_w):
        ctu = self.ctu_cfg.ctu_size
        nr = int(np.ceil(frame_h / ctu))
        nc = int(np.ceil(frame_w / ctu))

        # Pixel-level binary mask
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        for obj in objects:
            x1 = max(0, int(obj.x_center - obj.width / 2))
            y1 = max(0, int(obj.y_center - obj.height / 2))
            x2 = min(frame_w, int(obj.x_center + obj.width / 2))
            y2 = min(frame_h, int(obj.y_center + obj.height / 2))
            mask[y1:y2, x1:x2] = 1

        if mask.max() == 0:
            return np.zeros((nr, nc), dtype=np.float64)

        # Distance transform from background to nearest foreground
        dist = distance_transform_edt(1 - mask)
        max_dist = dist.max() + 1e-8
        importance_pixel = 1.0 - (dist / max_dist)

        # Downsample to CTU grid (average pooling)
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
# M8: Blurred ROI Mask
# =========================================================================

class M8_BlurredROI(BaseQPMethod):
    """M8: Binary ROI mask followed by Gaussian blur and normalization.

    Matched parameter budget: blur_sigma is the single tunable parameter.
    """

    method_id = "M8"
    method_name = "Blurred ROI"

    def __init__(self, *args, blur_sigma: float = 64.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.blur_sigma = blur_sigma

    def compute_importance_map(self, objects, frame_h, frame_w):
        ctu = self.ctu_cfg.ctu_size
        nr = int(np.ceil(frame_h / ctu))
        nc = int(np.ceil(frame_w / ctu))

        # Pixel-level binary mask
        mask = np.zeros((frame_h, frame_w), dtype=np.float64)
        for obj in objects:
            x1 = max(0, int(obj.x_center - obj.width / 2))
            y1 = max(0, int(obj.y_center - obj.height / 2))
            x2 = min(frame_w, int(obj.x_center + obj.width / 2))
            y2 = min(frame_h, int(obj.y_center + obj.height / 2))
            mask[y1:y2, x1:x2] = obj.confidence

        # Gaussian blur
        blurred = gaussian_filter(mask, sigma=self.blur_sigma)

        if blurred.max() > 1e-8:
            blurred = blurred / blurred.max()

        # Downsample to CTU grid
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
