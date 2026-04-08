"""Importance Potential Field (IPF) computation.

Implements the core mathematical formulation:
  1. Importance mass:  m_{j,t} = pi_j * tau_{j,t} * c_{j,t} * sqrt(w * h)
  2. Normalized distance: d_{j,t}(x,y)
  3. Field kernel: phi_{j,t}(x,y) = m / (d^beta + eps_k)
  4. Superposition: Phi_t(x,y) = sum_j phi_{j,t}(x,y)

All computations are vectorized over the CTU grid for performance.
"""

from __future__ import annotations

import numpy as np

from phase1.core.config import FieldConfig, MassConfig, CTUConfig
from phase1.core.schemas import ObjectState
from phase1.utils.log import get_logger

logger = get_logger("field.importance_field")


def compute_importance_mass(obj: ObjectState) -> float:
    """Compute importance mass for a single tracked object.

    m_{j,t} = pi_j * tau_{j,t} * c_{j,t} * sqrt(w_{j,t} * h_{j,t})
    """
    return obj.class_priority * obj.track_age * obj.confidence * np.sqrt(obj.width * obj.height)


def build_ctu_grid(
    frame_h: int,
    frame_w: int,
    ctu_size: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build a grid of CTU center coordinates.

    Args:
        frame_h: Frame height in pixels.
        frame_w: Frame width in pixels.
        ctu_size: CTU size in pixels.

    Returns:
        (grid_x, grid_y, n_rows, n_cols) where grid_x and grid_y
        are 2D arrays of shape (n_rows, n_cols) containing CTU center
        pixel coordinates.
    """
    n_cols = int(np.ceil(frame_w / ctu_size))
    n_rows = int(np.ceil(frame_h / ctu_size))

    col_centers = np.arange(n_cols) * ctu_size + ctu_size / 2.0
    row_centers = np.arange(n_rows) * ctu_size + ctu_size / 2.0

    grid_x, grid_y = np.meshgrid(col_centers, row_centers)
    return grid_x, grid_y, n_rows, n_cols


def compute_single_object_field(
    obj: ObjectState,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    cfg: FieldConfig,
) -> np.ndarray:
    """Compute the field contribution of a single object over the CTU grid.

    phi_{j,t}(x,y) = m_{j,t} / (d_{j,t}(x,y)^beta + eps_k)

    where d_{j,t}(x,y) = sqrt(
        ((x - x_j) / (alpha_w * w_j + eps_d))^2 +
        ((y - y_j) / (alpha_h * h_j + eps_d))^2
    )

    Args:
        obj: Object state with position, size, and importance attributes.
        grid_x: 2D array of CTU center x-coordinates.
        grid_y: 2D array of CTU center y-coordinates.
        cfg: Field configuration parameters.

    Returns:
        2D array (n_rows, n_cols) of field values from this object.
    """
    mass = compute_importance_mass(obj)

    dx = (grid_x - obj.x_center) / (cfg.alpha_w * obj.width + cfg.eps_d)
    dy = (grid_y - obj.y_center) / (cfg.alpha_h * obj.height + cfg.eps_d)

    d_norm = np.sqrt(dx**2 + dy**2)

    phi = mass / (d_norm**cfg.beta + cfg.eps_k)
    return phi


def compute_superposition_field(
    objects: list[ObjectState],
    frame_h: int,
    frame_w: int,
    field_cfg: FieldConfig,
    ctu_cfg: CTUConfig,
) -> tuple[np.ndarray, int, int]:
    """Compute the total importance field via superposition of all objects.

    Phi_t(x,y) = sum_j phi_{j,t}(x,y)        (if mode = "sum")
    Phi_t(x,y) = max_j phi_{j,t}(x,y)        (if mode = "max")

    Args:
        objects: List of ObjectState for current frame.
        frame_h: Frame height in pixels.
        frame_w: Frame width in pixels.
        field_cfg: Field parameters (beta, eps, alpha, superposition mode).
        ctu_cfg: CTU size parameters.

    Returns:
        (field_map, n_rows, n_cols) — field_map is shape (n_rows, n_cols).
    """
    grid_x, grid_y, n_rows, n_cols = build_ctu_grid(frame_h, frame_w, ctu_cfg.ctu_size)

    if not objects:
        logger.debug("No objects — returning zero field (%d x %d)", n_rows, n_cols)
        return np.zeros((n_rows, n_cols), dtype=np.float64), n_rows, n_cols

    fields = np.stack(
        [compute_single_object_field(obj, grid_x, grid_y, field_cfg) for obj in objects],
        axis=0,
    )  # shape: (n_objects, n_rows, n_cols)

    if field_cfg.superposition == "max":
        total_field = np.max(fields, axis=0)
    else:
        total_field = np.sum(fields, axis=0)

    logger.debug(
        "Field: %d objects, range [%.4f, %.4f]",
        len(objects),
        float(np.min(total_field)),
        float(np.max(total_field)),
    )
    return total_field, n_rows, n_cols
