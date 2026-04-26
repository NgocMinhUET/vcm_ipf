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


def _compute_normalized_distance(
    obj: ObjectState,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    cfg: FieldConfig,
) -> np.ndarray:
    """Compute normalized distance from object center to all CTU positions.

    d_{j,t}(x,y) = sqrt(
        ((x - x_j) / (alpha_w * w_j + eps_d))^2 +
        ((y - y_j) / (alpha_h * h_j + eps_d))^2
    )

    Returns:
        2D array (n_rows, n_cols) of normalized distances.
    """
    dx = (grid_x - obj.x_center) / (cfg.alpha_w * obj.width + cfg.eps_d)
    dy = (grid_y - obj.y_center) / (cfg.alpha_h * obj.height + cfg.eps_d)
    return np.sqrt(dx**2 + dy**2)


def compute_single_object_field(
    obj: ObjectState,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    cfg: FieldConfig,
) -> np.ndarray:
    """Compute the IPF (Cauchy/Lorentzian) field for a single object.

    phi_{j,t}(x,y) = m_{j,t} / (d_{j,t}(x,y)^beta + eps_k)

    With eps_k=1.0 and beta=2.0, this is a Cauchy/Lorentzian kernel:
        - HWHM at d_norm = 1 (one object-width in normalized units)
        - Power-law tail provides long-range influence beyond Gaussian

    Args:
        obj: Object state with position, size, and importance attributes.
        grid_x: 2D array of CTU center x-coordinates.
        grid_y: 2D array of CTU center y-coordinates.
        cfg: Field configuration parameters.

    Returns:
        2D array (n_rows, n_cols) of field values from this object.
    """
    mass = compute_importance_mass(obj)
    d_norm = _compute_normalized_distance(obj, grid_x, grid_y, cfg)
    phi = mass / (d_norm**cfg.beta + cfg.eps_k)
    return phi


def compute_single_object_gaussian_field(
    obj: ObjectState,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    cfg: FieldConfig,
) -> np.ndarray:
    """Gaussian kernel alternative for ablation study (A6).

    Uses the same normalized distance and importance mass as IPF,
    but replaces the Cauchy/Lorentzian kernel with a Gaussian:
        phi_{j,t}(x,y) = m_{j,t} * exp(-d^2 / 2)

    This isolates the kernel shape's contribution: if IPF > A6,
    the power-law tail of the Cauchy kernel is beneficial.
    """
    mass = compute_importance_mass(obj)
    d_norm = _compute_normalized_distance(obj, grid_x, grid_y, cfg)
    phi = mass * np.exp(-d_norm**2 / 2.0)
    return phi


def _lp_aggregate(fields: np.ndarray, mode: str, p: float) -> np.ndarray:
    """Aggregate a stack of per-object fields along axis 0.

    Academic core of Phase 3 — see `11_PHASE3_RESEARCH_PROTOCOL.md`
    §2 Level 4.  The L_p-norm family

        Phi_Lp(r,c) = ( sum_j phi_j(r,c)^p )^(1/p)

    is continuous in p ∈ (0, ∞] and strictly interpolates between
    sum (p=1) and hard max (p=inf). This function dispatches on the
    `mode` string, keeping the legacy `sum` and `max` paths for
    backward compatibility and exposing the new `lp` path.

    Args:
        fields: float64 array of shape (n_objects, n_rows, n_cols),
            non-negative. Assumed all phi_j >= 0 by construction.
        mode: aggregation mode: ``"sum"`` | ``"max"`` | ``"lp"``.
        p: exponent used when ``mode == "lp"``. May be ``float("inf")``,
            in which case hard max is used (matches the p → ∞ limit).

    Returns:
        float64 array of shape (n_rows, n_cols).
    """
    if mode == "sum":
        return np.sum(fields, axis=0)
    if mode == "max":
        return np.max(fields, axis=0)
    if mode == "lp":
        if not np.isfinite(p):
            return np.max(fields, axis=0)
        if p <= 0:
            raise ValueError(f"p_norm must be positive, got {p}")
        # Numerically stable L_p:  (sum x_j^p)^(1/p) = m * (sum (x_j/m)^p)^(1/p)
        # where m = max_j x_j.  Avoids overflow for large p and keeps the
        # kernel magnitude well-scaled when m is small.
        m = np.max(fields, axis=0)
        # Avoid 0/0 — when all phi_j == 0 at a cell, aggregate is 0.
        safe_m = np.where(m > 0.0, m, 1.0)
        normed = fields / safe_m[None, :, :]
        powsum = np.sum(normed ** p, axis=0)
        agg = m * (powsum ** (1.0 / p))
        # Restore zeros where the max was zero.
        return np.where(m > 0.0, agg, 0.0)
    raise ValueError(
        f"Unknown superposition mode '{mode}'. Expected sum | max | lp."
    )


def compute_superposition_field(
    objects: list[ObjectState],
    frame_h: int,
    frame_w: int,
    field_cfg: FieldConfig,
    ctu_cfg: CTUConfig,
) -> tuple[np.ndarray, int, int]:
    """Compute the total importance field via superposition of all objects.

    Aggregator is chosen by ``field_cfg.superposition``:

        - ``"sum"``  → Phi(r,c) = sum_j phi_j(r,c)            (legacy)
        - ``"max"``  → Phi(r,c) = max_j phi_j(r,c)            (legacy, IPF v2)
        - ``"lp"``   → Phi(r,c) = (sum_j phi_j^p)^(1/p)        (Phase 3, p = field_cfg.p_norm)

    Args:
        objects: List of ObjectState for current frame.
        frame_h: Frame height in pixels.
        frame_w: Frame width in pixels.
        field_cfg: Field parameters (beta, eps, alpha, superposition mode, p_norm).
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

    total_field = _lp_aggregate(fields, field_cfg.superposition, field_cfg.p_norm)

    logger.debug(
        "Field: %d objects, mode=%s, p=%s, range [%.4f, %.4f]",
        len(objects),
        field_cfg.superposition,
        field_cfg.p_norm,
        float(np.min(total_field)),
        float(np.max(total_field)),
    )
    return total_field, n_rows, n_cols


def compute_gaussian_superposition_field(
    objects: list[ObjectState],
    frame_h: int,
    frame_w: int,
    field_cfg: FieldConfig,
    ctu_cfg: CTUConfig,
) -> tuple[np.ndarray, int, int]:
    """Gaussian kernel superposition field for ablation A6.

    Same structure as compute_superposition_field but uses Gaussian
    kernel instead of Cauchy/Lorentzian. Uses the same mass weighting,
    distance normalization, and superposition mode.
    """
    grid_x, grid_y, n_rows, n_cols = build_ctu_grid(frame_h, frame_w, ctu_cfg.ctu_size)

    if not objects:
        return np.zeros((n_rows, n_cols), dtype=np.float64), n_rows, n_cols

    fields = np.stack(
        [compute_single_object_gaussian_field(obj, grid_x, grid_y, field_cfg) for obj in objects],
        axis=0,
    )

    total_field = _lp_aggregate(fields, field_cfg.superposition, field_cfg.p_norm)
    return total_field, n_rows, n_cols
