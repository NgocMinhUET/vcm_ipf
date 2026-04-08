"""Raw field-to-QP mapping with asymmetric ROI/BG curves.

Implements the mapping:
    if Phi_hat >= mu:
        Q_raw = Q_base - Delta_roi * ((Phi_hat - mu) / (1 - mu))^gamma_roi
    else:
        Q_raw = Q_base + Delta_bg  * ((mu - Phi_hat) / mu)^gamma_bg

This produces a continuous QP surface where:
    - ROI regions get lower QP (higher quality)
    - Background regions get higher QP (lower quality)
    - The transition is smooth and controlled by gamma parameters
"""

from __future__ import annotations

import numpy as np

from phase1.core.config import QPMappingConfig
from phase1.utils.log import get_logger

logger = get_logger("control.qp_mapper")


def map_field_to_qp(
    normalized_field: np.ndarray,
    cfg: QPMappingConfig,
) -> np.ndarray:
    """Map normalized importance field [0, 1] to raw QP values.

    Args:
        normalized_field: 2D array (n_rows, n_cols) in [0, 1].
        cfg: QP mapping configuration.

    Returns:
        2D array (n_rows, n_cols) of raw QP values (float).
    """
    qp_map = np.full_like(normalized_field, float(cfg.qp_base))

    # ROI region: Phi_hat >= mu → lower QP
    roi_mask = normalized_field >= cfg.mu
    if np.any(roi_mask):
        roi_strength = ((normalized_field[roi_mask] - cfg.mu) / (1.0 - cfg.mu + 1e-12)) ** cfg.gamma_roi
        qp_map[roi_mask] = cfg.qp_base - cfg.delta_roi * roi_strength

    # Background region: Phi_hat < mu → higher QP
    bg_mask = ~roi_mask
    if np.any(bg_mask):
        bg_strength = ((cfg.mu - normalized_field[bg_mask]) / (cfg.mu + 1e-12)) ** cfg.gamma_bg
        qp_map[bg_mask] = cfg.qp_base + cfg.delta_bg * bg_strength

    logger.debug(
        "QP map: range [%.1f, %.1f], ROI CTUs: %d/%d",
        float(np.min(qp_map)),
        float(np.max(qp_map)),
        int(np.sum(roi_mask)),
        roi_mask.size,
    )
    return qp_map
