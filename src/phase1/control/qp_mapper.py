"""Field-to-QP and field-to-dQP mappings (Level 2 asymmetric power law).

Reference: `IPF_Development_Plan/11_PHASE3_RESEARCH_PROTOCOL.md` §2, Level 2.

Both functions share the same asymmetric power-law mapping of the
normalized importance field Phi_hat ∈ [0, 1]:

    if Phi_hat >= mu:  boost ROI   => negative offset (better quality)
    if Phi_hat <  mu:  penalize BG => positive offset (lower quality)

The magnitude of each side is controlled by (Delta_roi, gamma_roi) and
(Delta_bg, gamma_bg) respectively. The two outputs differ only in
whether Q_base is added (absolute QP) or not (delta QP).

Phase 3 REQUIRES the delta form: an absolute map is only valid at the
single Q_base used for calibration (see PROJECT_STATE.md §5.4 "CRITICAL:
QP map calibration mismatch"). The delta form is Q_base-agnostic and
is composed with Q_base at encoder invocation time in Phase 2.
"""

from __future__ import annotations

import numpy as np

from phase1.core.config import QPMappingConfig
from phase1.utils.log import get_logger

logger = get_logger("control.qp_mapper")


def _asymmetric_delta(
    normalized_field: np.ndarray,
    cfg: QPMappingConfig,
) -> np.ndarray:
    """Shared core: compute the signed per-CTU QP OFFSET (dQP).

    Returns a float array of the same shape as `normalized_field`.
    Negative values correspond to ROI boost; positive values correspond
    to background penalty. Magnitudes are clipped implicitly by the
    configured delta_roi / delta_bg.
    """
    delta = np.zeros_like(normalized_field, dtype=np.float64)

    roi_mask = normalized_field >= cfg.mu
    if np.any(roi_mask):
        roi_strength = (
            (normalized_field[roi_mask] - cfg.mu) / (1.0 - cfg.mu + 1e-12)
        ) ** cfg.gamma_roi
        delta[roi_mask] = -cfg.delta_roi * roi_strength

    bg_mask = ~roi_mask
    if np.any(bg_mask):
        bg_strength = (
            (cfg.mu - normalized_field[bg_mask]) / (cfg.mu + 1e-12)
        ) ** cfg.gamma_bg
        delta[bg_mask] = +cfg.delta_bg * bg_strength

    return delta


def map_field_to_delta_qp(
    normalized_field: np.ndarray,
    cfg: QPMappingConfig,
) -> np.ndarray:
    """Map normalized importance field [0, 1] to per-CTU QP offsets (dQP).

    Phase 3 primary output. The delta map is calibration-free: it encodes
    the *relative* redistribution of bits between ROI and background and
    is independent of Q_base. The Phase 2 encoder composes it with the
    run-time Q_base as:  Q_final = clip(Q_base + delta, 1, 51).

    Args:
        normalized_field: 2D array (n_rows, n_cols) in [0, 1].
        cfg: QP mapping configuration.

    Returns:
        2D float array of signed deltas in
        [-cfg.delta_roi, +cfg.delta_bg].
    """
    delta = _asymmetric_delta(normalized_field, cfg)

    logger.debug(
        "dQP map: range [%+.2f, %+.2f], mean %+.3f, ROI CTUs: %d/%d",
        float(np.min(delta)),
        float(np.max(delta)),
        float(np.mean(delta)),
        int(np.sum(normalized_field >= cfg.mu)),
        delta.size,
    )
    return delta


def map_field_to_qp(
    normalized_field: np.ndarray,
    cfg: QPMappingConfig,
) -> np.ndarray:
    """LEGACY: absolute-QP map calibrated at cfg.qp_base.

    Retained for backward compatibility with Phase 1 / pilot v1 workflows.
    New Phase 3 work MUST use `map_field_to_delta_qp`.
    """
    qp_map = float(cfg.qp_base) + _asymmetric_delta(normalized_field, cfg)

    logger.debug(
        "QP map (absolute, legacy): range [%.1f, %.1f]",
        float(np.min(qp_map)),
        float(np.max(qp_map)),
    )
    return qp_map
