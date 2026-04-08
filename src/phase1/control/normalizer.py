"""Temporal-stable field normalization.

Implements EMA-smoothed percentile normalization to prevent
frame-to-frame QP oscillation caused by sudden field magnitude changes.

    a_t = rho * a_{t-1} + (1 - rho) * P_low(Phi_t)
    b_t = rho * b_{t-1} + (1 - rho) * P_high(Phi_t)
    Phi_hat_t = clip((Phi_t - a_t) / (b_t - a_t + eps), 0, 1)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from phase1.core.config import NormalizationConfig
from phase1.utils.log import get_logger

logger = get_logger("control.normalizer")


class TemporalNormalizer:
    """EMA-smoothed percentile normalizer for the importance field.

    Maintains running estimates of low/high percentiles across frames
    to produce stable [0, 1] normalization.
    """

    def __init__(self, cfg: NormalizationConfig):
        self.rho = cfg.rho
        self.p_low = cfg.percentile_low
        self.p_high = cfg.percentile_high
        self.eps = cfg.eps_n

        self._a: Optional[float] = None  # running lower bound
        self._b: Optional[float] = None  # running upper bound

    def normalize(self, field_map: np.ndarray) -> np.ndarray:
        """Normalize a raw field map to [0, 1] with temporal smoothing.

        Args:
            field_map: 2D array (n_rows, n_cols) of raw field values.

        Returns:
            Normalized field map in [0, 1], same shape as input.
        """
        p_lo = float(np.percentile(field_map, self.p_low))
        p_hi = float(np.percentile(field_map, self.p_high))

        if self._a is None:
            self._a = p_lo
            self._b = p_hi
        else:
            self._a = self.rho * self._a + (1.0 - self.rho) * p_lo
            self._b = self.rho * self._b + (1.0 - self.rho) * p_hi

        denom = self._b - self._a + self.eps
        normalized = np.clip((field_map - self._a) / denom, 0.0, 1.0)

        logger.debug(
            "Normalizer: a=%.4f, b=%.4f, out range [%.4f, %.4f]",
            self._a, self._b, float(np.min(normalized)), float(np.max(normalized)),
        )
        return normalized

    def reset(self) -> None:
        """Reset running estimates for a new video."""
        self._a = None
        self._b = None
