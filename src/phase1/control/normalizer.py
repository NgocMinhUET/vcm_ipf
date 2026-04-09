"""Temporal-stable field normalization.

Implements EMA-smoothed percentile normalization to prevent
frame-to-frame QP oscillation caused by sudden field magnitude changes.

    a_t = rho_eff * a_{t-1} + (1 - rho_eff) * P_low(Phi_t)
    b_t = rho_eff * b_{t-1} + (1 - rho_eff) * P_high(Phi_t)
    Phi_hat_t = clip((Phi_t - a_t) / (b_t - a_t + eps), 0, 1)

Progressive warmup: rho_eff ramps from 0 (instant adapt) to target rho
over the first ~1/(1-rho) frames, preventing cold-start instability when
track-age warmup causes early field magnitude ramps.
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

    Uses progressive rho warmup: the effective EMA factor starts at 0
    (pure instantaneous) and increases to the target rho as frame count
    grows. This is equivalent to an unbiased running average during the
    initial transient, converging to exponential smoothing at steady state.
    """

    def __init__(self, cfg: NormalizationConfig):
        self.rho = cfg.rho
        self.p_low = cfg.percentile_low
        self.p_high = cfg.percentile_high
        self.eps = cfg.eps_n

        self._a: Optional[float] = None  # running lower bound
        self._b: Optional[float] = None  # running upper bound
        self._frame_count: int = 0

    def _effective_rho(self) -> float:
        """Adaptive EMA factor: rho_eff = min(rho, 1 - 1/t).

        t=1: rho_eff=0 (use raw value directly)
        t=2: rho_eff=0.5 (equal weight)
        t=N: rho_eff → rho once t >= 1/(1-rho)
        """
        if self._frame_count <= 1:
            return 0.0
        return min(self.rho, 1.0 - 1.0 / self._frame_count)

    def normalize(self, field_map: np.ndarray) -> np.ndarray:
        """Normalize a raw field map to [0, 1] with temporal smoothing.

        Args:
            field_map: 2D array (n_rows, n_cols) of raw field values.

        Returns:
            Normalized field map in [0, 1], same shape as input.
        """
        self._frame_count += 1

        p_lo = float(np.percentile(field_map, self.p_low))
        p_hi = float(np.percentile(field_map, self.p_high))

        rho_eff = self._effective_rho()

        if self._a is None:
            self._a = p_lo
            self._b = p_hi
        else:
            self._a = rho_eff * self._a + (1.0 - rho_eff) * p_lo
            self._b = rho_eff * self._b + (1.0 - rho_eff) * p_hi

        denom = self._b - self._a + self.eps
        normalized = np.clip((field_map - self._a) / denom, 0.0, 1.0)

        logger.debug(
            "Normalizer: t=%d, rho_eff=%.3f, a=%.4f, b=%.4f, range [%.4f, %.4f]",
            self._frame_count, rho_eff,
            self._a, self._b,
            float(np.min(normalized)), float(np.max(normalized)),
        )
        return normalized

    def reset(self) -> None:
        """Reset running estimates for a new video."""
        self._a = None
        self._b = None
        self._frame_count = 0
