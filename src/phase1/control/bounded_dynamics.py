"""Bounded QP dynamics: temporal low-pass filter + slew-rate limiter.

Prevents abrupt QP jumps between consecutive frames by applying:
    1. Low-pass filter:  Q_bar_{i,t} = eta * Q_raw_{i,t} + (1 - eta) * Q_bar_{i,t-1}
    2. Slew-rate limit:  Q_ctl_{i,t} = clip(Q_bar, Q_{i,t-1} - delta_slew, Q_{i,t-1} + delta_slew)
    3. Final clamp:      Q_{i,t}     = clip(round(Q_ctl), Q_min, Q_max)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from phase1.core.config import BoundedDynamicsConfig
from phase1.utils.log import get_logger

logger = get_logger("control.bounded_dynamics")


class BoundedQPController:
    """Stateful controller that smooths and clamps QP maps across frames.

    Maintains per-CTU state from the previous frame.
    """

    def __init__(self, cfg: BoundedDynamicsConfig):
        self.eta = cfg.eta
        self.delta_slew = cfg.delta_slew
        self.qp_min = cfg.qp_min
        self.qp_max = cfg.qp_max

        self._q_bar_prev: Optional[np.ndarray] = None
        self._q_final_prev: Optional[np.ndarray] = None

    def apply(self, raw_qp: np.ndarray) -> np.ndarray:
        """Apply bounded dynamics to a raw QP map.

        Args:
            raw_qp: 2D array (n_rows, n_cols) of raw QP values (float).

        Returns:
            2D array (n_rows, n_cols) of final integer QP values.
        """
        # Step 1: Temporal low-pass filter
        if self._q_bar_prev is None:
            q_bar = raw_qp.copy()
        else:
            q_bar = self.eta * raw_qp + (1.0 - self.eta) * self._q_bar_prev

        # Step 2: Slew-rate limiting
        if self._q_final_prev is not None:
            q_ctl = np.clip(
                q_bar,
                self._q_final_prev - self.delta_slew,
                self._q_final_prev + self.delta_slew,
            )
        else:
            q_ctl = q_bar

        # Step 3: Round and clamp to valid range
        q_final = np.clip(np.round(q_ctl), self.qp_min, self.qp_max).astype(np.int32)

        # Update state
        self._q_bar_prev = q_bar.copy()
        self._q_final_prev = q_final.astype(np.float64)

        logger.debug(
            "BoundedQP: range [%d, %d], mean=%.1f",
            int(np.min(q_final)),
            int(np.max(q_final)),
            float(np.mean(q_final)),
        )
        return q_final

    def reset(self) -> None:
        """Reset state for a new video."""
        self._q_bar_prev = None
        self._q_final_prev = None
