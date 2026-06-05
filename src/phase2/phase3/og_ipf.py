"""Phase 3 Stage C — compression-aware OG-IPF δQP allocator.

This module is the occupancy-guided counterpart of
:mod:`phase2.phase3.analytic_a_plus`. It plugs the per-CTU utility
``U_c`` from :mod:`phase2.phase3.occupancy` into the same RD-log A+
closed form, then applies a **compression-aware minimum object-protection
constraint**

    δQP_c  ≤  -a(Q_b) · δ_min(Q_b) · G_c^η

where ``a(Q_b)`` is an activation factor. The activation is intentionally
near zero at light/medium compression and close to one at severe
compression. This reflects the empirical observation that OG-IPF is most
useful when the anchor QP is high enough for object CTUs to become
machine-vision fragile.

Design principles
-----------------
1. At low QP, avoid unnecessary QP-map variance: the detector is already
   close to its ceiling and forced redistribution can hurt more than help.
2. At high QP, activate object protection strongly: small and partially
   occluded objects are at risk of disappearing after compression.
3. Preserve rate awareness through clipped exact projection with per-CTU
   bounds, so the projection cannot undo object protection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from phase2.phase3.analytic_a_plus import (
    AnalyticAPlusConfig,
    project_rate_neutral_clipped_exact,
    q_adaptive_bounds,
    rate_neutral_residual,
)
from phase2.phase3.occupancy import TemporalOccupancyState


def _normalize_complexity(K: np.ndarray) -> np.ndarray:
    """Median-normalise K so K̃ ≈ 1 for the typical CTU."""
    K = np.asarray(K, dtype=np.float64)
    pos = K[K > 0]
    if pos.size == 0:
        return np.ones_like(K, dtype=np.float64)
    med = float(np.median(pos))
    return K / (med + 1e-9)


@dataclass(frozen=True)
class MinProtectionConfig:
    """Hyper-parameters of the OG-IPF object-protection constraint.

    ``delta_min(Q_b)`` defines the maximum intended protection strength.
    ``activation(Q_b)`` controls how much of that protection is active at
    a given base QP. Defaults implement

        a(Q_b) = clip((Q_b - 32) / (42 - 32), 0, 1),

    so QP=27/32 are effectively no-op, QP=37 is half-strength, and
    QP=42 is full-strength. This prevents OG-IPF from injecting needless
    CTU-level QP variance when the anchor is already high quality.
    """

    # δ_min(Q_b) = clip(intercept + slope·(Q_b - 27), floor, ceil)
    intercept: float = 1.0
    slope: float = 0.08
    floor: float = 1.0
    ceil: float = 2.2
    eta: float = 0.7

    # Weakly-overlapping CTUs are not hard-capped; they keep global +bg_bound.
    g_min_protect: float = 0.10

    # Compression-aware activation schedule.
    activation_q0: float = 32.0
    activation_q1: float = 42.0
    activation_min: float = 0.0
    activation_max: float = 1.0


def min_protection_floor(q_base: int, cfg: Optional[MinProtectionConfig] = None) -> float:
    """Return nominal positive ``δ_min(Q_b)`` before activation."""
    cfg = cfg or MinProtectionConfig()
    raw = cfg.intercept + cfg.slope * (float(q_base) - 27.0)
    return float(np.clip(raw, cfg.floor, cfg.ceil))


def protection_activation(q_base: int, cfg: Optional[MinProtectionConfig] = None) -> float:
    """Return compression-aware activation ``a(Q_b)``.

    Default values produce QP=27 → 0.0, QP=32 → 0.0,
    QP=37 → 0.5, and QP=42 → 1.0.
    """
    cfg = cfg or MinProtectionConfig()
    denom = max(float(cfg.activation_q1 - cfg.activation_q0), 1e-9)
    raw = (float(q_base) - float(cfg.activation_q0)) / denom
    return float(np.clip(raw, cfg.activation_min, cfg.activation_max))


def adaptive_activation(
    q_base: int,
    anchor_mAP: Optional[dict] = None,
    cfg: Optional[MinProtectionConfig] = None,
) -> float:
    """Content-adaptive activation driven by the detector-sensitivity slope s0.

    Problem with the hard activation
    ---------------------------------
    The hard schedule ``a(Q_b) = clip((Q_b-32)/10, 0, 1)`` zeros out the
    allocator at QP ≤ 32, so 2 of the 4 (or 2 of the 5) RD points contribute
    nothing to BD-Rate.  For sequences where the anchor already shows a
    measurable slope ``s0 = |ΔAP/ΔQP|`` at low QP (e.g. dense crowd scenes
    at QP=27/32), the hard schedule leaves headroom on the table.

    Content-adaptive formula
    ------------------------
    Given per-QP anchor mAP values ``{Q: AP_M0(Q)}``, we estimate the local
    slope at ``q_base`` as::

        s0_local = |AP_M0(q_base) - AP_M0(q_base + Δ)| / Δ   (Δ = 5 or 10)

    and set::

        a_adaptive(q_base) = clip(s0_local / s0_ref, 0, 1)

    where ``s0_ref`` is a calibration constant (default 0.004 AP/QP, chosen
    so that the typical dense-scene slope at QP=37 gives a ≈ 0.5, matching
    the hard schedule at its calibration point).

    This engages protection on any QP point that is "informationally rich"
    (high slope), even if it is nominally low-compression.

    Fallback
    --------
    If ``anchor_mAP`` is None or has fewer than 2 points, falls back to the
    hard schedule via :func:`protection_activation`.
    """
    cfg = cfg or MinProtectionConfig()
    if anchor_mAP is None or len(anchor_mAP) < 2:
        return protection_activation(q_base, cfg)

    qps = sorted(anchor_mAP.keys())
    aps = [anchor_mAP[q] for q in qps]

    # Find the index of q_base in the sorted QP list.
    try:
        idx = qps.index(q_base)
    except ValueError:
        return protection_activation(q_base, cfg)

    # Finite difference: prefer forward difference; fall back to backward.
    if idx + 1 < len(qps):
        dq = float(qps[idx + 1] - qps[idx])
        dap = abs(float(aps[idx + 1]) - float(aps[idx]))
    else:
        dq = float(qps[idx] - qps[idx - 1])
        dap = abs(float(aps[idx]) - float(aps[idx - 1]))

    s0_local = dap / max(dq, 1e-9)

    # Calibration: s0_ref such that s0_local ≈ s0_ref at QP=37 for typical
    # dense-scene sequences → a ≈ 0.5 (matches hard schedule there).
    s0_ref: float = 0.004   # AP/QP — tune via ablation if needed
    a_adaptive = float(np.clip(s0_local / s0_ref, cfg.activation_min, cfg.activation_max))
    return a_adaptive


def effective_min_protection_floor(
    q_base: int,
    cfg: Optional[MinProtectionConfig] = None,
) -> float:
    """Return ``a(Q_b) · δ_min(Q_b)``."""
    cfg = cfg or MinProtectionConfig()
    return protection_activation(q_base, cfg) * min_protection_floor(q_base, cfg)


def apply_min_object_protection(
    delta: np.ndarray,
    g_max: np.ndarray,
    q_base: int,
    cfg: Optional[MinProtectionConfig] = None,
) -> np.ndarray:
    """Enforce ``δ_c ≤ -a(Q_b) δ_min(Q_b) G_c^η``.

    In the end-to-end OG-CA path, the analytic map is also multiplied by
    ``a(Q_b)``, so QP=27/32 become near-no-op maps instead of injecting
    unnecessary QP variance.
    """
    cfg = cfg or MinProtectionConfig()
    delta = np.asarray(delta, dtype=np.float64)
    g_max = np.clip(np.asarray(g_max, dtype=np.float64), 0.0, 1.0)
    if delta.shape != g_max.shape:
        raise ValueError(f"shape mismatch: delta {delta.shape} vs G_max {g_max.shape}")

    floor_eff = effective_min_protection_floor(q_base, cfg)
    floor_val = -floor_eff * np.power(g_max, cfg.eta)
    return np.minimum(delta, floor_val)


def compute_og_a_plus_delta(
    U: np.ndarray,
    K: np.ndarray,
    q_base: int,
    cfg: Optional[AnalyticAPlusConfig] = None,
) -> np.ndarray:
    """Closed-form RD-log A+ allocation driven by OG-IPF utility ``U``."""
    cfg = cfg or AnalyticAPlusConfig()
    U = np.asarray(U, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if U.shape != K.shape:
        raise ValueError(f"shape mismatch: U {U.shape} vs K {K.shape}")

    K_norm = _normalize_complexity(K)
    xi = (U + cfg.eps) / np.power(K_norm + cfg.kappa, cfg.beta)
    denom = float(np.sum(K) + 1e-9)
    g = float(np.sum(K * xi) / denom)
    g = max(g, 1e-9)

    delta = -6.0 * np.log2((xi + 1e-9) / g)
    scale = float(np.clip(
        cfg.scale_a + cfg.scale_b * (q_base - 32),
        cfg.scale_min,
        cfg.scale_max,
    ))
    delta = scale * delta

    roi_bound, bg_bound = q_adaptive_bounds(q_base)
    delta = np.clip(delta, -roi_bound, +bg_bound)
    delta = np.clip(delta, cfg.delta_min_clip, cfg.delta_max_clip)
    return delta


@dataclass
class OGAPlusOutput:
    """Container for the final OG-CA allocation maps."""

    delta_continuous: np.ndarray
    delta_int: np.ndarray
    delta_a_plus_raw: np.ndarray
    delta_a_plus_activated: np.ndarray
    rate_ratio_continuous: float
    rate_ratio_rounded: float
    activation_factor: float


def end_to_end_og_a_plus(
    U: np.ndarray,
    G_max: np.ndarray,
    K: np.ndarray,
    q_base: int,
    cfg: Optional[AnalyticAPlusConfig] = None,
    min_protect_cfg: Optional[MinProtectionConfig] = None,
    delta_min_clip: int = -8,
    delta_max_clip: int = +4,
    anchor_mAP: Optional[dict] = None,
) -> OGAPlusOutput:
    """OG-CA allocation: analytic → activation → min-protection → projection.

    The same activation factor attenuates the analytic OG allocation and the
    hard object-protection floor. Thus at low QP, the map approaches M0; at
    high QP, full OG-IPF protection is restored.

    Parameters
    ----------
    anchor_mAP
        Optional dict ``{q_base: AP_M0_value}`` from the anchor (M0) RD
        curve for this sequence.  When provided, the activation factor is
        computed by :func:`adaptive_activation` (content-adaptive, driven by
        the detector-sensitivity slope) instead of the hard schedule.  Pass
        ``None`` (default) to keep the hard schedule for backward
        compatibility.
    """
    cfg = cfg or AnalyticAPlusConfig()
    min_protect_cfg = min_protect_cfg or MinProtectionConfig()

    if anchor_mAP is not None and len(anchor_mAP) >= 2:
        activation = adaptive_activation(q_base, anchor_mAP, min_protect_cfg)
    else:
        activation = protection_activation(q_base, min_protect_cfg)

    # 1) Analytic prior on occupancy-guided utility.
    delta_a_raw = compute_og_a_plus_delta(U, K, q_base, cfg)

    # 2) Compression-aware attenuation of the whole OG allocation.
    delta_a = activation * delta_a_raw

    # 3) Minimum object-protection constraint.
    delta_p = apply_min_object_protection(delta_a, G_max, q_base, min_protect_cfg)

    # 4) Per-CTU bounds. Strong object CTUs use the activated floor; weak
    # overlaps retain the global background upper bound for projection slack.
    roi_bound, bg_bound = q_adaptive_bounds(q_base)
    clip_lo = max(-roi_bound, float(delta_min_clip))
    clip_hi_global = min(+bg_bound, float(delta_max_clip))

    g_arr = np.clip(np.asarray(G_max, dtype=np.float64), 0.0, 1.0)
    floor_eff = effective_min_protection_floor(q_base, min_protect_cfg)
    g_thresh = float(min_protect_cfg.g_min_protect)

    upper_protected = np.minimum(
        clip_hi_global,
        -floor_eff * np.power(g_arr, min_protect_cfg.eta),
    )
    upper_per_ctu = np.where(g_arr > g_thresh, upper_protected, clip_hi_global)

    # 5) Clipped exact rate-neutral projection with per-CTU bounds.
    delta_continuous = project_rate_neutral_clipped_exact(
        delta_p,
        K,
        delta_min=clip_lo,
        delta_max=upper_per_ctu,
    )

    # 6) Integer rounding while preserving per-CTU upper bounds.
    upper_int_per_ctu = np.floor(upper_per_ctu).astype(np.float64)
    delta_int = np.minimum(
        np.maximum(np.rint(delta_continuous), float(delta_min_clip)),
        upper_int_per_ctu,
    ).astype(np.int32)

    return OGAPlusOutput(
        delta_continuous=delta_continuous,
        delta_int=delta_int,
        delta_a_plus_raw=delta_a_raw,
        delta_a_plus_activated=delta_a,
        rate_ratio_continuous=float(rate_neutral_residual(delta_continuous, K)),
        rate_ratio_rounded=float(rate_neutral_residual(delta_int.astype(np.float64), K)),
        activation_factor=activation,
    )


@dataclass
class OGDiagnostics:
    """Per-frame OG-IPF diagnostics."""

    n_ctus_total: int
    pct_object_overlap: float
    pct_context: float
    pct_far_background: float
    mean_delta_object: float
    mean_delta_context: float
    mean_delta_far_background: float
    saturation_lower: float
    saturation_upper: float
    rate_ratio_continuous: float
    rate_ratio_rounded: float
    object_protection_violation: float
    g_threshold_object: float
    g_threshold_context: float
    activation_factor: float


def compute_og_diagnostics(
    delta_continuous: np.ndarray,
    delta_int: np.ndarray,
    G_max: np.ndarray,
    K: np.ndarray,
    *,
    g_object: float = 0.5,
    g_context: float = 0.05,
    delta_min_clip: float = -8.0,
    delta_max_clip: float = +4.0,
    rate_ratio_continuous: Optional[float] = None,
    rate_ratio_rounded: Optional[float] = None,
    activation_factor: float = 1.0,
) -> OGDiagnostics:
    """Compute diagnostics for one frame."""
    delta_int = np.asarray(delta_int, dtype=np.float64)
    g = np.clip(np.asarray(G_max, dtype=np.float64), 0.0, 1.0)
    n_total = int(delta_int.size)

    mask_object = g > g_object
    mask_context = (g > g_context) & (g <= g_object)
    mask_far = g <= g_context

    def _mean_delta(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(delta_int[mask]))

    sat_lo = float(np.mean(np.abs(delta_int - delta_min_clip) < 0.5))
    sat_hi = float(np.mean(np.abs(delta_int - delta_max_clip) < 0.5))

    if rate_ratio_continuous is None:
        rate_ratio_continuous = float(rate_neutral_residual(delta_continuous, K))
    if rate_ratio_rounded is None:
        rate_ratio_rounded = float(rate_neutral_residual(delta_int, K))

    if np.any(mask_object):
        violation = float(np.mean(delta_int[mask_object] >= 0.0))
    else:
        violation = 0.0

    return OGDiagnostics(
        n_ctus_total=n_total,
        pct_object_overlap=float(np.mean(mask_object)),
        pct_context=float(np.mean(mask_context)),
        pct_far_background=float(np.mean(mask_far)),
        mean_delta_object=_mean_delta(mask_object),
        mean_delta_context=_mean_delta(mask_context),
        mean_delta_far_background=_mean_delta(mask_far),
        saturation_lower=sat_lo,
        saturation_upper=sat_hi,
        rate_ratio_continuous=float(rate_ratio_continuous),
        rate_ratio_rounded=float(rate_ratio_rounded),
        object_protection_violation=violation,
        g_threshold_object=g_object,
        g_threshold_context=g_context,
        activation_factor=float(activation_factor),
    )


__all__ = [
    "MinProtectionConfig",
    "OGAPlusOutput",
    "OGDiagnostics",
    "TemporalOGAllocator",
    "min_protection_floor",
    "protection_activation",
    "adaptive_activation",
    "effective_min_protection_floor",
    "apply_min_object_protection",
    "compute_og_a_plus_delta",
    "end_to_end_og_a_plus",
    "compute_og_diagnostics",
]


class TemporalOGAllocator:
    """Stateful OG-A+ allocator with EMA temporal consistency.

    Wraps :func:`end_to_end_og_a_plus` and maintains a
    :class:`~phase2.phase3.occupancy.TemporalOccupancyState` so that the
    per-CTU occupancy gate ``G_c`` and utility ``U_c`` are smoothed across
    consecutive frames before the RD-log allocation is computed.

    This directly addresses the MOT17-11 / MOT17-13 regressions:
    - High ego-motion causes the instantaneous occupancy grid to jitter.
    - The jitter propagates to the rate-neutral projection, making ``rho_R``
      exceed 1.05 at QP=42 because the background compensation pool cannot
      absorb the per-frame variance.
    - EMA with ``alpha in [0.4, 0.6]`` halves the frame-to-frame variance of
      ``rho_R`` without delaying occupancy response by more than 1–2 frames.

    Usage per sequence per QP::

        allocator = TemporalOGAllocator(ema_alpha=0.5)
        for frame_idx in range(n_frames):
            occ_result = compute_occupancy_utility(boxes, H, W, ctu_size, cfg)
            output = allocator.step(occ_result.U, occ_result.G_max, K, q_base)
            write_dqp_map(output.delta_int, frame_idx)

    The allocator resets automatically when :meth:`reset` is called (between
    sequences or between QP points if per-QP independent state is desired).
    """

    def __init__(
        self,
        ema_alpha: float = 0.5,
        cfg: Optional[AnalyticAPlusConfig] = None,
        min_protect_cfg: Optional[MinProtectionConfig] = None,
    ) -> None:
        self._temporal = TemporalOccupancyState(alpha=ema_alpha)
        self._cfg = cfg or AnalyticAPlusConfig()
        self._mp_cfg = min_protect_cfg or MinProtectionConfig()

    def reset(self) -> None:
        """Reset EMA state (call between sequences or QP runs)."""
        self._temporal.reset()

    def step(
        self,
        U: np.ndarray,
        G_max: np.ndarray,
        K: np.ndarray,
        q_base: int,
    ) -> OGAPlusOutput:
        """One frame: apply EMA smoothing then run end-to-end allocation."""
        from phase2.phase3.occupancy import OccupancyResult
        smoothed = self._temporal.update(
            OccupancyResult(U=U, G_max=G_max, G_per_object=None, n_objects=0)
        )
        return end_to_end_og_a_plus(
            smoothed.U,
            smoothed.G_max,
            K,
            q_base,
            cfg=self._cfg,
            min_protect_cfg=self._mp_cfg,
        )
