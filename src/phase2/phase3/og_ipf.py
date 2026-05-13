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
) -> OGAPlusOutput:
    """OG-CA allocation: analytic → activation → min-protection → projection.

    The same activation factor attenuates the analytic OG allocation and the
    hard object-protection floor. Thus at low QP, the map approaches M0; at
    high QP, full OG-IPF protection is restored.
    """
    cfg = cfg or AnalyticAPlusConfig()
    min_protect_cfg = min_protect_cfg or MinProtectionConfig()

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
    "min_protection_floor",
    "protection_activation",
    "effective_min_protection_floor",
    "apply_min_object_protection",
    "compute_og_a_plus_delta",
    "end_to_end_og_a_plus",
    "compute_og_diagnostics",
]
