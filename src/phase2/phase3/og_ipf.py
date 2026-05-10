"""Phase 3 Stage C — OG-IPF analytic δQP allocator + min-protection.

This module is the *occupancy-guided* counterpart of
:mod:`phase2.phase3.analytic_a_plus`. It plugs the per-CTU utility
``U_c`` from :mod:`phase2.phase3.occupancy` into the same RD-log A+
closed form, then enforces a **minimum object-protection constraint**

    δQP_c  ≤  -δ_min(Q_b) · G_c^η                                 (★)

so that any CTU with high object-occupancy gate ``G_c`` is guaranteed
at least mild negative δQP, even if its rate complexity ``K_c`` would
otherwise pull the analytic allocation toward 0 or positive δQP.

Why this constraint matters
---------------------------
The classical IPF and the original A+ allocator are *purely analytic*
optimisations of an RD-log Lagrangian:

    minimise   Σ K_c · 2^{-(Q_b + δ_c)/6}
    subject to Σ K_c · 2^{-(Q_b + δ_c)/6} = Σ K_c

There is no hard floor on δQP at object-overlapping CTUs. Empirically
this leads to two failure modes (PROJECT_STATE §7.4 + §7.20):

1. *Small distant pedestrians* → small ``√(w h)`` → small mass → the
   analytic δ_c can land at +1 .. +3 (i.e. *worse* quality than the
   anchor) because the optimiser harvests rate from "low-importance"
   CTUs to fund nearby high-mass objects. This kills the very
   detections the bit budget should preserve.
2. *Texture-rich background CTUs near objects* sometimes get protected
   more than the object's CTUs themselves, because their high ``K_c``
   makes them "expensive" in the Lagrangian.

(★) is the smallest possible correction: it touches only CTUs with
``G_c > 0`` and only to the extent that ``G_c`` is large. ``δ_min``
grows with ``Q_b`` because, at high QP, +1 of compression headroom
costs more mAP — so the minimum protection must be larger to keep
the same per-detection quality margin.

Design property
---------------
*Rate-neutrality is preserved by construction*: the constraint is
applied **before** the rate-neutral projection. The projection then
re-normalises the global shift so ``Σ K · 2^{-δ/6} = Σ K`` holds
exactly (modulo integer rounding).

Public functions
----------------
* :func:`min_protection_floor`            — δ_min(Q_b) schedule
* :func:`apply_min_object_protection`     — enforce (★)
* :func:`compute_og_a_plus_delta`         — analytic prior on U_c
* :func:`end_to_end_og_a_plus`            — analytic prior + (★) +
                                            rate-neutral projection +
                                            (optional) integer rounding
* :func:`compute_og_diagnostics`          — produces the §9 diagnostics
                                            payload for the apply step
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from phase2.phase3.analytic_a_plus import (
    AnalyticAPlusConfig,
    project_rate_neutral_clipped_exact,
    project_rate_neutral_exact,
    q_adaptive_bounds,
    rate_neutral_residual,
)


def _normalize_complexity(K: np.ndarray) -> np.ndarray:
    """Median-normalise K so K̃ ≈ 1 for the typical CTU.

    Duplicated locally (not imported from ``analytic_a_plus``) to keep
    OG-IPF self-contained and to avoid taking a dependency on a
    ``_``-prefixed private helper.
    """
    K = np.asarray(K, dtype=np.float64)
    pos = K[K > 0]
    if pos.size == 0:
        return np.ones_like(K, dtype=np.float64)
    med = float(np.median(pos))
    return K / (med + 1e-9)


# ---------------------------------------------------------------------------
# Min-protection schedule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MinProtectionConfig:
    """Hyper-parameters of the (★) constraint.

    The defaults are documented in PROJECT_STATE §7.20 and were chosen
    so that:

    * a fully-occupying CTU (``G_c = 1``) at ``Q_b = 27`` gets at
      least δQP = -1.0, growing to ≈ -2.2 at ``Q_b = 42``;
    * a half-occupying CTU (``G_c = 0.5``) gets ≈ -0.6 at ``Q_b = 27``
      (because of the ``η = 0.7`` exponent), growing to ≈ -1.4 at
      ``Q_b = 42``;
    * boundary / weakly-occupying CTUs (``G_c < 0.1``) are essentially
      unaffected (floor < -0.2).
    """

    # δ_min(Q_b) = clip(intercept + slope·(Q_b - 27), floor, ceil)
    intercept: float = 1.0
    slope:     float = 0.08
    floor:     float = 1.0
    ceil:      float = 2.2
    eta:       float = 0.7


def min_protection_floor(q_base: int, cfg: Optional[MinProtectionConfig] = None
                          ) -> float:
    """Smooth schedule for ``δ_min(Q_b)``.

    Returns a *positive* number; the constraint (★) uses ``-δ_min``.
    """
    cfg = cfg or MinProtectionConfig()
    raw = cfg.intercept + cfg.slope * (float(q_base) - 27.0)
    return float(np.clip(raw, cfg.floor, cfg.ceil))


def apply_min_object_protection(
    delta: np.ndarray,
    g_max: np.ndarray,
    q_base: int,
    cfg: Optional[MinProtectionConfig] = None,
) -> np.ndarray:
    """Enforce ``δ_c ≤ -δ_min(Q_b) · G_c^η``.

    Parameters
    ----------
    delta
        Per-CTU δQP map (float, before integer rounding).
    g_max
        Per-CTU occupancy gate ``G_c = max_j G_{j,c}``, in ``[0, 1]``.
    q_base
        Anchor QP for this encode (used to compute ``δ_min``).
    cfg
        :class:`MinProtectionConfig`; defaults documented above.

    Returns
    -------
    np.ndarray
        Same shape as ``delta``, with the constraint applied.

    Notes
    -----
    The operation is ``δ_c ← min(δ_c, -δ_min · G_c^η)`` — i.e. *push
    down* CTUs that are not negative enough. CTUs already below the
    floor (e.g. analytic A+ already gave them δ_c = -3) are left
    untouched. The function is idempotent.
    """
    cfg = cfg or MinProtectionConfig()
    delta = np.asarray(delta, dtype=np.float64)
    g_max = np.clip(np.asarray(g_max, dtype=np.float64), 0.0, 1.0)
    if delta.shape != g_max.shape:
        raise ValueError(
            f"shape mismatch: delta {delta.shape} vs G_max {g_max.shape}")
    floor_val = -min_protection_floor(q_base, cfg) * np.power(g_max, cfg.eta)
    return np.minimum(delta, floor_val)


# ---------------------------------------------------------------------------
# OG-A+ analytic prior on U_c
# ---------------------------------------------------------------------------

def compute_og_a_plus_delta(
    U: np.ndarray,
    K: np.ndarray,
    q_base: int,
    cfg: Optional[AnalyticAPlusConfig] = None,
) -> np.ndarray:
    """Closed-form RD-log A+ allocation driven by the OG-IPF utility ``U``.

    Mirrors :func:`phase2.phase3.analytic_a_plus.compute_a_plus_delta`
    but takes ``U`` (occupancy-guided) instead of the legacy ``Φ``.
    The two functions are intentionally identical in math; only the
    upstream signal differs. We keep them as **separate symbols** so
    callers can be inspected for "did the OG variant land here?" via
    ``grep`` rather than via runtime branching.
    """
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
        cfg.scale_min, cfg.scale_max,
    ))
    delta = scale * delta

    roi_bound, bg_bound = q_adaptive_bounds(q_base)
    delta = np.clip(delta, -roi_bound, +bg_bound)
    delta = np.clip(delta, cfg.delta_min_clip, cfg.delta_max_clip)
    return delta


# ---------------------------------------------------------------------------
# End-to-end helper
# ---------------------------------------------------------------------------

@dataclass
class OGAPlusOutput:
    """Container with all the maps a downstream apply / dataset
    builder is likely to want.

    Attributes
    ----------
    delta_continuous
        Float δQP map after analytic + min-protection + clipped exact
        rate-neutral projection. **Not yet rounded.**
    delta_int
        Integer δQP map after rounding to ``{-8, ..., +4}``. This is
        what gets written to ``qp_*.txt``.
    delta_a_plus_raw
        The *pre*-min-protection analytic prior. Useful for paper
        figures and for the "how often does (★) actually fire?"
        diagnostic.
    rate_ratio_continuous
        ``Σ K · 2^{-δ_continuous/6} / Σ K`` — should be ≈ 1.0 (machine
        precision) by construction.
    rate_ratio_rounded
        Same ratio after integer rounding. Typical drift 0.5–2 %.
    """

    delta_continuous:      np.ndarray
    delta_int:             np.ndarray
    delta_a_plus_raw:      np.ndarray
    rate_ratio_continuous: float
    rate_ratio_rounded:    float


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
    """One-shot OG-A+ allocation: analytic + (★) + projection + rounding.

    Parameters
    ----------
    U, G_max, K
        Outputs of :func:`phase2.phase3.occupancy.compute_occupancy_utility`
        plus the per-CTU rate complexity from
        :mod:`phase2.phase3.rate_surrogate`.
    q_base
        Anchor QP.
    cfg
        Analytic A+ config (clipping bounds, scale schedule).
    min_protect_cfg
        Min-protection config for (★).
    delta_min_clip, delta_max_clip
        VVC legal-range clip applied to the rounded integer map.
    """
    cfg = cfg or AnalyticAPlusConfig()
    min_protect_cfg = min_protect_cfg or MinProtectionConfig()

    # 1) Analytic prior on the occupancy-guided utility.
    delta_a = compute_og_a_plus_delta(U, K, q_base, cfg)

    # 2) Minimum object-protection constraint applied to the analytic
    #    prior: δ_c ≤ -δ_min(Q_b) · G_c^η. This is also exposed as a
    #    *per-CTU upper bound* below so the projection can never undo it.
    delta_p = apply_min_object_protection(delta_a, G_max, q_base, min_protect_cfg)

    # 3) Build per-CTU bounds. Lower bound is the global Q-adaptive
    #    floor (tighter than the legal range). Upper bound at protected
    #    CTUs is `-δ_min · G^η`, ensuring rate-neutral projection cannot
    #    push them above zero (the §7.20 / Test 6 failure mode).
    roi_bound, bg_bound = q_adaptive_bounds(q_base)
    clip_lo = max(-roi_bound, float(delta_min_clip))
    clip_hi_global = min(+bg_bound, float(delta_max_clip))
    g_arr = np.clip(np.asarray(G_max, dtype=np.float64), 0.0, 1.0)
    delta_min_q = min_protection_floor(q_base, min_protect_cfg)
    upper_per_ctu = np.minimum(
        clip_hi_global,
        -delta_min_q * np.power(g_arr, min_protect_cfg.eta),
    )
    # Where G == 0 the upper-bound formula gives 0 (since 0^η = 0), but
    # we want unprotected CTUs to keep the original +bg_bound ceiling
    # so background can still be encoded *worse* than the anchor.
    upper_per_ctu = np.where(g_arr > 0.0, upper_per_ctu, clip_hi_global)

    # 4) Clipped exact rate-neutral projection (per-CTU bounds).
    delta_continuous = project_rate_neutral_clipped_exact(
        delta_p, K, delta_min=clip_lo, delta_max=upper_per_ctu,
    )

    # 5) Integer rounding. We clip the rounded value with the
    #    *floor-rounded* per-CTU upper bound so the result remains an
    #    integer and the (★) constraint still holds: if the continuous
    #    upper bound is -0.93, the integer cap is floor(-0.93) = -1
    #    (i.e. δ ≤ -1 ⇒ definitely δ ≤ -0.93). Lower clip stays at the
    #    VVC legal range so we don't accidentally lose bits.
    upper_int_per_ctu = np.floor(upper_per_ctu).astype(np.float64)
    delta_int = np.minimum(
        np.maximum(np.rint(delta_continuous), float(delta_min_clip)),
        upper_int_per_ctu,
    ).astype(np.int32)

    # 5) Rate-neutrality monitors.
    return OGAPlusOutput(
        delta_continuous=delta_continuous,
        delta_int=delta_int,
        delta_a_plus_raw=delta_a,
        rate_ratio_continuous=float(rate_neutral_residual(delta_continuous, K)),
        rate_ratio_rounded=float(rate_neutral_residual(delta_int.astype(np.float64), K)),
    )


# ---------------------------------------------------------------------------
# Diagnostics (§9 of the user spec)
# ---------------------------------------------------------------------------

@dataclass
class OGDiagnostics:
    """Per-frame OG-IPF diagnostics, see PROJECT_STATE §7.20.

    All numbers are scalars (one frame). The orchestrator aggregates
    them into a sequence-level JSON + summary CSV.
    """

    n_ctus_total:                   int
    pct_object_overlap:             float
    pct_context:                    float
    pct_far_background:             float
    mean_delta_object:              float
    mean_delta_context:             float
    mean_delta_far_background:      float
    saturation_lower:               float    # |δ - δ_min_clip| < 0.5 fraction
    saturation_upper:               float    # |δ - δ_max_clip| < 0.5 fraction
    rate_ratio_continuous:          float
    rate_ratio_rounded:             float
    object_protection_violation:    float    # G_c > 0.5  AND  δ_c >= 0
    g_threshold_object:             float    # =0.50, kept here for traceability
    g_threshold_context:            float    # =0.05, kept here for traceability


def compute_og_diagnostics(
    delta_continuous: np.ndarray,
    delta_int:        np.ndarray,
    G_max:            np.ndarray,
    K:                np.ndarray,
    *,
    g_object:  float = 0.5,
    g_context: float = 0.05,
    delta_min_clip: float = -8.0,
    delta_max_clip: float = +4.0,
    rate_ratio_continuous: Optional[float] = None,
    rate_ratio_rounded:    Optional[float] = None,
) -> OGDiagnostics:
    """Compute the §9 diagnostics for one frame.

    The function uses the *integer* delta map for the per-region
    statistics (because that is what actually goes to the encoder),
    but accepts the continuous map only to recompute the rate-ratio
    if the caller did not pre-compute it.
    """
    delta_int = np.asarray(delta_int, dtype=np.float64)
    g = np.clip(np.asarray(G_max, dtype=np.float64), 0.0, 1.0)
    n_total = int(delta_int.size)

    # Region masks. ``object`` = strongly-overlapping CTUs (G > g_object).
    # ``context`` = mildly-overlapping or near-bbox CTUs (g_context < G ≤ g_object).
    # ``far`` = essentially zero overlap and far from any bbox (G ≤ g_context).
    mask_object  = g > g_object
    mask_context = (g > g_context) & (g <= g_object)
    mask_far     = g <= g_context

    def _mean_delta(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(delta_int[mask]))

    sat_lo = float(np.mean(np.abs(delta_int - delta_min_clip) < 0.5))
    sat_hi = float(np.mean(np.abs(delta_int - delta_max_clip) < 0.5))

    if rate_ratio_continuous is None:
        rate_ratio_continuous = float(rate_neutral_residual(
            np.asarray(delta_continuous, dtype=np.float64), K))
    if rate_ratio_rounded is None:
        rate_ratio_rounded = float(rate_neutral_residual(delta_int, K))

    # Object-protection violation: any object CTU that ends up at
    # δ_c ≥ 0 (i.e. equal or worse quality than the anchor) breaks
    # the (★) guarantee. The min-protection step forbids this for
    # G_c · η > 0; if you see > 0 here, look for clipping at +bg_bound
    # *fighting* the constraint.
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
        rate_ratio_continuous=rate_ratio_continuous,
        rate_ratio_rounded=rate_ratio_rounded,
        object_protection_violation=violation,
        g_threshold_object=g_object,
        g_threshold_context=g_context,
    )


__all__ = [
    "MinProtectionConfig",
    "OGAPlusOutput",
    "OGDiagnostics",
    "min_protection_floor",
    "apply_min_object_protection",
    "compute_og_a_plus_delta",
    "end_to_end_og_a_plus",
    "compute_og_diagnostics",
]
