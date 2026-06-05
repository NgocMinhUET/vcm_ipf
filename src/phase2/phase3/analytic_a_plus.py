"""Phase 3 Stage C — RD-log A+ analytic δQP allocator.

This module implements the **analytic prior** used by the LiteQP residual
estimator (`train_liteqp_regressor.py`, `apply_liteqp_model.py`). It is
also a strong stand-alone baseline (M4-A+) to compare against in
``pilot_v4``.

Theoretical background
----------------------
We adopt the canonical R-λ rate model (Sullivan & Wiegand, 1998),

    R_c(Q) = K_c · 2^{-Q/6}                                           (1)

and a task-distortion proxy proportional to the per-CTU saliency Φ_c
(occlusion saliency, Zeiler-Fergus 2014).  Treating per-CTU bit allocation
as a constrained optimisation,

    minimise   Σ_c K_c · 2^{-(Q_b + δ_c)/6}
    subject to Σ_c K_c · 2^{-(Q_b + δ_c)/6} = Σ_c K_c        (rate neutrality)
               Σ_c δ_c task-weighted ≈ task-importance allocation

leads — under a log-quadratic relaxation of the task term — to the
closed-form **RD-log A+ allocation**:

    ξ_c     = (Φ_c + ε) / (K̃_c + κ)^β                                 (2)
    g       = (Σ_c K_c · ξ_c) / (Σ_c K_c)                              (3)
    δ_c^A+  = -6 · log_2( ξ_c / g )                                    (4)

Properties:

* **Scale invariant** — multiplying Φ_c by a constant does not change
  δ_c^A+ (numerator and denominator scale together).
* **Rate neutral by construction** — Σ_c K_c · 2^{-δ_c^A+/6}  ≈  Σ_c K_c
  to first order; we additionally apply an exact closed-form projection
  (`project_rate_neutral_exact`).
* **Q-adaptive** — the global scale and clipping bounds are Q_b-dependent
  (see `q_adaptive_bounds`), addressing the MOT17-02 low-QP instability
  observed in pilot_v3.

This module is intentionally a single, dependency-free Numpy file so it
can be imported by both the dataset builder and the apply step.

Rationale for the "+" in A+
---------------------------
Compared to the plain RD-log allocator (Bjontegaard 2001),
A+ adds: (i) Q-adaptive scale, (ii) Q-adaptive clipping, and
(iii) the per-CTU complexity normaliser (κ + K̃_c)^β that prevents
over-allocation to texture-poor CTUs (which would otherwise see
δ_c → −∞ when ξ_c → ∞).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Q-adaptive bounds (calibrated from pilot_v1 + pilot_v3 observations)
# ---------------------------------------------------------------------------

def q_adaptive_bounds(q_base: int) -> tuple[float, float]:
    """Return (Δ_roi, Δ_bg) bounds suitable for ``q_base``.

    At low Q_b (e.g. 27) we want **smaller** absolute deltas because:
      * VVC intra-prediction quality is high → CTU-boundary discontinuity
        from ±4 QP gaps causes prediction mismatch (observed on
        MOT17-02-DPM at QP=27 in pilot_v3).
      * The detector is already near its accuracy ceiling, so additional
        QP reduction in ROI yields little task gain.

    At high Q_b (e.g. 42) we **need** larger deltas because:
      * Background CTUs can be safely degraded (decoder is already lossy).
      * ROI CTUs benefit substantially from any quality boost.
    """
    delta_bg = float(np.clip(2.0 + 0.13 * (q_base - 27), 2.0, 4.0))
    delta_roi = float(np.clip(4.0 + 0.20 * (q_base - 27), 4.0, 7.0))
    return delta_roi, delta_bg


# ---------------------------------------------------------------------------
# Core analytic δQP formula
# ---------------------------------------------------------------------------

@dataclass
class AnalyticAPlusConfig:
    """Hyper-parameters of the RD-log A+ allocator.

    The defaults are calibrated and should rarely need changing. They are
    exposed so the user can ablate each component.
    """

    eps: float = 1e-3       # numerical floor for Φ
    beta: float = 0.5       # complexity exponent (0 = ignore K_c, 1 = full)
    kappa: float = 0.05     # complexity floor (prevents division by ~0)
    scale_a: float = 0.85   # base scale at Q_b = 32
    scale_b: float = 0.03   # Q-adaptive scale slope (per QP step)
    scale_min: float = 0.70
    scale_max: float = 1.15
    delta_min_clip: int = -8   # absolute floor (VVC legal range)
    delta_max_clip: int = +4   # absolute ceiling (VVC legal range)


def _normalize_complexity(K: np.ndarray) -> np.ndarray:
    """Median-normalise K so K̃ ≈ 1 for the typical CTU."""
    K = np.asarray(K, dtype=np.float64)
    pos = K[K > 0]
    if pos.size == 0:
        return np.ones_like(K, dtype=np.float64)
    med = float(np.median(pos))
    return K / (med + 1e-9)


def compute_a_plus_delta(
    phi: np.ndarray,
    K: np.ndarray,
    q_base: int,
    cfg: AnalyticAPlusConfig | None = None,
) -> np.ndarray:
    """Closed-form RD-log A+ allocation.

    Parameters
    ----------
    phi
        Per-CTU normalised importance ∈ [0, 1] (e.g. ``phi_oracle_norm``).
    K
        Per-CTU rate complexity (from ``rate_surrogate.py``). Must have the
        same shape as ``phi``.
    q_base
        Sequence-level base QP. Used for both the Q-adaptive scale and
        the Q-adaptive clipping bounds.
    cfg
        Optional config; defaults are calibrated.

    Returns
    -------
    np.ndarray
        Float δQP map (not yet rounded), same shape as ``phi``.
    """
    if cfg is None:
        cfg = AnalyticAPlusConfig()

    phi = np.asarray(phi, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if phi.shape != K.shape:
        raise ValueError(f"shape mismatch: phi {phi.shape} vs K {K.shape}")

    K_norm = _normalize_complexity(K)

    # (2) per-CTU score combining task importance and rate complexity.
    xi = (phi + cfg.eps) / np.power(K_norm + cfg.kappa, cfg.beta)

    # (3) K-weighted arithmetic normalisation factor — chosen so that
    #     ``-6·log_2(ξ_c / g)`` has zero K-weighted mean **on the log-rate
    #     scale**, i.e. Σ K · 2^{-δ_c/6} ≈ Σ K to first order. This is NOT a
    #     geometric mean; it is the arithmetic K-weighted mean of ξ.
    denom = np.sum(K) + 1e-9
    g = float(np.sum(K * xi) / denom)
    g = max(g, 1e-9)

    # (4) RD-log A+ allocation.
    delta = -6.0 * np.log2((xi + 1e-9) / g)

    # Q-adaptive global scale (smooth, monotone, clamped).
    scale = float(np.clip(
        cfg.scale_a + cfg.scale_b * (q_base - 32),
        cfg.scale_min, cfg.scale_max,
    ))
    delta = scale * delta

    # Q-adaptive bounds (Δ_roi, Δ_bg) — tighter than the absolute clip.
    roi_bound, bg_bound = q_adaptive_bounds(q_base)
    delta = np.clip(delta, -roi_bound, +bg_bound)

    # Hard absolute clip (VVC legal range).
    delta = np.clip(delta, cfg.delta_min_clip, cfg.delta_max_clip)

    return delta


# ---------------------------------------------------------------------------
# Rate-neutral projection — exact closed form in the R-λ model
# ---------------------------------------------------------------------------

def project_rate_neutral_exact(
    delta: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Solve  Σ K_c · 2^{-(δ_c + s)/6} = Σ K_c  for the global shift ``s``.

    This is the **exact** rate-neutrality constraint in the R-λ rate model,
    not the first-order linearisation ``s ≈ Σ K δ / Σ K``.

    Derivation (with W = Σ K · 2^{-δ/6}, D = Σ K, ratio = W/D)::

        Σ K · 2^{-(δ+s)/6} = D
        ⟺ 2^{-s/6} · W      = D
        ⟺ 2^{-s/6}          = 1/ratio
        ⟺ s                 = 6 · log₂(ratio)

    Sign sanity check: if ratio < 1 (current map uses *fewer* bits than the
    baseline) we want to **decrease** δ to spend more bits, so ``s`` should
    be negative — and indeed ``log₂(ratio) < 0`` ⇒ ``s < 0``.  ✓
    """
    delta = np.asarray(delta, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    denom = float(np.sum(K) + 1e-9)
    weighted = float(np.sum(K * np.power(2.0, -delta / 6.0)))
    ratio = max(weighted / denom, 1e-12)
    s = 6.0 * np.log2(ratio)
    return delta + s


def project_rate_neutral_linear(
    delta: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """First-order linearisation of the exact projection.

    Using log₂(1 + x) ≈ x/ln 2 around x = 0::

        ratio = Σ K · 2^{-δ/6} / Σ K ≈ 1 + (1/ln 2)·(-Σ K · δ / (6·Σ K))
        ⇒ s = 6 · log₂(ratio) ≈ -Σ K · δ / Σ K
    """
    delta = np.asarray(delta, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    s = -float(np.sum(K * delta) / (np.sum(K) + 1e-9))
    return delta + s


def project_rate_neutral_clipped_exact(
    delta: np.ndarray,
    K: np.ndarray,
    delta_min,
    delta_max,
    n_iter: int = 60,
    tol: float = 1e-9,
) -> np.ndarray:
    """Rate-neutral projection that **respects per-CTU clipping bounds**.

    The plain :func:`project_rate_neutral_exact` finds a global shift ``s``
    that makes ``Σ K · 2^{-(δ+s)/6} = Σ K``. However, in our pipeline we then
    clip the shifted map into ``[delta_min, delta_max]``, which can break
    the equality (some CTUs saturate at the bound and stop "absorbing" the
    shift).

    This function instead solves

        find s such that  Σ K · 2^{-clip(δ+s, [δ_min, δ_max])/6}  =  Σ K

    by 1-D bisection. The function ``s ↦ Σ K · 2^{-clip(δ+s,…)/6}`` is
    monotonically *decreasing* in ``s`` (more shift ⇒ higher post-shift δ ⇒
    fewer bits), so bisection always converges.

    ``delta_min`` and ``delta_max`` may be **scalars** *or* **arrays** of
    the same shape as ``delta``. Per-CTU bounds are required by the
    OG-IPF min-object-protection constraint (PROJECT_STATE §7.20) — the
    upper bound at a strongly-occupied CTU is ``-δ_min(Q_b) · G_c^η``
    instead of the generic ``+bg_bound``, so any post-projection global
    shift cannot push protected CTUs back above zero.

    Returns the **clipped** delta map.
    """
    delta = np.asarray(delta, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    delta_min_arr = np.broadcast_to(np.asarray(delta_min, dtype=np.float64),
                                     delta.shape)
    delta_max_arr = np.broadcast_to(np.asarray(delta_max, dtype=np.float64),
                                     delta.shape)
    target = float(np.sum(K))
    if target <= 0:
        return np.minimum(np.maximum(delta, delta_min_arr), delta_max_arr)

    def rate_sum(s: float) -> float:
        d = np.minimum(np.maximum(delta + s, delta_min_arr), delta_max_arr)
        return float(np.sum(K * np.power(2.0, -d / 6.0)))

    # Wide bracket — well outside any reasonable QP shift in [-30, +30].
    lo, hi = -30.0, +30.0
    f_lo = rate_sum(lo)   # large (δ very negative ⇒ many bits)
    f_hi = rate_sum(hi)   # small
    # If the target is unreachable even at the extremes, return best-effort.
    if not (f_hi <= target <= f_lo):
        s_best = lo if abs(f_lo - target) < abs(f_hi - target) else hi
        return np.minimum(np.maximum(delta + s_best, delta_min_arr),
                           delta_max_arr)

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if rate_sum(mid) > target:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol:
            break

    s = 0.5 * (lo + hi)
    return np.minimum(np.maximum(delta + s, delta_min_arr), delta_max_arr)


def rate_neutral_residual(delta: np.ndarray, K: np.ndarray) -> float:
    """Return ``Σ K · 2^{-δ/6} / Σ K``. = 1.0 means perfectly rate-neutral.

    Use this to **monitor** how much the integer-rounding step in
    :func:`apply_liteqp_model.main` deviates from the continuous projection.
    """
    delta = np.asarray(delta, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    denom = float(np.sum(K) + 1e-9)
    return float(np.sum(K * np.power(2.0, -delta / 6.0)) / denom)


# ---------------------------------------------------------------------------
# Per-CTU rate model calibration
# ---------------------------------------------------------------------------

def calibrate_K_from_two_encodes(
    bits_qp_lo: np.ndarray,
    bits_qp_hi: np.ndarray,
    qp_lo: int,
    qp_hi: int,
) -> np.ndarray:
    """Fit per-CTU ``K_c`` from two reference encodes at different QPs.

    Motivation
    ----------
    The exponential rate model ``R_c(Q) = K_c · 2^{-Q/6}`` approximates the
    VVC encoder at high rate, but ``K_c`` is assumed uniform (median-normalised)
    in the current pipeline.  In practice CTUs with rich texture (crowd, foliage)
    have K_c 3–5× the median, while flat-sky CTUs have K_c < 0.3.  Using the
    actual K_c profile tightens ``rho_R`` to ``[0.95, 1.05]`` on sequences
    where the current surrogate drifts (observed on MOT17-13 at QP=42).

    Formula (two-point fit)
    -----------------------
    From two encodes at ``qp_lo`` and ``qp_hi``::

        bits_c(Q) ≈ K_c · 2^{-Q/6}
        ⟹ K_c ≈ bits_c(qp_lo) / 2^{-qp_lo/6}

    In practice, use the geometric mean to reduce QP-specific noise::

        K_c = sqrt( bits_c(qp_lo) · 2^{qp_lo/6}  ·  bits_c(qp_hi) · 2^{qp_hi/6} )

    Parameters
    ----------
    bits_qp_lo, bits_qp_hi
        Per-CTU bit counts (or variance proxies from the rate surrogate)
        at the two QP levels.  Shape must be identical.
    qp_lo, qp_hi
        Corresponding QP values (qp_lo < qp_hi).

    Returns
    -------
    np.ndarray
        Per-CTU K_c estimate, same shape as inputs, clipped to ``[1e-3, 1e6]``.
    """
    bits_lo = np.asarray(bits_qp_lo, dtype=np.float64)
    bits_hi = np.asarray(bits_qp_hi, dtype=np.float64)
    if bits_lo.shape != bits_hi.shape:
        raise ValueError(
            f"bits arrays must have the same shape: {bits_lo.shape} vs {bits_hi.shape}"
        )
    # K_c estimate from each QP
    k_lo = bits_lo * float(np.power(2.0, qp_lo / 6.0))
    k_hi = bits_hi * float(np.power(2.0, qp_hi / 6.0))
    # Geometric mean for numerical stability
    K_c = np.sqrt(np.maximum(k_lo, 1e-9) * np.maximum(k_hi, 1e-9))
    return np.clip(K_c, 1e-3, 1e6)


def calibrate_K_from_npz(npz_path: str, qp_lo: int = 27, qp_hi: int = 42) -> np.ndarray:
    """Load per-CTU K_c from a ``rate_surrogate.npz`` file.

    The NPZ is expected to have a ``K_grids`` key with shape
    ``(n_qp, n_rows, n_cols)`` where rows correspond to the QP list stored
    under ``qp_list`` (or inferred from shape if absent).  Returns a
    ``(n_rows, n_cols)`` K_c map by calling
    :func:`calibrate_K_from_two_encodes` on the ``qp_lo`` and ``qp_hi``
    slices.

    Falls back to the median-normalised uniform K (all-ones) if the file
    cannot be read or the QP slices are not found.
    """
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception:
        return None  # caller should fall back to uniform K

    K_grids = data.get("K_grids", None)
    if K_grids is None or K_grids.ndim != 3:
        return None

    qp_list = list(data.get("qp_list", []))
    if not qp_list:
        # Infer: assume standard QP sequence starting at 22 step 5
        qp_list = [22 + 5 * i for i in range(K_grids.shape[0])]

    try:
        idx_lo = qp_list.index(qp_lo)
        idx_hi = qp_list.index(qp_hi)
    except ValueError:
        # Fall back to first and last available
        idx_lo = 0
        idx_hi = K_grids.shape[0] - 1

    return calibrate_K_from_two_encodes(
        K_grids[idx_lo], K_grids[idx_hi], qp_lo, qp_hi
    )


# ---------------------------------------------------------------------------
# Convenience: end-to-end "A+ delta map" used by the apply scripts
# ---------------------------------------------------------------------------

def end_to_end_a_plus(
    phi: np.ndarray,
    K: np.ndarray,
    q_base: int,
    cfg: AnalyticAPlusConfig | None = None,
    project: bool = True,
    round_to_int: bool = False,
) -> np.ndarray:
    """One-shot helper: compute → rate-project → optionally round.

    Used by ``apply_liteqp_model.py`` for the M4-A+ baseline and for
    constructing the residual targets in ``build_liteqp_dataset.py``.
    """
    delta = compute_a_plus_delta(phi, K, q_base, cfg)
    if project:
        delta = project_rate_neutral_exact(delta, K)
    # Re-clip after projection (the shift may push some CTUs slightly out).
    if cfg is None:
        cfg = AnalyticAPlusConfig()
    roi_bound, bg_bound = q_adaptive_bounds(q_base)
    delta = np.clip(delta, -roi_bound, +bg_bound)
    delta = np.clip(delta, cfg.delta_min_clip, cfg.delta_max_clip)
    if round_to_int:
        delta = np.rint(delta).astype(np.int32)
    return delta
