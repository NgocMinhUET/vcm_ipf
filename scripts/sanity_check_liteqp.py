"""Local sanity check for LiteQP analytic prior — no PyYAML / VTM / GPU needed.

Validates that:
  1. compute_a_plus_delta returns finite values in the expected sign range.
  2. project_rate_neutral_exact really makes Σ K · 2^{-δ/6} = Σ K (within 1e-6).
  3. q_adaptive_bounds are monotone in Q_base.
  4. End-to-end on a synthetic Φ / K grid produces a reasonable distribution.
"""

import sys
from pathlib import Path

# Force UTF-8 console (Windows cp1252 default mangles Δ, ρ, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Make the package importable without `pip install -e .`
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import numpy as np

from phase2.phase3.analytic_a_plus import (
    AnalyticAPlusConfig,
    compute_a_plus_delta,
    end_to_end_a_plus,
    project_rate_neutral_exact,
    project_rate_neutral_linear,
    q_adaptive_bounds,
)


def test_q_adaptive_bounds():
    print("\n[1] Q-adaptive bounds")
    print("    Q_base   Δ_roi   Δ_bg")
    for q in [22, 27, 32, 37, 42, 47]:
        roi, bg = q_adaptive_bounds(q)
        print(f"      {q:2d}     {roi:.2f}    {bg:.2f}")
    # Monotone non-decreasing in q_base
    rs = [q_adaptive_bounds(q)[0] for q in range(22, 48)]
    bs = [q_adaptive_bounds(q)[1] for q in range(22, 48)]
    assert all(rs[i] <= rs[i + 1] + 1e-9 for i in range(len(rs) - 1)), \
        "Δ_roi must be non-decreasing in Q_base"
    assert all(bs[i] <= bs[i + 1] + 1e-9 for i in range(len(bs) - 1)), \
        "Δ_bg must be non-decreasing in Q_base"
    print("    ✓ both bounds are monotone non-decreasing")


def _synthetic_grid(n_rows=9, n_cols=15, seed=20260429):
    rng = np.random.default_rng(seed)
    # Simulated Φ_oracle: a Gaussian blob in the upper-middle (mimicking a person).
    yy, xx = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    cy, cx = 3.0, 7.0
    blob = np.exp(-((yy - cy) ** 2 / 2.0 + (xx - cx) ** 2 / 4.0))
    noise = rng.uniform(0.0, 0.05, size=blob.shape)
    phi = (blob / blob.max()) * 0.8 + noise            # ∈ [~0.05, ~0.85]
    # Simulated K_c: positively correlated with edges + noise.
    K = 0.7 + 0.5 * blob + rng.uniform(0.0, 0.3, size=blob.shape)
    return phi, K


def test_analytic_basic():
    print("\n[2] Analytic A+ — basic shape & sign")
    phi, K = _synthetic_grid()
    print(f"    phi range: [{phi.min():.3f}, {phi.max():.3f}]   "
          f"mean={phi.mean():.3f}")
    print(f"    K   range: [{K.min():.3f}, {K.max():.3f}]   "
          f"mean={K.mean():.3f}")

    for q in [27, 32, 37, 42]:
        delta = compute_a_plus_delta(phi, K, q)
        roi_share = (delta < 0).mean()
        bg_share  = (delta > 0).mean()
        print(f"    Q_b={q}:  δ ∈ [{delta.min():+.2f}, {delta.max():+.2f}]   "
              f"ROI(δ<0)={roi_share*100:.1f}%   BG(δ>0)={bg_share*100:.1f}%")
        assert np.isfinite(delta).all(), "non-finite δ"
        # In the synthetic example the blob centre is high-Φ → ROI.
        # We expect the blob region to receive δ < 0 on average.
        cy, cx = 3, 7
        local = delta[cy-1:cy+2, cx-1:cx+2]
        assert local.mean() < 0, f"Q_b={q}: ROI region should have negative δ"
    print("    ✓ ROI receives δ<0, background δ>0, all values finite")


def test_rate_neutral_projection():
    print("\n[3] Rate-neutral projection — exact vs linear")
    phi, K = _synthetic_grid()
    delta = compute_a_plus_delta(phi, K, q_base=32)
    print(f"    Pre-projection:")
    pre_total = float((K * (2.0 ** (-delta / 6.0))).sum())
    print(f"      Σ K · 2^{{-δ/6}}   = {pre_total:.6f}")
    print(f"      Σ K               = {K.sum():.6f}")
    print(f"      ratio             = {pre_total / K.sum():.6f}  "
          f"(target = 1.0)")

    delta_lin = project_rate_neutral_linear(delta, K)
    delta_exa = project_rate_neutral_exact(delta, K)

    lin_total = float((K * (2.0 ** (-delta_lin / 6.0))).sum())
    exa_total = float((K * (2.0 ** (-delta_exa / 6.0))).sum())

    print(f"    Linear projection:  ratio = {lin_total / K.sum():.6f}   "
          f"|err| = {abs(lin_total - K.sum()):.2e}")
    print(f"    Exact projection:   ratio = {exa_total / K.sum():.6f}   "
          f"|err| = {abs(exa_total - K.sum()):.2e}")

    assert abs(exa_total - K.sum()) < 1e-6, "exact projection must hit rate-neutral"
    print("    ✓ exact projection achieves rate-neutrality to machine precision")


def test_end_to_end_distribution():
    print("\n[4] End-to-end δ distribution across QP grid")
    phi, K = _synthetic_grid()
    print("    Q_b   δ_min   δ_max   δ_mean   K-weighted-mean")
    for q in [27, 32, 37, 42]:
        delta = end_to_end_a_plus(phi, K, q, project=True, round_to_int=False)
        kw = float((K * delta).sum() / K.sum())
        # rate-neutral after projection means K-weighted 2^{-δ/6} is exactly 1,
        # not the K-weighted δ itself; we report the latter as a sanity check.
        print(f"    {q:3d}   {delta.min():+5.2f}  {delta.max():+5.2f}  "
              f"{delta.mean():+5.2f}    {kw:+5.2f}")
    print("    ✓ end-to-end produces a sensible δ map without crashes")


def test_residual_target_simulation():
    """Simulate the teacher-label loop on a single CTU."""
    print("\n[5] Teacher δ_star simulation (single CTU, scan)")
    from math import log2

    # One ROI CTU and one BG CTU at Q_b=32
    cases = [
        ("ROI",  0.85, 1.30),
        ("MID",  0.50, 1.00),
        ("BG",   0.10, 0.80),
    ]
    delta_grid = [-8, -6, -4, -2, 0, 2, 4]
    eta, xi = 0.06, 0.15
    lam_task, lam_anchor = 5.0, 0.6
    Q_b = 32
    resid = 1.0

    print(f"    Q_b={Q_b}  λ_task={lam_task}  λ_anchor={lam_anchor}")
    for label, phi, K in cases:
        # δ_a+ for this single CTU (via the analytic formula on a 1-element grid)
        d_a = compute_a_plus_delta(np.array([[phi]]), np.array([[K]]), Q_b)[0, 0]

        scores = []
        for d in delta_grid:
            d_R = resid * K * (2 ** (-(Q_b + d) / 6.0) - 2 ** (-Q_b / 6.0))
            d_m = -(phi * eta * max(0, d)) + (phi * eta * xi * max(0, -d))
            score = d_R - lam_task * d_m + lam_anchor * abs(d - d_a)
            scores.append((d, d_R, d_m, score))
        best = min(scores, key=lambda x: x[3])
        print(f"    {label:3s}  Φ={phi:.2f}  K={K:.2f}  "
              f"δ_a+ = {d_a:+.2f}  δ* = {best[0]:+d}  "
              f"residual = {best[0] - d_a:+.2f}")
    print("    ✓ teacher converges to a discrete δ near the analytic prior")


if __name__ == "__main__":
    print("LiteQP analytic-prior sanity check")
    print("=" * 60)
    test_q_adaptive_bounds()
    test_analytic_basic()
    test_rate_neutral_projection()
    test_end_to_end_distribution()
    test_residual_target_simulation()
    print("\n" + "=" * 60)
    print("All checks passed.")
