# D4 — Empirical η calibration: design spec

## Purpose

PROJECT_AUDIT.md §1.4 identifies the synthetic ΔmAP oracle as the most
critical scientific debt. The oracle hard-codes:

```
ΔmAP(c, δ) = -φ_c · η · max(0, δ) + φ_c · η · ξ · max(0, -δ)
        with η = 0.06,  ξ = 0.15
```

D4 measures the *real* relationship between (φ, δ) and observed ΔmAP via
controlled perturbation experiments on a single sequence, so we can either

* validate the formula (η_emp ≈ 0.06, linear in δ, scaled by φ), or
* re-calibrate the constants, or
* falsify the linear-in-δ assumption and motivate a richer model.

## Hypothesis under test

H_0 (current oracle): for every CTU c and every QP base Q_b,

   E[ΔmAP_c(δ)] = -φ_c · 0.06 · max(0, δ) + φ_c · 0.009 · max(0, -δ)

H_1 (alternatives we want to discriminate among):

   H_1a: η depends on Q_b (e.g. larger at high QP).
   H_1b: ΔmAP is non-linear in δ (saturating).
   H_1c: ΔmAP is not proportional to φ (e.g. only top-decile φ matters).
   H_1d: ΔmAP is dominated by frame-level effects, not per-CTU.

## Experimental design

### Variables

* **Sequence**: MOT17-04-DPM (50 frames). Chosen because it has the most
  detected boxes per frame (rich task signal) and the most stable M0 mAP.
* **Base QP**: {27, 32, 37, 42} — same grid as pilot evaluation.
* **Pattern**: 6 controlled δ maps per QP. All maps are exactly rate-neutral
  by construction (Σ K_c · 2^{-δ_c/6} = Σ K_c) so any ΔmAP we measure is
  attributable to *redistribution*, not net rate change.

### Patterns (all rate-neutral)

| ID  | Description | Top-decile φ CTUs δ | Bottom-decile φ CTUs δ | Mid CTUs δ |
|-----|-------------|---------------------|------------------------|------------|
| P0  | M0 (control) | 0 | 0 | 0 |
| P1  | small ROI tightening | -2 | +2 (proj.) | 0 |
| P2  | medium ROI tightening | -4 | +4 (proj.) | 0 |
| P3  | large ROI tightening | -6 | +6 (proj.) | 0 |
| P4  | inverted (ablation) | +2 (give bits *away* from ROI) | -2 (proj.) | 0 |
| P5  | random sign-balanced | random ±2 | random ±2 | random ±2 |

After choosing the integer δ in top/bottom deciles, the remaining CTUs are
shifted by a global scalar `s` such that the rate-neutral condition holds
exactly (use `analytic_a_plus.project_rate_neutral_clipped_exact`).

P4 is critical: if ΔmAP truly tracks φ·δ, P4 should produce a *negative*
ΔmAP with magnitude similar to P1's positive. Asymmetry quantifies ξ.

### Measurements

For each (Q_b, pattern):

1. Encode the 50-frame YUV with the synthesised δ map.
2. Decode + extract frames.
3. Run YOLOv8 with conf=0.001 on decoded vs uncompressed reference frames.
4. Compute true COCO mAP_50 via `phase2.diagnostics.d1_true_map`.
5. Record per-frame TP/FP/FN at IoU=0.5 (for D2-style stat tests).
6. Record actual rate in kbps (should match M0 within 2%).
7. Compute the predicted ΔmAP from the synthetic oracle for the same δ map.

### Output table

For each (Q_b, pattern):

| Q_b | Pattern | Σ K δ / Σ K | rate kbps | ΔmAP_real | ΔmAP_pred | Δrate% |
|-----|---------|-------------|-----------|-----------|-----------|--------|

### Fitting η_emp

Across all (Q_b, pattern) cells with ΔmAP measurements, fit:

```
ΔmAP_real ≈ η_emp · Σ_c φ_c · g(δ_c)   where g(δ) = -max(0,δ) + ξ·max(0,-δ)
```

Solve via least squares for (η_emp, ξ_emp). Report 95% bootstrap CI on η_emp.

If R² < 0.5 the linear-in-δ assumption is rejected; fit a quadratic in δ as
the nearest-neighbour model.

If η_emp is not constant across Q_b (e.g. fitted separately per Q_b), report
η_emp(Q_b) and update the oracle to be Q-dependent.

## Compute estimate

* 4 QPs × 6 patterns = 24 encodes per sequence
* Each encode: 50 frames at QP 27–42, ≈ 25 min on the server
* Total: ≈ 10 hours
* Detection: D1 takes ≈ 1 minute per cell × 24 = 25 min on GPU

Total wall-clock ≈ 11 hours, single sequence.

## Pre-registered decision rules

After D4 finishes:

1. **If η_emp ≈ 0.06 ± 0.02 and R² > 0.7 across all QPs** → the synthetic
   oracle is empirically validated. Continue with current pipeline; the
   instability we observed is finite-sample noise (D2 verdict). Recommendation:
   honest paper with modest claim + larger-N future work.

2. **If η_emp ∈ [0.02, 0.04] (formula overestimates by 1.5–3×) with R² > 0.5**
   → re-build the LiteQP oracle with η_emp, retrain CNN, run pilot_v10.
   Predicted effect: smaller |δ| at the optimum, less aggressive maps,
   probably more stable across pilots.

3. **If R² < 0.5** → linear ΔmAP assumption is wrong. Pause CNN training.
   Fit a richer model (quadratic, sigmoidal, or per-CTU look-up table).

4. **If P4 (inverted) does NOT produce negative ΔmAP** → φ·δ ranking is
   not the right importance signal. Falsifies the entire IPF→δQP framework.
   Pivot to a different importance representation (e.g. per-detection-box
   QP rather than per-CTU).

## Implementation checklist

This spec will be implemented as:

* `phase2/diagnostics/d4_build_patterns.py`: synthesise the 6 δ maps per QP
  using the existing `analytic_a_plus.project_rate_neutral_clipped_exact`.
* `phase2/configs/d4_eta_calibration.yaml`: orchestrates the 24 encodes.
* `phase2/diagnostics/d4_fit_eta.py`: post-hoc fit of η_emp from D1 outputs.

These are TODOs deferred until Phase 1 (D1, D2, D3) results land — they may
overturn assumptions in this design.

## Risks / caveats

* **Per-frame noise**: a single 50-frame ΔmAP estimate has σ ≈ 0.01–0.02
  (D2 will confirm). Six patterns × 4 QPs = 24 data points may not be
  enough to fit (η, ξ) with tight CI. If so, increase patterns to 10 or
  use more sequences.
* **YOLOv8 sensitivity**: if the detector is largely insensitive to the
  perturbations we apply, ΔmAP will be near zero for all patterns. This
  itself is informative — it means the detector is the bottleneck, not
  the encoding.
* **Rate-neutral projection precision**: post-VVC-encode rate may drift
  ±2–3% from target. Filter cells with > 5% rate drift before fitting.
