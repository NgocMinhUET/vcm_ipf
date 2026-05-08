# PROJECT AUDIT — Scientific Critique (May 2026)

> **Purpose**: comprehensive scientific audit of the IPF→δQP framework after
> pilot_v9 instability. Documents 6 critical findings, their evidence, and the
> diagnostic plan that must run *before* any further pilot encodes.
>
> **Trigger**: user feedback dated 2026-05-08 — *"results are not convincing,
> unstable, this is only against M0 baseline. Has the IPF construction and QP
> mapping been scientific and convincing? Does the calculation actually help
> AI tasks?"*
>
> **Status**: under-active — diagnostic battery in `phase2/diagnostics/` runs
> in parallel with `pilot_v9b`. Verdict written here will inform whether the
> framework is salvageable or needs reformulation.

---

## 0. TL;DR

After 9 pilots and many architectural pivots (parametric → oracle-direct →
LiteQP-MLP → LiteQP-CNN → Q-aware clip → QP=27-only) the project has not
produced a stable improvement over the M0 baseline. Average BD-Rate-Task
fluctuates ±10pp across pilots that differ only in inference-time tweaks.
The user's intuition is correct: this is **not** publishable in current form.

Root cause is **not** the choice of MLP vs CNN, nor the clipping schedule.
Root cause is six largely-unexamined scientific debts at the foundation:

| # | Debt | Severity | Fix path |
|---|------|----------|----------|
| 1 | φ_oracle (saliency) — task-driven occlusion | OK | none needed |
| 2 | K_c (rate sensitivity) — per-frame regressed, per-CTU formula | Medium | per-CTU validation against VTM stats |
| 3 | Hard-coded "6" in A+ formula — borrowed from R-PSNR theory | Medium | calibrate vs task |
| 4 | Synthetic ΔmAP oracle (η=0.06 hard-coded) — never validated | **Critical** | Diagnostic D4 |
| 5 | "mAP" computed as `precision × recall` at one threshold — not real AP | **Critical** | Diagnostic D1 |
| 6 | n=3 sequences × 50 frames + GT = M0 detections | High | larger N + real GT |

Until D1 and D4 land we have **no scientific basis** to claim any pilot
"beats" or "loses to" any other.

---

## 1. The full audit

### 1.1 Φ_oracle (task-driven saliency) — *OK*

`occlusion_saliency.py`:
* Method = patch-occlusion + detector-confidence drop (Zeiler & Fergus 2014)
* Per-CTU score = Σ_{boxes overlapping CTU} max(0, conf_clean - conf_blurred)
* Diagnostic ρ(Φ_IPF, Φ_oracle) ≥ 0.30 enforced; warning otherwise
* **Task-driven** (uses YOLOv8 confidence, not edge density)

This is the one piece that is scientifically defensible. No action.

### 1.2 K_c rate surrogate — *medium issue*

`rate_surrogate.py`:
* Formula `K_c = α · σ_Y(c)^ρ · (1 + |MV(c)|)^ν` (Sullivan-Wiegand 1998 +
  Lin-Chao TCSVT 2016)
* Calibrated by least-squares against per-frame total bits at 4 QPs
* Per-frame fit: empirical ✓
* Per-CTU split: assumed proportional to predicted activity weight — **never
  measured**

We never extracted real per-CTU bit counts from VTM logs. Per-CTU K_c could
be off by a factor of 2-3 in either direction, particularly for CTUs covered
by inter-prediction (handled poorly by σ_Y proxy).

**Impact on rate-neutral projection**: Σ K_c · 2^{-δ_c/6} = Σ K_c assumes
K_c is correct per-CTU. If K_c is wrong, rate is *not* neutral after VVC
encode — observed empirically: post-round rate drift up to 2.9% (per
liteqp_metadata.json `rate_neutral_log` for Q=32).

### 1.3 Hard-coded "6" in A+ formula — *medium issue*

`analytic_a_plus.py:162`:
```python
delta = -6.0 * np.log2((xi + 1e-9) / g)
```
The "6" comes directly from VVC quantizer step doubling per 6 QP units. It
is the *PSNR*-rate optimal Lagrangian ratio under the high-rate R-D model
of Sullivan & Wiegand. It is **not** derived for *task* accuracy.

A more general form is `δ = -K · log_2(ξ/g)` where `K` is calibrated against
the empirical task-rate slope. We assume K = 6 without testing.

**Impact**: A+ prior is biased toward PSNR-optimal QP redistribution, not
task-optimal. This may explain why pilot_v8b (CNN-direct, no A+ prior) tied
or beat pilot_v8a (CNN-residual on top of A+) — the A+ prior was actively
harmful, not just redundant.

### 1.4 Synthetic ΔmAP oracle — **CRITICAL**

`build_liteqp_dataset.py:136-141`:
```python
def _delta_task_per_ctu(phi: float, delta: int,
                         eta: float, xi: float) -> float:
    """Asymmetric per-CTU mAP response: positive when δ < 0 (better quality)."""
    pos = max(0, delta)
    neg = max(0, -delta)
    return float(-(phi * eta * pos) + (phi * eta * xi * neg))
```
With **η = 0.06** and **ξ = 0.15** hard-coded.

What this says: **at every CTU, mAP changes linearly in δ, scaled exactly by
local saliency φ, with constant slope η = 0.06 per QP step.**

Implications if treated literally:
* φ=1, δ=+5 → predicted ΔmAP = -0.30 (HUGE; observed real changes are <0.05)
* φ=1, δ=-5 → predicted ΔmAP = +0.045 (asymmetric: ξ=0.15 means lowering QP
  helps task only 15% as much as raising QP hurts)
* No diminishing returns
* No interaction across CTUs
* No QP dependence (the curve at QP=27 vs QP=42 is identical)
* No frame-content dependence

**Every pilot since v3** uses this synthetic oracle:
1. Compute φ from occlusion + K from rate-surrogate
2. Build δ_star = arg min_δ [ΔR(δ) - λ · ΔmAP_synthetic(δ)] PER CTU
3. Train MLP/CNN to mimic δ_star
4. Apply at inference, encode, measure REAL mAP

The CNN never sees real mAP. It learns to mimic a hand-crafted formula. If
the formula is wrong (almost certainly it is), the learned δ map is biased
toward what the formula thinks is good, not what actually is good.

**This is the single largest scientific debt in the project.**

### 1.5 "mAP" is actually `precision × recall` — **CRITICAL**

`task_accuracy.py:227-229`:
```python
precision = tp_total / max(tp_total + fp_total, 1)
recall    = tp_total / max(tp_total + fn_total, 1)
ap        = precision * recall
```
The reported `mAP50` is one (precision, recall) point at conf=0.25, IoU=0.5,
multiplied. Real AP integrates precision over recall as the confidence
threshold sweeps from 1.0 down to 0.

Issues:
1. **Not standard**: reviewers will assume COCO mAP. Numbers will not
   reproduce.
2. **Threshold-coupled**: a small change in YOLOv8 confidence (0.25 → 0.30)
   flips many TP/FP near threshold. Real AP is insensitive to this.
3. **Conflates P and R**: `P × R` weights both equally; AUC-PR weights by
   confidence ranking. They can disagree on sign for the same data.
4. **Pseudo-GT = M0 detections**: ground truth is what YOLOv8 sees on
   *uncompressed* frames, not the MOT17 ground-truth annotations. Any error
   in M0 detections is treated as truth.

### 1.6 n=3 × 50 frames + pseudo-GT — *high issue*

* Total samples ~150 frame-detections per (method, QP) cell
* mAP variance from finite sampling: estimated ±0.02-0.04 per cell
* Observed M4-vs-M0 differences: ±0.005 to ±0.05
* → signal often inside the noise floor
* No held-out test set — same 3 sequences used for training and reporting
* No comparison vs published prior art; we only beat (or lose to) M0

---

## 2. Why the framework still produces non-zero numbers

It is *not* the case that the entire pipeline is junk. Specifically:
* φ_oracle is real task signal (D3 will quantify how spatially varied it is)
* K_c is reasonable in aggregate
* The δ map applied to VTM does cause measurable rate and mAP changes
* Some cells (notably MOT17-02 QP=27) show large rate savings at iso-mAP,
  suggesting *some* IPF structure is being exploited correctly

But the overall instability — average BD-Rate-Task swings of ±15pp across
pilots that differ only in inference-time clip schedules — is consistent
with **CNN learning a signal that partially overlaps with truth and
partially overlaps with the synthetic oracle's bias**.

---

## 3. Diagnostic plan (executes alongside pilot_v9b)

### Phase 1 — local, no encode required (~2-4 h)

| ID | Script | Cost | Output |
|----|--------|------|--------|
| D1 | `phase2/diagnostics/d1_true_map.py` | local re-detect on cached decoded frames | true COCO-style mAP per cell |
| D2 | `phase2/diagnostics/d2_statistical_significance.py` | local, from D1 outputs | paired Wilcoxon, Cohen's d, bootstrap CI per cell |
| D3 | `phase2/diagnostics/d3_phi_distribution.py` | local, reads φ_oracle .npy | spatial entropy, percentile spread, "headroom" indicator |
| D-doc | this file + `PROJECT_STATE.md` update | manual | audit trail for paper §Limitations |

**Pre-condition for D1**: cached decoded frames or detection JSONs from
existing pilots must still be on the server. If the orchestrator deletes
`run_dir/decoded_frames/` after evaluation, D1 needs limited re-encode.

### Phase 2 — server, controlled encodes (~15-25 h)

| ID | Script | Cost | Output |
|----|--------|------|--------|
| D4 | `phase2/diagnostics/d4_eta_calibration_design.md` (spec) → impl | 1 seq × 4 QPs × ~6 patterns ≈ 24 encodes ≈ 10 h | empirical η, linearity check |
| D5 | hand-crafted ROI baselines | 3 seq × 4 QPs × 2 patterns ≈ 24 encodes ≈ 10 h | does IPF beat trivial heuristics? |
| D6 | frame-level dQP ablation | 1 seq × 4 QPs × 5 dQP levels ≈ 20 encodes ≈ 8 h | is per-CTU the right granularity? |

### Phase 3 — conditional decision tree

Outcomes from Phase 1+2 determine the path:

```
D1 results (true mAP):
├── Conclusions overturn (e.g. M4 was beating M0 only on P×R, loses on real AP):
│   → Pause all CNN work. The whole CNN was optimizing the wrong target.
│   → Re-train with real-mAP-derived oracle (D4 result).
│
├── Conclusions hold (true mAP shows similar M4 wins/losses):
│   → P×R was approximately monotonic with real AP for our regime.
│   → Continue to D4 to fix oracle.

D4 results (empirical η):
├── η_emp ≪ 0.06 (formula massively overestimates):
│   → δ_star for our pilots has been biased toward overly-aggressive QP swings.
│   → Re-build oracle with η_emp, re-train CNN, re-pilot. Expect smaller |δ|.
│
├── η_emp ≈ 0.06 but nonlinear in δ:
│   → Replace linear ΔmAP with quadratic or sigmoidal model.
│
├── η_emp ≈ 0.06 linear (formula validated):
│   → Oracle is fine; problem is finite-sample noise (D2 result).
│   → Need larger N or stronger detector for paper.

D2 results (stat significance):
├── < 30% of M4-vs-M0 cells have p < 0.05 + |d| > 0.2:
│   → Most "wins" and "losses" are noise. Honest paper must reframe scope.

D3 results (φ distribution):
├── Spatial entropy of φ near max (uniform):
│   → IPF concept fundamentally has no headroom on this data.
│   → Pivot to pure rate-control or to sequences with stronger spatial bias.
```

---

## 4. What this audit does NOT yet validate

We still owe ourselves checks on:
* Does the VTM dQP actually take effect at the CTU level we ask, or does the
  encoder remap our dQP map? (encoder log inspection needed)
* Is the YOLOv8n detector strong enough that any QP change matters? Compare
  with YOLOv8m on the same encodes.
* Is intra-only the right coding mode? Random-access or low-delay-P could
  reveal larger headroom and is closer to deployment.
* Are MOT17-02/04/09 representative? They are all surveillance with similar
  motion patterns. Need diverse content (e.g. driving scenes).

These are deferred to Phase 3+ once the foundational issues are addressed.

---

## 5. Recommended decision rule for the paper

If — after D1, D2, D4 — the picture is:
1. M4 (any pilot) beats M0 by > 2× the noise floor (D2),
2. on at least 2/3 sequences (or 6/12 cells),
3. with true mAP (D1), not P×R,
4. using a calibrated η (D4) so the oracle is empirically grounded,

then the paper has a defensible scientific contribution.

Otherwise, the honest write-up is:

> "We explored a CTU-level learned QP allocation for VCM under VTM-23.4
> intra-only encoding. After validating the rate surrogate (K_c), the task
> oracle (η_emp), and the evaluation metric (true COCO mAP), we found that
> on the MOT17 surveillance subset (3 sequences × 50 frames) the achievable
> rate-task improvement over uniform-QP M0 lies within the metric's noise
> floor. This negative result motivates [larger-N evaluation / stronger
> detector / non-intra coding / different content]."

A negative-tendency paper at top venue is strictly more scientifically
valuable than a paper claiming gains that vanish under proper evaluation.

---

## 6. Action items

- [x] Draft this document
- [ ] Implement D1 (true mAP) — re-evaluate all pilots
- [ ] Implement D2 (statistical significance + bootstrap CI)
- [ ] Implement D3 (φ distribution analysis)
- [ ] Spec D4 (η empirical calibration) for server
- [ ] Update `PROJECT_STATE.md` §5.4 with these blockers
- [ ] Run pilot_v9b in parallel (already prepped) — gives one more data
      point but does **not** answer any of the audit questions

After Phase 1 results land, decide whether to:
* Run D4 on server (validate oracle), or
* Stop and write up a scoped paper, or
* Pivot to a different framework (frame-level QP / RL / no-IPF baseline).
