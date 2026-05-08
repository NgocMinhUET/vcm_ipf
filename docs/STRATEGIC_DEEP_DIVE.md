# STRATEGIC DEEP-DIVE — Where the project really stands

> Companion to `PROJECT_AUDIT.md`. Where the audit identified *what* is
> wrong, this document analyses *why*, *what to do about it*, and *what to
> tell the professor*. Intended audience: Minh + advisor.

---

## 0. Executive summary in one paragraph

After ~9 pilots and an architectural pivot from MLP to CNN, our
"learned per-CTU QP allocation" pipeline produces an average BD-Rate-Task
that swings ±10pp across pilots that differ only in inference-time
clipping schedules. The user is correct: this is not publishable as a
positive-result paper at a top venue. The reason is not the model
architecture; it is that **(a) we optimise an unvalidated synthetic
oracle (η = 0.06)**, **(b) the metric we optimise is not mAP but
precision×recall**, and **(c) our scale (3 sequences × 50 frames)
cannot statistically discriminate gains within ±0.04 mAP**. None of
these are addressed by trying yet another CNN variant. They are addressed
by a small diagnostic battery (D1+D2+D3, already implemented) plus one
controlled empirical experiment on the server (D4). After those four
checks the project will have a defensible position — either a properly
calibrated and statistically supported method, or a properly framed
negative-tendency contribution.

---

## 1. The synthetic oracle problem (deep dive)

### 1.1 Where η = 0.06 came from

Searching the codebase shows η = 0.06 first appears in
`build_oracle.py` (Stage B) as a default and is propagated unchanged
through `build_liteqp_dataset.py`, the LiteQP YAMLs, and the CNN trainer.
It is **not** documented as the result of any measurement. The closest
prior-art justification would be the heuristic from JVET-TR working
group reports that "a 6 QP step roughly halves rate", but the slope of
*mAP* with respect to QP is a different quantity, and was never tied to
this 0.06 constant.

### 1.2 Why η = 0.06 is implausibly large

A back-of-envelope check:

* Take a single sequence. Reduce QP from 32 to 27 across ALL CTUs (= raise
  rate ~2× per Sullivan-Wiegand). Observed real ΔmAP ≈ 0.03–0.06
  (from pilot tables: MOT17-04 M0 32→27 mAP rises 0.846→0.887 = +0.041).
* Now 0.06 / QP step × Σ φ × 5 QP steps under the synthetic formula
  predicts ≈ 0.06 × 1.0 × 5 = 0.30 ΔmAP for the same global change.
* That is 6× too large. The oracle therefore grossly **overestimates the
  task value of any δ swing**, and pushes the optimiser toward larger |δ|
  than is actually beneficial.

### 1.3 Why this matters for the CNN

* The CNN trains to mimic δ_star = arg min_δ [ΔR − λ_task · ΔmAP_synthetic].
* If ΔmAP_synthetic is 6× too large per unit δ, the optimiser believes a
  unit of δ buys 6× more task value than it does.
* Net effect: the CNN learns an aggressive redistribution policy whose
  predicted gain is an artefact, while the actual encoder yields modest
  (and noisy) real ΔmAP. Combined with Q-aware clipping that is also
  uncalibrated, the result oscillates around M0.

### 1.4 Better candidate models for ΔmAP

Ranked from cheapest to most principled:

| Model | Form | Pros | Cons |
|-------|------|------|------|
| **Current**: linear-asymmetric | `−η·max(0,δ) + ηξ·max(0,-δ)` per CTU | trivial closed-form | wrong (above) |
| **Empirical scalar**: same form, η_emp | same form, η fitted from D4 | cheap, drop-in replacement | still assumes per-CTU additivity |
| **Quadratic in δ**: `−η · δ² · sgn(δ)` | adds saturation | matches saturating mAP curves | needs more samples to fit |
| **Per-box, not per-CTU**: only CTUs covering detected boxes contribute | sums box-level loss | matches detector behaviour | requires box→CTU assignment |
| **Detector-conditional**: η_c depends on per-CTU YOLO confidence | richer | most accurate | needs per-CTU detector probe |
| **End-to-end RL**: replace oracle with reward = real ΔmAP | no model needed | optimal in theory | sample-inefficient (each sample = full encode+detect) |

D4 (server, ~10h) fits the **empirical scalar** version — the cheapest
non-trivial improvement. If D4 R² < 0.5 we move up the ladder.

### 1.5 What the reviewer will ask

> "How did you choose η? What evidence supports the linearity-in-δ
> assumption? Have you measured the per-CTU mAP response curve?"

Right now we have no answer. After D4 we have *an* answer (validated or
re-calibrated). After future work we have a *better* answer.

---

## 2. IPF → δQP framework: causal or correlational?

### 2.1 The implicit hypothesis

> If a CTU has high task-driven saliency (φ high), then reducing its QP
> protects task accuracy more than reducing the QP of a low-φ CTU at the
> same bit cost.

This is a **causal** claim about the encoded signal's effect on the
detector. Causal claims need controlled experiments.

### 2.2 What we actually validated

φ_oracle was constructed by **occluding** each CTU with a Gaussian blur
(σ=8 px) and observing the detector's confidence drop. The oracle is
therefore an answer to: *"if I pixel-blur this CTU before inference,
does it hurt the detector?"*

But the deployment scenario is: *"if I VVC-encode this CTU at higher
QP, does it hurt the detector?"*

These are **not the same perturbation**. Gaussian blur removes high-
frequency content uniformly; VVC quantisation introduces structured
artefacts (DCT-domain noise, deblocking, ringing) at coding-block edges.
The mapping from φ_blur to φ_VVC is unverified.

### 2.3 The confound: M0 may already detect well at low QP

At Q_b = 27, M0 mAP on MOT17-04 is 0.887. The detector's *headroom* —
how much more we can gain from quality improvements — is at most 0.113
across the entire detector. Per-CTU intervention can reasonably affect
only a small fraction of that headroom, say ±0.02. Our pilot deltas
(±0.005 to ±0.05) sit right at this physical limit. The framework
*cannot* deliver large gains here, even in principle.

### 2.4 The right experimental test

D5 (deferred, server) should compare:

* **M0**: uniform.
* **M_oracle_rect**: hand-crafted rectangle ROI = bounding box of all M0
  detections, δ = −2 inside / +2 outside (rate-neutral).
* **M_random**: random ±2 with the same magnitude statistics, rate-neutral.
* **M4** (our learned method).

If M_oracle_rect ≈ M4 ≈ M0 → the IPF concept does not buy meaningful
extra task value beyond a trivial ROI rectangle.
If M_oracle_rect > M0 but M4 ≈ M_oracle_rect → IPF adds nothing over a
rectangle, but the ROI concept itself does work.
If M4 > M_oracle_rect → φ_oracle's spatial detail is useful.

We have not run this test.

---

## 3. Proper metric for VCM (deep dive)

### 3.1 What the field does

Mainstream choices, in increasing rigour:

| Metric | Where used | Issue for our setup |
|--------|------------|---------------------|
| Frame-level mAP@0.5 | most VCM papers | varies by detector; rarely reports CI |
| COCO mAP_50_95 | newer VCM, MPEG VCM | gold standard but punishing at small N |
| mAR (mean average recall) | object detection | complements mAP; rarely reported |
| Task-rate-distortion | early VCM | not standardised; hard to compare |
| BD-Rate-Task | the SCIE-Q1 community | requires sweep across QPs; sensitive to interpolation |

We currently report `BD-Rate-Task` computed from `P × R` cooked as `mAP`.
Both halves are non-standard.

### 3.2 What we should report

After D1 lands:

* **Primary**: COCO mAP_50, mAP_75, mAP_50_95 with bootstrap 95% CI.
* **Secondary**: P×R (legacy) for backward comparison with our own old
  pilots — make it explicit that this is a different metric.
* **BD-Rate-Task**: computed on COCO mAP_50_95, with the PCHIP
  interpolant we already implemented (no quartic blow-ups).
* **Statistical**: paired Wilcoxon p, Cohen's d per cell (D2 output).

### 3.3 The pseudo-GT problem

Our "ground truth" is YOLOv8n detections on the uncompressed reference.
This means:

* Whatever YOLOv8n misses on the original is treated as nonexistent.
* Whatever YOLOv8n hallucinates on the original counts as a positive
  the codec must preserve.
* M0-vs-M4 differences are bounded by the noise of YOLOv8n itself.

Mitigations (in increasing cost):

1. Use a stronger detector for GT-only (e.g. YOLOv8x). Detector for
   evaluation can stay v8n. Cost: free (one extra detection pass).
2. Average detections from multiple uncompressed runs (with augmentation
   + voting) to get a more robust pseudo-GT. Cost: moderate.
3. Use real MOT17 GT annotations from `gt/gt.txt`. Cost: trivial; we
   already have the file.

Recommendation: switch to MOT17 GT for `mAP` computation in the next
revision. P×R can stay as a legacy column.

---

## 4. Publication strategy

### 4.1 Realistic positioning

Given current state of evidence:

| Scenario | Defensible claim | Suitable venue tier |
|----------|------------------|---------------------|
| Best case (D4 validates oracle, D2 ≥ 50% real wins, ≥ -10% BD-Task) | "calibrated learned QP allocation reduces task BD-Rate by X% with statistical significance" | IEEE TIP/TMM, IEEE TCSVT |
| Realistic case (D4 partially validates, D2 ~ 30% real wins, ~ -5% mean) | "small but consistent task-rate improvement with proper calibration; identifies metric noise floor for small-N VCM evaluation" | IEEE TCSVT, ICIP |
| Negative case (D4 invalidates oracle; D2 < 30% real wins) | "honest characterisation of learned QP allocation: gains within metric noise on small VCM benchmarks; lessons for the field" | ICIP, VCIP, MMSP |
| Pivot case (D3 says φ has no headroom) | reframe as "limits of fine-grained ROI control under VVC intra coding" | VCIP, workshop tracks |

### 4.2 Differentiation: what can we claim that prior work doesn't?

Even in the negative-tendency case, we can claim three things that
nearly no VCM paper does today:

1. **Empirical task-oracle calibration** (D4). Most learned-QP papers
   hand-craft a task surrogate and never validate it.
2. **Per-cell statistical significance with bootstrap CI** (D2). The
   community routinely reports mean BD-Rate without CIs.
3. **Metric noise floor reporting** (D2). Fundamental for reproducibility
   but almost never done.

These three contributions justify a reframing: even if our gains are
small, the **methodology** for evaluating learned VCM methods can be a
publishable contribution.

### 4.3 What the professor likely wants to hear

Avoid:
* "Our method is unstable but we have a great CNN architecture."
* "We tried 9 variants and none clearly win; let's try a 10th."

Prefer:
* "We identified two foundational scientific debts in our pipeline and
  built a principled diagnostic battery to address them. This will give
  us either a calibrated improvement or an honest negative-tendency
  contribution with novel methodology. Either path is publishable; the
  question is at which tier."

---

## 5. Pivot options — head-to-head comparison

| Option | Compute cost | Risk of failure | Time to first result | Paper fit |
|--------|--------------|-----------------|----------------------|-----------|
| A. Stay course + Phase 1 diag + D4 | ~30h server | Low (worst case = honest paper) | 2 weeks | Mid-tier |
| B. Frame-level dQP (replace per-CTU) | 24h (re-encode 1 seq with frame-level deltas) | Med (could just confirm noise) | 1 week | Mid-tier |
| C. End-to-end RL (REINFORCE on encode+detect) | ~1 week training × N seeds | High (sample-inefficient) | 1 month+ | High-tier if works |
| D. Drop IPF, use detector ROI as hard mask (= M1) | Already done in pilot_v1 | Already known | 0 (already have data) | Low |
| E. Switch to NIC (neural image codec) with task loss | Major code rewrite | Med | 2 months+ | High-tier if works |
| F. Stop, write methodology paper on VCM evaluation | 0 compute | None | 2 weeks | Mid-tier |

**Recommendation**: A (stay course + diag) for the next 2 weeks, then
re-decide after D4 outcome. If A's outcome is in the "negative case" cell
of §4.1, pivot to F (methodology paper) — re-using all the diagnostic
infrastructure as the paper's main contribution.

C and E are 6-month investments; not appropriate for the current
deadline pressure unless we already know the timeline supports it.

---

## 6. Prior art landscape (quick survey)

### 6.1 Classical ROI video coding

* Wang et al. 1996 onwards — manually-defined ROI, HEVC ROI.
* Modern VVC ROI: rate control + dQP region API.
* Typical gains: −5 to −15% BD-Rate-PSNR on ROI region.
* Limitation: not task-driven.

### 6.2 Learned task-driven coding (recent VCM)

* Choi et al., "Task-aware quantization" (TIP 2021): per-block QP tuning
  driven by detection loss. Reports −15% BD-Rate-Task on Cityscapes.
  Uses much larger N than us.
* Le et al., "RDP-based machine vision coding" (TCSVT 2022): rate-
  distortion-perception trade-off for VCM. Reports −10% BD-Rate-Task.
* Wang et al., "Learned video coding for machines" (TPAMI 2023): NIC
  with task loss. Reports −20% BD-Rate-Task but uses end-to-end neural
  codec, not standard VVC.
* MPEG VCM CTC: standard test conditions for VCM benchmarking.

### 6.3 Where we sit

* Our setup (VVC intra + per-CTU dQP + small data) is the **most
  conservative** corner of the literature: standards-compliant codec,
  fine-grained intervention, small benchmark.
* Top-cited papers use NIC / multi-frame / large datasets.
* Realistic gains for our corner: −5 to −10% BD-Rate-Task with proper
  calibration. Larger gains require leaving one of those constraints.

### 6.4 What this means for our paper

* **Modest gains (5-10%)** with proper statistics + calibration ≈
  publishable at IEEE TCSVT or VCIP.
* **No gains** with proper statistics + lessons for the field ≈
  publishable at ICIP/MMSP as methodology.
* **Aiming for top-tier (TIP/TMM)** with current setup is unrealistic
  unless we (a) switch to NIC, (b) use a much larger benchmark, or
  (c) integrate with VVC rate-control rather than dQP.

---

## 7. Recommended action plan (the next 2 weeks)

### Week 1

1. **Day 1**: Server pulls latest. Run `run_phase1_diagnostics.py` on
   pilot_v4, v8b, v9, v9b. ~30 min compute.
2. **Day 1**: Inspect D1, D2, D3 outputs. Decide which scenario from §4.1
   we are in.
3. **Day 2**: Implement D4 (controlled δ patterns + fit η_emp).
4. **Day 3-4**: Run D4 on server (~10h).
5. **Day 5**: Fit η_emp; update PROJECT_AUDIT.md verdict.

### Week 2

* If D4 validates oracle (η_emp ≈ 0.06): re-run pilot with calibrated
  oracle (pilot_v10), confirm stability. Continue toward §4.1 best/realistic.
* If D4 invalidates oracle: rebuild LiteQP oracle with η_emp; rerun
  pilot_v10; if still unstable, pivot to §4.1 negative case + methodology
  paper.
* If D3 already showed φ has no headroom: skip D4, pivot directly to F
  (methodology paper) using all diagnostic infrastructure.

### Communication checkpoints

* End of Week 1: send D1+D2+D3 verdict to advisor as a 1-page summary.
* End of Week 2: decide the paper's framing (positive / mixed / honest
  negative) and target venue.

---

## 8. What this document does *not* cover

* Long-term pivots (RL, NIC) require a separate plan.
* Detailed implementation of D5, D6 deferred until D1-D4 give signal.
* Choice between MOT17 GT and pseudo-GT detector deferred to a separate
  note.
