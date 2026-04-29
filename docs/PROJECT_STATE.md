# PROJECT STATE — Persistent Memory

> **Canonical location (in-repo)**: `phase2/docs/PROJECT_STATE.md` on the
> `phase2` branch of `vcm_ipf`. The legacy local-only copy at
> `IPF_Development_Plan/PROJECT_STATE.md` (outside any git repo) is
> deprecated and will not be pushed; treat this file as the source of truth.
>
> **Purpose**: This file is the single source of truth that the AI assistant
> reads at the start of every session to maintain continuity across chats.
> When making meaningful progress, **always update this file** to reflect the
> latest project state.
>
> **Reading order**: Latest section first (top of file). Older context is
> archived in `## Historical Log` at the bottom.

---

## START-OF-SESSION CHECKLIST (for the AI assistant)

Before answering ANY question:
1. Read this entire file (don't rely on summaries).
2. Check §5.4 "Outstanding issues" — these are the active blockers.
3. Check §5.3 for the most recent commits to understand what code is on the server.
4. If the user asks "what's next?", answer based on §5.4 ordered by severity.

After making meaningful progress, UPDATE this file:
- New bug discovered → add to §5.4
- Bug fixed → move to §5.3 with commit hash
- Server experiment ran → log result in §6 or §10
- Branch / file moved → update §3 and §5.2

---

## TEST TIER HIERARCHY

**ALWAYS run the smallest tier that answers the question.** Never re-launch
`pilot` after a code change without first proving end-to-end correctness on
`smoke` and `quick`.

| Tier | Config | Sequences | Methods | QPs | Frames | Total runs | Wall-clock |
|------|--------|-----------|---------|-----|--------|------------|------------|
| **smoke** | `phase2/configs/smoke.yaml` | 1 (MOT17-04) | 2 (M0, M4) | 2 (27, 37) | 5 | 4 | ~5–10 min |
| **quick** | `phase2/configs/quick.yaml` | 1 (MOT17-04) | 5 (M0-M6) | 2 (27, 37) | 10 | 10 | ~30–60 min |
| **pilot** | `phase2/configs/pilot.yaml` | 3 | 5 | 4 | 50 | 60 | ~30–40 h |

**Workflow protocol**:
1. Code change → `smoke` (5 min) → confirm M0 ≠ M4, mAP > 0, PSNR-ROI ≠ PSNR-full
2. `smoke` passes → `quick` (1 h) → confirm RD curves & all baselines work
3. `quick` passes → `pilot` (overnight) → final paper data

Server commands:
```bash
# Smoke
bash scripts/run_pilot.sh configs/smoke.yaml

# Quick (after smoke passes)
bash scripts/run_pilot.sh configs/quick.yaml

# Re-evaluate existing recon.yuv (no re-encoding) — much faster!
python -m phase2.pipeline.rerun_evaluation --config configs/<tier>.yaml
```

---

## 0. Identity & Communication

- **User name**: Minh (UET).
- **Assistant name**: Minh (per user rule).
- **Always respond in English** even though some user messages are in Vietnamese.
- **Ngôn ngữ user**: Vietnamese (mixed with English technical terms).

---

## 1. Project: IPF — Importance Potential Field for Video Coding for Machines (VCM)

**Goal**: Submit a SCIE-Q1 paper proposing IPF, a physics-inspired ROI-aware
QP control framework that achieves better BD-Rate vs. machine-vision task
accuracy on MOT17 sequences than competing soft-map baselines.

**Master plan**: `IPF_Development_Plan/09_MASTER_EXECUTION_PLAN.md`.

---

## 2. Server Environment

- **Host**: `tanpx-Ubuntu` (Linux, GCC 11.4, CUDA-capable).
- **Conda env**: `ipf` (always `conda activate ipf` first).
- **Server account**: `guest@tanpx-Ubuntu`.
- **Project root on server**: `~/Minh/ipf/`.
  - `phase1/` — soft-map QP generators (Phase 1 complete, on `main` branch).
  - `phase2/` — VTM integration pipeline (currently on `phase2` branch).
  - `vtm/VVCSoftware_VTM/` — VTM-23.4 source + patched binaries.
    - **Patched encoder**: `~/Minh/ipf/vtm/VVCSoftware_VTM/bin/EncoderAppStatic` ← supports `--ExternalQPMapDir`
    - **Patched decoder**: `~/Minh/ipf/vtm/VVCSoftware_VTM/bin/DecoderAppStatic`
    - The `bin/umake/gcc-11.4/x86_64/release/EncoderApp` binary is UNPATCHED — do NOT use it.
  - `datasets/MOT17/MOT17/train/MOT17-{02,04,09,11,13}-DPM/img1/` — MOT17 frames (**`.jpg` format!**).
  - `datasets/yuv/MOT17-*.yuv` — YUV 4:2:0 conversions (**1920x1152**, CTU-128 padded from 1080).
- **Phase 1 outputs**: `~/Minh/ipf/phase1_outputs/multi_seq_<sequence>/<method>/qp_vtm/`.
  - Phase 1b **COMPLETED** (5/5 sequences: MOT17-02,04,09,11,13-DPM, 50 frames each).
  - Cross-sequence analysis at `~/Minh/ipf/phase1_outputs/cross_sequence_analysis/`.
- **Phase 2 outputs**: `~/Minh/ipf/phase2_outputs/`.

## 3. Git Workflow

- **Repo**: `https://github.com/NgocMinhUET/vcm_ipf.git`
- **Branches**: `phase1` (Phase 1 complete), `phase2` (active development).
- **Local repos** (each is a separate git repo, NOT a single monorepo):
  - `IPF_Development_Plan/phase1/` → branch `phase1`
  - `IPF_Development_Plan/phase2/` → branch `phase2`
- **Sync to server**: user runs `git pull origin phase2` then `pip install -e . -q`.

## 4. Methods (M0–M6)

| ID | Name | Description | Phase 1 status |
|----|------|-------------|----------------|
| M0 | Uniform | Baseline: constant QP everywhere (anchor for BD-Rate) | N/A (no QP maps) |
| M1 | Binary | Hard ROI mask: ΔQP=−6 inside box, +0 outside | Complete |
| M2 | Linear | Linear QP gradient from box center | Complete |
| M3 | Gaussian | Gaussian falloff from box center | Complete |
| M4 | **IPF v2** | **Our method**: physics-inspired potential field, **MAX superposition** | Complete |
| M5 | Anisotropic | Direction-aware Gaussian | Complete |
| M6 | RD-driven | Lagrangian-style optimal allocation | Complete |

**Pilot uses subset**: M0, M1, M4, M5, M6. M4 is the primary method to compare.

## 5. Phase 2 Pipeline — Current State (last updated 2026-04-26)

### 5.1 Pipeline architecture

```
┌─ Phase 1 outputs (qp_vtm/ per-method) ──┐
│                                          ▼
│  YUV (1920x1152) ──→ VTM EncoderApp (patched, --ExternalQPMapDir) ──→ bitstream.bin + recon.yuv
│                                                                          │
│                                          ┌──── DecoderApp (compliance) ──┘
│                                          ▼
│  PSNR (full + ROI + BG)  ←── compute_sequence_psnr (needs boxes!)
│  Task accuracy (mAP)     ←── YOLOv8 detect on decoded vs. original frames
│                                          ▼
└────────────────── BD-Rate / BD-Task aggregation
```

### 5.2 Critical files

| File | Purpose | Last touched |
|------|---------|--------------|
| `phase2/scripts/apply_vtm_patch.sh` | Patches VTM C++ to support `--ExternalQPMapDir` + lambda update | 2026-04-09 |
| `phase2/src/phase2/encoding/vtm_encoder.py` | VTM wrapper with bitrate/PSNR parsing + env var pass-through | 2026-04-10 |
| `phase2/src/phase2/pipeline/encode_pipeline.py` | Orchestrates encode→decode→PSNR→mAP per (seq, method, QP) | 2026-04-10 |
| `phase2/src/phase2/analysis/results_aggregator.py` | BD-Rate/BD-Task aggregation, t-tests, CIs | 2026-04-09 |
| `phase2/configs/pilot.yaml` | 3 sequences × 5 methods × 4 QPs = 60 runs (QP: 32,37,42,47) | 2026-04-10 |

### 5.3 Resolved issues (commits)

| Commit | Fix |
|--------|-----|
| `49ef937` | Bitrate file-size fallback, adaptive timeout, n_frames 200→50 |
| `07f38b0` | SUMMARY block parser handles VTM-23.4 `I Frames` header |
| `392bdfd` | YUV-based PSNR fallback (Tier-4) |
| `2f4eeed` | BUG #2 fix: PSNR-ROI now uses YOLO boxes. BUG #3 fix: task glob `.jpg/.jpeg/.png`. Adds `rerun_evaluation.py`. |
| `9d904ee` | Test tier infrastructure: smoke/quick/pilot configs + tier-aware `run_pilot.sh`. |
| `9873ad7` | Python 3.8 compat in phase1 config.py + constants.py |
| `15ee2b3` | Phase 1 outputs → `~/Minh/ipf/phase1_outputs/` |
| `4c2950a` | Phase 2 configs → point to new phase1_output_dir |
| (binary fix) | BUG #1 root cause fixed: EncoderAppStatic is patched binary |
| (lambda fix) | VTM patch: `setLambda()` alongside `setSliceQp()` per CTU; env-var fallback `VTM_EXTERNAL_QP_DIR` |

### 5.4 Outstanding issues

**🔴 CRITICAL: QP map calibration mismatch (identified from pilot_v1)**
- **Description**: Phase 1 IPF maps store ABSOLUTE QP values (~33.6 avg for M4).
  At Q_base=37/42/47, ALL CTUs receive a QP LOWER than the base (since map avg=33.6 < 37/42/47).
  This destroys the relative ROI protection: every CTU gets a quality boost vs. the base,
  so there is NO differential allocation between ROI and background.
  The method only works correctly at Q_base=32 (the Phase 1 calibration QP).
- **Impact**: BD-Rate-Task = +57.5% averaged across sequences (M4 is worse than M0 overall).
- **Fix required**: Generate QP maps with RELATIVE offsets (dQP, not absolute QP),
  OR generate separate maps per Q_base level.
  New formula: `QP_map(x,y) = Q_base + IPF_delta(x,y)` where `IPF_delta ∈ [-8, +4]`.

**🟡 PSNR-ROI paradox (diagnostic finding)**
- **Description**: PSNR-ROI is LOWER for all adaptive methods vs M0 despite lower ROI CTU QP.
  Root cause: YOLO bboxes span many CTUs including background border CTUs.
  IPF-compressed background CTUs "spill" into the bbox area, lowering mean PSNR-ROI.
- **Implication**: PSNR-ROI is NOT a useful proxy for machine task quality.
  mAP50 is the correct optimization target. Phase 3 should minimize BD-Rate-Task, not BD-Rate-ROI.
- **Action**: Remove PSNR-ROI from paper's main metric table; keep as supplementary diagnostic.

---

## 6. Pilot v1 Results — FINAL (2026-04-26)

**Status**: ✅ Complete — 60/60 runs successful, 3 sequences × 5 methods × 4 QPs × 50 frames.

**QP points**: 32, 37, 42, 47

### 6.1 BD-Metrics Summary (averaged over 3 sequences: MOT17-{02,04,09}-DPM)

| Method | BD-Rate-PSNR | BD-Rate-ROI | **BD-Rate-Task** | BD-PSNR (dB) | **BD-Task** | Rank |
|--------|-------------|------------|-----------------|-------------|------------|------|
| M5 (Anisotropic) | +10.2% | +22.1% | **+30.5%** | −0.374 | **+0.0016** | 1 ← best |
| M4 (IPF v2 — Ours) | +11.1% | +22.1% | **+57.5%** | −0.402 | **−0.0083** | 2 |
| M6 (RD-driven) | +10.3% | +21.0% | **+102.2%** | −0.378 | **−0.0101** | 3 |
| M1 (Binary) | +11.6% | +25.8% | **+170.2%** | −0.423 | **−0.0115** | 4 |

**Verdict**: All adaptive methods are currently WORSE than M0 in BD-Rate-Task.
M5 is the best (BD-Task = +0.0016, marginally positive) but not statistically significant.
Only ONE statistical test reaches significance (M4 vs M1 on BD-Rate-ROI, p=0.026, n=3).

### 6.2 Per-QP breakdown — MOT17-04-DPM (most diagnostic)

| QP | M0 Rate | M0 mAP | M4 Rate | M4 mAP | ΔRate | ΔmAP | Verdict |
|----|---------|--------|---------|--------|-------|------|---------|
| 32 | 400.5 | 0.846 | 396.5 | **0.871** | −1.0% | **+0.025** | M4 wins ✓ |
| 37 | 215.9 | 0.821 | 214.9 | 0.788 | −0.5% | −0.033 | M0 wins ✗ |
| 42 | 116.6 | 0.802 | 115.8 | 0.777 | −0.7% | −0.025 | M0 wins ✗ |
| 47 | 59.8 | 0.617 | 62.0 | **0.664** | +3.7% | **+0.047** | M4 wins ✓ |

### 6.3 Key finding

IPF v2 ONLY works correctly at Q_base=32 (its calibration QP).
At Q_base=37/42/47, the absolute QP values in the map (~33.6) are LOWER than the
base QP → every CTU gets quality boost, destroying differential ROI protection.

---

## 7. Phase 3 Plan — Learned QP Adaptation

**Governing document**: `IPF_Development_Plan/11_PHASE3_RESEARCH_PROTOCOL.md`.
Read that file before touching Phase 3 code or configs.

### 7.1 Core problem (summary of §1 of the protocol)
`QP_map(x,y) = Q_base − max_i Φ_i(x,y)` has two flaws:
1. Uses absolute QP → breaks at QP ≠ calibration point (pilot v1 evidence).
2. `max` operator is hand-picked — not established as Pareto-optimal.

### 7.2 Function family hierarchy (protocol §2)
Levels 1–5 are parametric families, each strictly generalizing the previous.
Level 6 (lightweight DNN, ≤ 2 000 params) is a pre-registered fallback only
invoked if Level 5 does not reach the target BD-Task (see §5.3 of the protocol).

| Level | Form | Params |
|-------|------|--------|
| L1 | δ = −α·Φ̂ | 1 |
| L2 | asymmetric power-law (current IPF mapping) | 5 |
| L3 | L2 × Q-adaptive scale | 6 |
| **L4** | **L2 composed with L_p-norm superposition** | **6** |
| L5 | L2 composed with softmax-temperature aggregator | 6 |
| L6 | 15→16→8→1 MLP (conditional) | 409 |

### 7.3 Phase 3 stages (protocol §7.1 + this session)
| Stage | Action | Status |
|-------|--------|--------|
| 0 | Phase 1 dQP calibration fix + Lp-norm infra | DONE (commit 2026-04-26) |
| A | smoke_v5 verifies M4 ≠ M0 across QP grid | DONE (2026-04-27) |
| A | quick_v3 (10 frames, p=∞): −0.4..−2.6 % rate, +0.04 mAP @ QP=32 | DONE — gains marginal |
| B.1 | Occlusion saliency Φ_oracle | DONE (2026-04-27) |
| B.2 | Rate surrogate K_c from luma variance + R-λ theory | DONE (2026-04-27) |
| B.3 | Surrogate-driven oracle dataset (no VTM) | DONE |
| B.4 | Parametric fit Levels 1-5 + LOSO + bootstrap CI | DONE — fits hit bounds |
| B.6 | Oracle-Direct (fixed params) → pilot_v3 | DONE — mixed results |
| pilot_v3 | M4 = oracle-direct (μ=0.35, Δroi=6, Δbg=4) | DONE 2026-04-29 |
| pilot_v3 | MOT17-04 ~46 % rate saving at equal mAP=0.846 | DONE — headline result |
| pilot_v3 | MOT17-02 unstable at QP=27 (low-QP intra mismatch) | DIAGNOSED |
| **C.0** | **Stage C plan: analytic prior + tiny MLP residual (LiteQP)** | **DONE (this commit)** |
| **C.1** | **`analytic_a_plus.py` — RD-log A+ + Q-adaptive bounds + exact rate-neutral proj.** | **DONE (this commit)** |
| **C.2** | **`build_liteqp_dataset.py` — features + δ_a+ + δ_star + residual** | **DONE (this commit)** |
| **C.3** | **`train_liteqp_regressor.py` — MLP residual w/ mandatory LOSO** | **DONE (this commit)** |
| **C.4** | **`apply_liteqp_model.py` — per-QP δQP maps (A+ + r̂ → projection → clip)** | **DONE (this commit)** |
| **C.5** | **`pilot_v4.yaml` + `run_phase3_liteqp_pipeline.py` orchestrator** | **DONE (this commit)** |
| 1 | Run `phase2/scripts/run_phase3_liteqp_pipeline.py` end-to-end on server | pending user |
| 2 | Encode `pilot_v4.yaml` (M0 vs M4-LiteQP) | pending Stage 1 |
| 3 | Compare pilot_v3 (oracle-direct) vs pilot_v4 (LiteQP) → final table | pending Stage 2 |

### 7.4 Stage B rationale (academic justification)
Why surrogate-driven instead of pure VTM oracle:

1. **Cost.** Per-CTU exhaustive VTM perturbation = 17 000 encodes ≈ 280 h.
   Surrogate fit costs ~50 min and lets us evaluate ~50 000 (CTU, δ) cells.
2. **Theoretical grounding.** Rate surrogate uses the canonical R-λ
   form `R_c = K_c · 2^{-Q/6}` (Sullivan & Wiegand 1998). Per-CTU
   complexity proxy `K_c ∝ σ_Y(c)^ρ · (1+|MV(c)|)^ν` is the standard
   complexity model in HEVC/VVC rate-control literature
   (Lin & Chao TCSVT 2016, Liu JVT 2017).
3. **Task surrogate from occlusion.** Φ_oracle(c) is defined exactly
   as Zeiler & Fergus 2014 occlusion sensitivity — a well-cited
   ground-truth signal for vision-task importance.
4. **Falsifiability free.** While running occlusion saliency we
   compute ρ_spearman(Φ_IPF, Φ_oracle) for free; ρ < 0.30 raises a
   warning to consider switching from IPF to direct saliency.
5. **Verification step.** After fitting, surrogates are spot-checked
   against ~5 real VTM encodes (this is what `pilot_v3.yaml` does).

### 7.5 Immediate next action for user — Stage C (LiteQP)

The LiteQP method = analytic RD-log A+ prior + tiny MLP residual correction
(≤2 QP), rate-neutral projected, Q-adaptively clipped. This addresses the
mixed pilot_v3 results (especially MOT17-02 instability at QP=27).

```bash
# On server:
cd ~/Minh/ipf/phase2
git pull origin phase2
pip install -e . -q

# 1) Run Stage C end-to-end (~60 min on a single A100; reuses Stage B
#    saliency / rate caches if they already exist):
cd ~/Minh/ipf/phase2
PYTHONPATH=src python scripts/run_phase3_liteqp_pipeline.py \
    --config configs/phase3_liteqp.yaml

# Sanity-check the LOSO MAE report:
cat ~/Minh/ipf/phase3_outputs/fit/liteqp_mlp.report.json | python -m json.tool
# Healthy: loso_mean_mae ≤ 0.6, |bias| ≤ 0.2

# Verify per-QP delta maps exist:
ls ~/Minh/ipf/phase3_outputs/learned/liteqp_MOT17-04-DPM/M4/qp_vtm_delta_QP32/ | head -3

# 2) Encode pilot_v4 (~14 h):
cd ~/Minh/ipf/phase2
bash scripts/run_pilot.sh configs/pilot_v4.yaml

# 3) Compare BD-Rate-Task: pilot_v1 vs pilot_v3 vs pilot_v4
python scripts/compare_pilots.py \
    ~/Minh/ipf/phase2_outputs/pilot_v1/experiment_summary.json \
    ~/Minh/ipf/phase2_outputs/pilot_v3/experiment_summary.json \
    ~/Minh/ipf/phase2_outputs/pilot_v4/experiment_summary.json
```

### 7.6 Phase 3 evaluation targets (pre-registered)
Primary: BD-Rate-Task(IPF_θ ‖ M0) < 0 %, 95 % bootstrap CI excluding 0.
Secondary: BD-Rate-Task(IPF_θ ‖ M5) ≤ −5 %.
Diagnostic: ρ_spearman(Φ_IPF, Φ_oracle) over all 50 × 3 frames.

### 7.7 Phase 3 critical files

#### Stage B (DONE, fits hit bounds → superseded by Stage C)
| Path | Purpose |
|------|---------|
| `phase2/src/phase2/phase3/occlusion_saliency.py` | Φ_oracle generator + ρ diagnostic |
| `phase2/src/phase2/phase3/rate_surrogate.py` | K_c calibrator from σ_Y + temporal diff |
| `phase2/src/phase2/phase3/build_oracle.py` | Surrogate-driven oracle assembler (Stage B fits) |
| `phase2/src/phase2/phase3/fit_parametric.py` | L1-L5 + bootstrap + LOSO |
| `phase2/src/phase2/phase3/train_dnn.py` | Level 6 lightweight MLP fallback |
| `phase2/src/phase2/phase3/apply_oracle_qp.py` | Pilot_v3 oracle-direct (fixed params) |

#### Stage C — LiteQP residual learning (THIS COMMIT)
| Path | Purpose |
|------|---------|
| `phase2/src/phase2/phase3/analytic_a_plus.py` | RD-log A+ closed-form prior + Q-adaptive bounds + exact rate-neutral projection (NEW) |
| `phase2/src/phase2/phase3/build_liteqp_dataset.py` | Per-CTU JSONL with 9 features + δ_a+ + δ_star + residual (NEW) |
| `phase2/src/phase2/phase3/train_liteqp_regressor.py` | sklearn MLP-(16-8) residual + LOSO + bootstrap report (NEW) |
| `phase2/src/phase2/phase3/apply_liteqp_model.py` | A+ + MLP residual → projection → Q-clip → δQP map; supports `--mode a_plus` ablation (NEW) |
| `phase2/src/phase2/pipeline/encode_pipeline.py` | `_find_qp_maps` now Q_base-aware (per-QP `qp_vtm_delta_QP{n}/` preferred) (UPDATED) |
| `phase2/scripts/run_phase3_liteqp_pipeline.py` | End-to-end Stage C orchestrator (NEW) |
| `phase2/configs/phase3_liteqp.yaml` | Stage C path/parameter config (NEW) |
| `phase2/configs/pilot_v4.yaml` | M0 vs M4-LiteQP encode config (NEW) |

#### Foundational (unchanged this commit)
| Path | Purpose |
|------|---------|
| `11_PHASE3_RESEARCH_PROTOCOL.md` | Governing protocol (read first) |
| `phase1/src/phase1/control/qp_mapper.py` | `map_field_to_delta_qp()` |
| `phase1/src/phase1/export/qp_exporter.py` | `export_delta_qp_vtm()` |
| `phase1/configs/ipf_lp.yaml` | `save_field_npy: true` (Stage B input) |
| `phase2/src/phase2/encoding/vtm_encoder.py` | Auto-compose dQP + Q_base |
| `phase2/configs/pilot_v3.yaml` | Pilot v3 encode config (oracle-direct method) |

---

## 8. Standing Instructions

- **Always update this file** when:
  - A new bug is discovered or fixed
  - Branch/commit state changes
  - The user gives a new high-level directive
  - A new server path or environment fact is learned
- **Always read this file first** at the start of a new session.
- **Be brutally honest** when results don't make scientific sense. Never
  declare success on a run with mAP=0 or M0=M1=M4=M5=M6.

---

## Historical Log (oldest first)

- **2026-03**: Phase 1 (proxy metric only, single-sequence) completed and
  reported in `BAO_CAO_IPF_v3.docx` + `IPF_Presentation.pptx`. IPF v2
  (max superposition) won across all proxy metrics on `comp_v4` results.
- **2026-04 early**: Phase 1b (multi-sequence statistical validation) +
  Phase 2 (VTM integration) implemented. Many fix iterations on VTM patch,
  binary path, deprecated VTM options, log parsing.
- **2026-04-10**: Pilot v1 started (60 runs). 3 critical bugs discovered
  in cross-method comparison (BUG #1: M0=M4, BUG #2: PSNR-ROI=0, BUG #3: mAP=0).
- **2026-04-21**: Phase 1b completed (5/5 sequences). Python 3.8 compat fixed.
  BUG #1 root cause: UNPATCHED binary + missing lambda update in VTM C++ patch.
  BUG #2 fixed: PSNR-ROI now uses YOLO boxes.
  BUG #3 fixed: YOLO task accuracy working.
  Lambda fix applied (VTM patch now updates m_pcRdCost->setLambda per CTU).
  smoke_v4 confirmed BUG #1 resolved. quick_v2 run (10 frames) confirmed mAP non-zero.
- **2026-04-26**: Pilot v1 COMPLETE (60/60). Re-evaluation with PSNR-ROI fix done.
  BD-Rate-Task = +57.5% for M4 (WORSE than M0). Root cause identified: QP maps are
  ABSOLUTE values calibrated for Q_base=32 only. At Q_base=37/42/47, maps apply
  universal quality boost with no relative ROI protection.
  Phase 3 plan drafted: convert to relative dQP format, then data-driven Lp-norm fitting.
- **2026-04-26 (late)**: Phase 3 Stage 0 implemented.
  - Wrote `11_PHASE3_RESEARCH_PROTOCOL.md` — registered-report-style protocol
    governing all Phase 3 work (function-family hierarchy L1–L6, oracle design,
    Lagrangian fitting, pre-registered decision tree, success criteria).
  - Phase 1: added `map_field_to_delta_qp()` + `export_delta_qp_vtm()` so Phase 1
    now emits a Q_base-agnostic `qp_vtm_delta/` directory alongside the legacy
    absolute `qp_vtm/`. Added `FieldConfig.p_norm` and L_p-norm superposition
    (`_lp_aggregate`) continuously interpolating between sum (p=1) and max (p=∞).
  - Phase 2: `VTMEncoder` auto-detects the QP-map format from the file header
    and composes deltas with the run-time Q_base into a temp abs-QP directory
    before invoking VTM. `EncodingPipeline._find_qp_maps` prefers `qp_vtm_delta/`.
  - New `phase1/configs/ipf_lp.yaml` + `phase1/scripts/run_lp_ablation.py` sweep
    p ∈ {1, 2, 4, ∞} across MOT17 pilot sequences.
  - New `phase2/src/phase2/phase3/{collect_oracle,fit_parametric}.py` scaffold
    Stage 2 & Stage 3 of the protocol.
  - PROJECT_STATE §7 rewritten to reference the protocol as source of truth.
  - Next user action: server smoke test (§7.4), then oracle collection.
- **2026-04-27**: Stage A confirmation runs.
  - `smoke_v5` (4 QPs × M0 + M4 × MOT17-04, 5 frames): M4 ≠ M0 at every QP →
    dQP calibration fix verified.
  - `quick_v3` (10 frames, p=∞, MOT17-04-DPM): M4 vs M0 at QPs 27/32/37/42 →
    ΔRate ∈ {−2.6, −0.8, −1.0, −0.4} %, ΔmAP ∈ {−0.001, +0.037, +0.009, −0.028}.
    Calibration fix works but Lp sweep alone gives marginal gains; user
    requested a comprehensive academic review of the data-collection
    methodology.
- **2026-04-29**: pilot_v3 results in (24/24 successful). MOT17-04 shows
  ~46 % bitrate saving at equal mAP=0.846 (M4@QP37 vs M0@QP32). MOT17-09
  near-neutral. MOT17-02 unstable at QP=27 (−5.6 % rate but −0.057 mAP).
  BD-Rate-Task average mixed. Diagnosis: oracle-direct fixed params
  (μ=0.35, Δroi=6, Δbg=4) are not Q-adaptive — at low QP the +4 background
  delta breaks VVC intra-prediction across CTU boundaries.
- **2026-04-29 (late)**: Phase 3 Stage C — LiteQP residual learning.
  Decision: instead of unconstrained CNN/DNN that risks rate violation
  and is hard to defend with reviewers, adopt **analytic prior + bounded
  residual** approach.
  - **`analytic_a_plus.py`** (NEW): closed-form RD-log A+ allocation
    `δ_c = -6·log_2(ξ_c / g)` with `ξ_c = (Φ_c+ε) / (K̃_c+κ)^β`. Q-adaptive
    bounds `(Δ_roi, Δ_bg) = (4..7, 2..4)` linearly in Q_b. Exact closed-form
    rate-neutral projection (not first-order linearisation).
  - **`build_liteqp_dataset.py`** (NEW): per-CTU 9-feature rows, teacher
    label δ_star = arg min over discrete Δ of RD-Lagrangian + Tikhonov
    anchor `λ_s |δ - δ_a+|`. Residual capped at ±2 QP.
  - **`train_liteqp_regressor.py`** (NEW): sklearn MLP-(16,8,1) with
    SiLU-equivalent (ReLU) and bounded output. **Mandatory LOSO CV**
    on 3 sequences for honest held-out evaluation.
  - **`apply_liteqp_model.py`** (NEW): full forward pass A+ → r̂ → exact
    rate-neutral projection → Q-adaptive clip → integer rounding. Also
    supports `--mode a_plus` for ablation (M4-A+ baseline, no MLP).
  - **`encode_pipeline.py`** (UPDATED): `_find_qp_maps` is now Q_base-aware
    and prefers `qp_vtm_delta_QP{n}/` (Stage C per-QP layout) over the
    generic `qp_vtm_delta/` (pilot_v3 layout). Backward compatible.
  - **`run_phase3_liteqp_pipeline.py`** + `phase3_liteqp.yaml` + `pilot_v4.yaml`
    (NEW): end-to-end orchestration; reuses Stage B saliency / rate caches.
  - Next user action: §7.5 (run Stage C pipeline + encode pilot_v4).
- **2026-04-27 (late)**: Phase 3 Stage B implemented.
  - User chose `skip_validation` + `full_hierarchy` → straight to surrogate-driven
    optimization with Levels 1-6.
  - **Occlusion saliency (B.1)** — `phase2/src/phase2/phase3/occlusion_saliency.py`:
    per-CTU Φ_oracle via Zeiler-Fergus 2014 occlusion. Free side-effect: prints
    Spearman ρ between Φ_IPF and Φ_oracle and warns if ρ < 0.30.
  - **Rate surrogate (B.2)** — `rate_surrogate.py`: per-CTU
    `K_c ∝ σ_Y(c)^ρ · (1 + |Δluma|)^ν`, calibrated by regressing pilot_v1
    M0 per-frame bits across QPs (no new VTM run).
  - **Oracle builder (B.3)** — `build_oracle.py`: closed-form prediction of
    (ΔR, ΔmAP) for any (CTU, δ) sample without VTM, producing a JSONL/Parquet
    compatible with fit_parametric.py.
  - **Parametric fitter (B.4)** — `fit_parametric.py` extended with:
    surrogate model wrapping per-CTU K_c and asymmetric ΔmAP, bootstrap CI
    on the BD-Rate-Task proxy, leave-one-sequence-out cross-validation.
  - **Level 6 DNN (B.5)** — `train_dnn.py`: 273-parameter MLP trained with
    Lagrangian + TV smoothness + L2 reg, only invoked if no parametric Level
    meets the pre-registered criterion.
  - **Apply formula (B.6)** — `apply_formula.py`: writes `qp_*.txt` delta maps
    compatible with `phase2.encoding.vtm_encoder`.
  - **Orchestration**: `phase2/scripts/run_phase3_pipeline.py` runs all stages.
    Configs: `phase2/configs/{phase3_fit,pilot_v3}.yaml`.
  - Phase 1 runner now writes `field_*.npy` when `output.save_field_npy=true`
    (default flipped to true in `ipf_lp.yaml`) so Phase 3 can ingest the
    aggregated field.
  - Next user action: §7.5 — re-run Phase 1 with field tensors saved, run
    `run_phase3_pipeline.py`, then encode `pilot_v3.yaml`.
