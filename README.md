# IPF Phase 2: VTM/VVC Integration

Phase 2 integrates Phase 1's QP map generation with actual VVC encoding
using VTM (VVC Test Model), enabling real rate-distortion evaluation
with BD-Rate, PSNR, and task accuracy (mAP) metrics.

## Directory Structure

```
phase2/
  src/phase2/
    core/config.py           # Phase 2 Pydantic configuration
    encoding/
      vtm_encoder.py         # VTM encoder wrapper
      vtm_decoder.py         # VTM decoder wrapper
      yuv_utils.py           # YUV 4:2:0 I/O utilities
    evaluation/
      psnr.py                # PSNR (full-frame + ROI regions)
      task_accuracy.py       # YOLOv8 mAP on decoded frames
      bd_rate.py             # Bjontegaard Delta metrics
    pipeline/
      encode_pipeline.py     # Full encode-decode-evaluate orchestrator
    analysis/
      results_aggregator.py  # Cross-method/sequence BD-Rate aggregation
      paper_figures.py       # Publication-quality figure generation
  configs/
    pilot.yaml               # Pilot experiment configuration
  scripts/
    build_vtm.sh             # Build VTM from source
    frames_to_yuv.sh         # Convert image frames to YUV 4:2:0
    apply_vtm_patch.sh       # Apply external QP map patch to VTM
    verify_compliance.sh     # Verify bitstream standard compliance
    setup_env.sh             # Full environment setup
    run_pilot.sh             # Run pilot experiment
    sync_to_server.sh        # Sync code to server
  vtm_patch/
    external_qp_reader.h     # C++ QP map reader for VTM
    README.md                # Patch documentation
```

## Quick Start (on server)

```bash
# 1. Sync code to server
bash scripts/sync_to_server.sh

# 2. On server: setup environment (builds VTM, converts YUV)
ssh guest@100.104.64.97
cd ~/Minh/ipf/phase2
bash scripts/setup_env.sh

# 3. Apply VTM patch
bash scripts/apply_vtm_patch.sh

# 4. Run Phase 1 multi-sequence first (QP maps needed)
cd ~/Minh/ipf/phase1
bash scripts/run_multi_sequence.sh cuda:0 200

# 5. Run Phase 2 pilot experiment
cd ~/Minh/ipf/phase2
bash scripts/run_pilot.sh

# 6. Analyze results
python -m phase2.analysis.results_aggregator \
    --experiment-dir ~/Minh/ipf/phase2_outputs/pilot_v1

# 7. Generate paper figures
python -m phase2.analysis.paper_figures \
    --experiment-dir ~/Minh/ipf/phase2_outputs/pilot_v1
```

## Experiment Matrix (Pilot)

| Parameter     | Values                          |
|---------------|----------------------------------|
| QP base       | 22, 27, 32, 37                  |
| Methods       | M0, M1, M4 (IPF v2), M5, M6    |
| Sequences     | MOT17-02, MOT17-04, MOT17-09   |
| Total runs    | 4 x 5 x 3 = 60                 |

## Metrics

| Metric          | Description                              |
|-----------------|------------------------------------------|
| BD-Rate (PSNR)  | Bitrate saving at equal full-frame PSNR  |
| BD-Rate (ROI)   | Bitrate saving at equal ROI PSNR         |
| BD-Rate (Task)  | Bitrate saving at equal mAP             |
| BD-PSNR         | Quality gain at equal bitrate           |
| BD-Task         | mAP gain at equal bitrate               |

## KPI Targets (from Project Charter)

- KPI-1: >= 5% BD-Rate reduction vs M1 (binary ROI)
- KPI-2: >= 10% temporal QP variance reduction (Phase 1)
- KPI-3: <= 20% encoding-time overhead
- KPI-4: Statistical significance with p-value and effect size
