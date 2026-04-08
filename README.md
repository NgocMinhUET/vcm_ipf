# IPF Phase 1 — Server-Side Precompute Prototype

**Tracker-Informed Spatiotemporal Importance Field with Bounded QP Dynamics**
for ROI-aware VVC Coding for Machine Vision

## Quick Start

```bash
# 1. Create environment
conda create -n ipf python=3.10 -y
conda activate ipf

# 2. Install dependencies
pip install -e .

# 3. Run on a single video
ipf-run single --config configs/default.yaml --video /path/to/video.mp4 --output-dir outputs/run_001

# 4. Run on a dataset directory
ipf-run batch --config configs/default.yaml --video-dir /path/to/videos/ --output-dir outputs/batch_001

# 5. Generate visualizations
ipf-viz field-maps --run-dir outputs/run_001
```

## Project Structure

```
phase1/
├── configs/             # YAML configuration files
├── src/phase1/          # Main source code
│   ├── core/            # Schemas, config, constants
│   ├── io/              # Video/frame loading
│   ├── tracking/        # Detector + tracker wrapper
│   ├── field/           # IPF computation (mass, kernel, superposition)
│   ├── control/         # Temporal normalization, QP mapping, bounded dynamics
│   ├── export/          # QP map export, state serialization
│   ├── viz/             # Visualization utilities
│   ├── pipeline/        # End-to-end pipeline orchestration
│   ├── utils/           # Logging, timing, helpers
│   └── cli/             # Typer CLI entry points
├── scripts/             # Bash helper scripts
├── tests/               # Unit and integration tests
└── mock_data/           # Synthetic test inputs
```

## Phase 1 Goal

Validate the full IPF pipeline **without VTM integration**:
1. Detect + track objects in video
2. Compute importance field per frame
3. Generate CTU-level QP maps
4. Export QP maps for future VTM injection
5. Produce diagnostic visualizations
