#!/bin/bash
# =============================================================================
# Run comp_v4: IPF v2 (max superposition) + all baselines + ablation
# =============================================================================
# This is the DEFINITIVE Phase-1 comparison using the improved IPF v2 config.
# Key change: M4 now uses max superposition instead of sum.
#
# Usage:
#   bash scripts/run_comp_v4.sh [device]
# Example:
#   bash scripts/run_comp_v4.sh cpu
#   bash scripts/run_comp_v4.sh cuda:0
# =============================================================================

set -euo pipefail

DEVICE="${1:-cpu}"
VIDEO="/home/guest/Minh/ipf/datasets/MOT17/MOT17-04-DPM.mp4"
CONFIG="configs/ipf_v2.yaml"
RUN_ID="comp_v4"
METHODS="M0,M1,M4,M5,M6,M7,M8"
MAX_FRAMES=100

echo "=============================================="
echo "IPF Phase-1 Comparison — comp_v4 (IPF v2)"
echo "=============================================="
echo "Config:     ${CONFIG} (max superposition)"
echo "Video:      ${VIDEO}"
echo "Methods:    ${METHODS} + ablation variants"
echo "Run ID:     ${RUN_ID}"
echo "Device:     ${DEVICE}"
echo "Max frames: ${MAX_FRAMES}"
echo "=============================================="

python -m phase1.cli.compare \
    --config "${CONFIG}" \
    --video "${VIDEO}" \
    --run-id "${RUN_ID}" \
    --device "${DEVICE}" \
    --methods "${METHODS}" \
    --max-frames "${MAX_FRAMES}" \
    --warmup-skip 10 \
    --ablation

echo ""
echo "=============================================="
echo "Done. Results at: ~/ipf_outputs/${RUN_ID}/"
echo "  comparison_table_full.txt   — full sequence metrics"
echo "  comparison_table_steady.txt — steady-state metrics (warmup excluded)"
echo "  comparison_summary.json     — machine-readable metrics"
echo "=============================================="
