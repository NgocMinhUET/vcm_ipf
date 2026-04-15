#!/bin/bash
# =============================================================================
# Phase 1b: Multi-Sequence Validation for IPF v2
# =============================================================================
# Runs IPF v2 comparison (comp_v4 config) on multiple MOT17 sequences to
# provide cross-sequence statistical evidence.
#
# Sequences selected for diversity:
#   MOT17-02-DPM: Static camera, moderate density (~25 pedestrians)
#   MOT17-04-DPM: Static camera, high density (~80 pedestrians) [already done]
#   MOT17-09-DPM: Static camera, low density (~10 pedestrians)
#   MOT17-11-DPM: Moving camera, moderate density (~20 pedestrians)
#   MOT17-13-DPM: Moving camera, high density (~60 pedestrians)
#
# Usage:
#   bash scripts/run_multi_sequence.sh [device] [max_frames]
# Examples:
#   bash scripts/run_multi_sequence.sh cuda:0 200
#   bash scripts/run_multi_sequence.sh cpu 100
# =============================================================================

set -euo pipefail

DEVICE="${1:-cpu}"
MAX_FRAMES="${2:-200}"
CONFIG="configs/ipf_v2.yaml"
METHODS="M0,M1,M4,M5,M6,M7,M8"
WARMUP_SKIP=10
DATA_DIR="/home/guest/Minh/ipf/datasets/MOT17"

SEQUENCES=(
    "MOT17-02-DPM"
    "MOT17-04-DPM"
    "MOT17-09-DPM"
    "MOT17-11-DPM"
    "MOT17-13-DPM"
)

echo "======================================================"
echo "Phase 1b: Multi-Sequence Validation (IPF v2)"
echo "======================================================"
echo "Config:       ${CONFIG}"
echo "Methods:      ${METHODS} + ablation"
echo "Device:       ${DEVICE}"
echo "Max frames:   ${MAX_FRAMES}"
echo "Warmup skip:  ${WARMUP_SKIP}"
echo "Sequences:    ${#SEQUENCES[@]}"
for SEQ in "${SEQUENCES[@]}"; do
    echo "  - ${SEQ}"
done
echo "======================================================"
echo ""

FAILED=()
SUCCEEDED=()

for SEQ in "${SEQUENCES[@]}"; do
    VIDEO="${DATA_DIR}/${SEQ}.mp4"
    RUN_ID="multi_seq_${SEQ}"

    echo "------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] Starting: ${SEQ}"
    echo "------------------------------------------------------"

    if [ ! -f "${VIDEO}" ]; then
        echo "WARNING: Video not found: ${VIDEO}"
        echo "  Trying frame directory: ${DATA_DIR}/${SEQ}/img1/"
        VIDEO="${DATA_DIR}/${SEQ}/img1/"
        if [ ! -d "${VIDEO}" ]; then
            echo "ERROR: Neither .mp4 nor frame directory found for ${SEQ}"
            FAILED+=("${SEQ}")
            continue
        fi
    fi

    if python -m phase1.cli.compare \
        --config "${CONFIG}" \
        --video "${VIDEO}" \
        --run-id "${RUN_ID}" \
        --device "${DEVICE}" \
        --methods "${METHODS}" \
        --max-frames "${MAX_FRAMES}" \
        --warmup-skip "${WARMUP_SKIP}" \
        --ablation; then
        echo "[$(date +%H:%M:%S)] Completed: ${SEQ}"
        SUCCEEDED+=("${SEQ}")
    else
        echo "[$(date +%H:%M:%S)] FAILED: ${SEQ}"
        FAILED+=("${SEQ}")
    fi
    echo ""
done

echo "======================================================"
echo "Multi-Sequence Run Complete"
echo "======================================================"
echo "Succeeded: ${#SUCCEEDED[@]}/${#SEQUENCES[@]}"
for S in "${SUCCEEDED[@]}"; do
    echo "  [OK] ${S}"
done
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed: ${#FAILED[@]}/${#SEQUENCES[@]}"
    for F in "${FAILED[@]}"; do
        echo "  [FAIL] ${F}"
    done
fi
echo ""
echo "Results at: ~/ipf_outputs/multi_seq_*/"
echo ""
echo "Next step: run cross-sequence analysis"
echo "  python -m phase1.analysis.cross_sequence_stats \\"
echo "    --output-dir ~/ipf_outputs \\"
echo "    --run-prefix multi_seq_"
echo "======================================================"
