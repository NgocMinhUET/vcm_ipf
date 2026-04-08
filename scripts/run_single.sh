#!/bin/bash
# =============================================================================
# Run IPF Phase 1 pipeline on a single video
# =============================================================================
# Usage:
#   bash scripts/run_single.sh <video_path> [run_id] [max_frames] [device]
# Example:
#   bash scripts/run_single.sh /data/MOT17/train/MOT17-02/video.mp4 mot17_02 100 cuda:0
# =============================================================================

set -euo pipefail

VIDEO="${1:?Usage: $0 <video_path> [run_id] [max_frames] [device]}"
RUN_ID="${2:-run_$(date +%Y%m%d_%H%M%S)}"
MAX_FRAMES="${3:-}"
DEVICE="${4:-cuda:0}"

CONFIG="configs/default.yaml"
OUTPUT_DIR="outputs"

CMD="python -m phase1.cli.run single --config ${CONFIG} --video ${VIDEO} --run-id ${RUN_ID} --device ${DEVICE}"

if [ -n "${MAX_FRAMES}" ]; then
    CMD="${CMD} --max-frames ${MAX_FRAMES}"
fi

echo "=============================================="
echo "IPF Phase 1 — Single Video Run"
echo "=============================================="
echo "Video:      ${VIDEO}"
echo "Run ID:     ${RUN_ID}"
echo "Device:     ${DEVICE}"
echo "Max frames: ${MAX_FRAMES:-all}"
echo "Config:     ${CONFIG}"
echo "Output:     ${OUTPUT_DIR}/${RUN_ID}"
echo "=============================================="

${CMD}

echo ""
echo "Run complete. Check outputs at: ${OUTPUT_DIR}/${RUN_ID}"
