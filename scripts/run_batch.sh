#!/bin/bash
# =============================================================================
# Run IPF Phase 1 pipeline on all videos in a directory
# =============================================================================
# Usage:
#   bash scripts/run_batch.sh <video_dir> [max_frames] [device]
# Example:
#   bash scripts/run_batch.sh /data/MOT17/train/ 300 cuda:0
# =============================================================================

set -euo pipefail

VIDEO_DIR="${1:?Usage: $0 <video_dir> [max_frames] [device]}"
MAX_FRAMES="${2:-}"
DEVICE="${3:-cuda:0}"

CONFIG="configs/default.yaml"
OUTPUT_DIR="outputs"

CMD="python -m phase1.cli.run batch --config ${CONFIG} --video-dir ${VIDEO_DIR} --device ${DEVICE}"

if [ -n "${MAX_FRAMES}" ]; then
    CMD="${CMD} --max-frames ${MAX_FRAMES}"
fi

echo "=============================================="
echo "IPF Phase 1 — Batch Run"
echo "=============================================="
echo "Video dir:  ${VIDEO_DIR}"
echo "Device:     ${DEVICE}"
echo "Max frames: ${MAX_FRAMES:-all}"
echo "=============================================="

${CMD}
