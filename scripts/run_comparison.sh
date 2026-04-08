#!/bin/bash
# =============================================================================
# Run multi-method comparison on a video
# =============================================================================
# Usage:
#   bash scripts/run_comparison.sh <video_path> [run_id] [max_frames] [device]
# Example:
#   bash scripts/run_comparison.sh data/mot17_04.mp4 comp_mot17_04 100 cpu
# =============================================================================

set -euo pipefail

VIDEO="${1:?Usage: $0 <video_path> [run_id] [max_frames] [device]}"
RUN_ID="${2:-comp_$(date +%Y%m%d_%H%M%S)}"
MAX_FRAMES="${3:-}"
DEVICE="${4:-cpu}"

CONFIG="configs/default.yaml"
METHODS="M0,M1,M4,M5,M6,M7,M8"

CMD="python -m phase1.cli.compare run --config ${CONFIG} --video ${VIDEO} --run-id ${RUN_ID} --device ${DEVICE} --methods ${METHODS}"

if [ -n "${MAX_FRAMES}" ]; then
    CMD="${CMD} --max-frames ${MAX_FRAMES}"
fi

echo "=============================================="
echo "IPF Multi-Method Comparison"
echo "=============================================="
echo "Video:      ${VIDEO}"
echo "Methods:    ${METHODS}"
echo "Run ID:     ${RUN_ID}"
echo "Device:     ${DEVICE}"
echo "Max frames: ${MAX_FRAMES:-all}"
echo "=============================================="

${CMD}

echo ""
echo "Results at: ~/ipf_outputs/${RUN_ID}/"
echo "  - comparison_table.txt    (readable table)"
echo "  - comparison_summary.json (machine-readable)"
echo "  - M*/qp_vtm/              (per-method QP maps)"
