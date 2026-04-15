#!/bin/bash
# =============================================================================
# Batch YUV Conversion for Phase 2 Pilot Sequences
# =============================================================================
# Converts all 3 MOT17 sequences used in pilot.yaml to raw YUV 4:2:0 format.
#
# Usage:
#   bash scripts/prepare_yuv.sh [MOT17_ROOT] [OUTPUT_DIR] [MAX_FRAMES]
# Defaults:
#   MOT17_ROOT  = ~/Minh/ipf/datasets/MOT17
#   OUTPUT_DIR  = ~/Minh/ipf/datasets/yuv
#   MAX_FRAMES  = 200
# =============================================================================

set -euo pipefail

MOT17_ROOT="${1:-${HOME}/Minh/ipf/datasets/MOT17}"
OUTPUT_DIR="${2:-${HOME}/Minh/ipf/datasets/yuv}"
MAX_FRAMES="${3:-200}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONVERTER="${SCRIPT_DIR}/frames_to_yuv.sh"

mkdir -p "${OUTPUT_DIR}"

# MOT17 extraction layout: <MOT17_ROOT>/MOT17/train/<seq>/img1/
# Also check <MOT17_ROOT>/<seq>/img1/ as a direct fallback.
MOT17_TRAIN="${MOT17_ROOT}/MOT17/train"

# Pilot sequences from pilot.yaml
declare -A SEQUENCES=(
    ["MOT17-04-DPM"]="1920 1080"
    ["MOT17-02-DPM"]="1920 1080"
    ["MOT17-09-DPM"]="1920 1080"
)

echo "======================================================"
echo "Batch YUV Conversion — Phase 2 Pilot Sequences"
echo "======================================================"
echo "MOT17 root:  ${MOT17_ROOT}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Max frames:  ${MAX_FRAMES}"
echo "======================================================"

SUCCESS=0
FAILED=0
SKIPPED=0

for SEQ in "${!SEQUENCES[@]}"; do
    WH="${SEQUENCES[$SEQ]}"
    W="${WH%% *}"
    H="${WH##* }"
    OUTPUT_YUV="${OUTPUT_DIR}/${SEQ}.yuv"

    echo ""
    echo "------------------------------------------------------"
    echo "Sequence: ${SEQ}  (${W}x${H})"
    echo "------------------------------------------------------"

    # Resolve frame directory: try standard MOT17 layout first, then flat layout.
    FRAMES_DIR="${MOT17_TRAIN}/${SEQ}/img1"
    if [ ! -d "${FRAMES_DIR}" ]; then
        FRAMES_DIR="${MOT17_ROOT}/${SEQ}/img1"
    fi
    if [ ! -d "${FRAMES_DIR}" ]; then
        echo "  SKIP: frames dir not found."
        echo "        Tried: ${MOT17_TRAIN}/${SEQ}/img1"
        echo "        Tried: ${MOT17_ROOT}/${SEQ}/img1"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if [ -f "${OUTPUT_YUV}" ]; then
        META="${OUTPUT_YUV%.yuv}.meta"
        if [ -f "${META}" ]; then
            N=$(grep "^n_frames=" "${META}" | cut -d= -f2)
            if [ "${N:-0}" -ge "${MAX_FRAMES}" ]; then
                echo "  SKIP: ${OUTPUT_YUV} already exists with ${N} frames."
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
        fi
    fi

    bash "${CONVERTER}" "${FRAMES_DIR}" "${OUTPUT_YUV}" "${W}" "${H}" "${MAX_FRAMES}" 30
    if [ $? -eq 0 ]; then
        echo "  OK: ${OUTPUT_YUV}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  FAILED: ${SEQ}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "======================================================"
echo "Batch conversion done."
echo "  Success: ${SUCCESS}   Skipped: ${SKIPPED}   Failed: ${FAILED}"
echo "======================================================"

if [ "${FAILED}" -gt 0 ]; then
    echo "WARNING: ${FAILED} sequence(s) failed. Check output above."
    exit 1
fi
