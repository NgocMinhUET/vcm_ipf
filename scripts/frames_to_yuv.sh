#!/bin/bash
# =============================================================================
# Convert MOT17 image frames to YUV 4:2:0 for VTM encoding
# =============================================================================
# VTM requires raw YUV 4:2:0 input. MOT17 provides JPEG frames.
# This script converts frame sequences to YUV using ffmpeg.
#
# VTM encoding requirements:
#   - Resolution must be multiple of CTU size (typically 128 or 64)
#   - YUV 4:2:0 planar format, 8-bit or 10-bit
#   - Frames in a single concatenated .yuv file
#
# MOT17 resolutions (vary by sequence):
#   MOT17-02: 1920x1080 (aligned to 128-CTU: 1920x1024 after crop)
#   MOT17-04: 1920x1080
#   MOT17-09: 1920x1080
#   MOT17-11: 1920x1080
#   MOT17-13: 1920x1080
#
# Usage:
#   bash scripts/frames_to_yuv.sh <frames_dir> <output_yuv> [width] [height] [max_frames] [fps]
# Example:
#   bash scripts/frames_to_yuv.sh /path/to/MOT17-04/img1 output/MOT17-04.yuv 1920 1080 200 30
# =============================================================================

set -euo pipefail

FRAMES_DIR="${1:?Usage: $0 <frames_dir> <output_yuv> [width] [height] [max_frames] [fps]}"
OUTPUT_YUV="${2:?Usage: $0 <frames_dir> <output_yuv> [width] [height] [max_frames] [fps]}"
WIDTH="${3:-1920}"
HEIGHT="${4:-1080}"
MAX_FRAMES="${5:-0}"
FPS="${6:-30}"

CTU_SIZE=128
ALIGNED_W=$(( (WIDTH + CTU_SIZE - 1) / CTU_SIZE * CTU_SIZE ))
ALIGNED_H=$(( (HEIGHT + CTU_SIZE - 1) / CTU_SIZE * CTU_SIZE ))

echo "======================================================"
echo "Frame-to-YUV Conversion"
echo "======================================================"
echo "Input:      ${FRAMES_DIR}"
echo "Output:     ${OUTPUT_YUV}"
echo "Original:   ${WIDTH}x${HEIGHT}"
echo "CTU-aligned: ${ALIGNED_W}x${ALIGNED_H} (CTU=${CTU_SIZE})"
echo "Max frames: ${MAX_FRAMES} (0 = all)"
echo "FPS:        ${FPS}"
echo "======================================================"

FRAME_COUNT=$(ls -1 "${FRAMES_DIR}"/*.jpg 2>/dev/null | wc -l)
if [ "${FRAME_COUNT}" -eq 0 ]; then
    FRAME_COUNT=$(ls -1 "${FRAMES_DIR}"/*.png 2>/dev/null | wc -l)
fi
echo "Found ${FRAME_COUNT} frames in source directory"

if [ "${FRAME_COUNT}" -eq 0 ]; then
    echo "ERROR: No .jpg or .png frames found in ${FRAMES_DIR}"
    exit 1
fi

OUTPUT_DIR=$(dirname "${OUTPUT_YUV}")
mkdir -p "${OUTPUT_DIR}"

VFRAMES_OPT=""
if [ "${MAX_FRAMES}" -gt 0 ]; then
    VFRAMES_OPT="-vframes ${MAX_FRAMES}"
fi

PATTERN_TYPE=""
FRAME_EXT="jpg"
if ls "${FRAMES_DIR}"/*.jpg >/dev/null 2>&1; then
    FRAME_EXT="jpg"
elif ls "${FRAMES_DIR}"/*.png >/dev/null 2>&1; then
    FRAME_EXT="png"
fi

FIRST_FRAME=$(ls -1 "${FRAMES_DIR}"/*.${FRAME_EXT} | head -1 | xargs basename)
if echo "${FIRST_FRAME}" | grep -qE '^[0-9]{6}\.' ; then
    INPUT_PATTERN="${FRAMES_DIR}/%06d.${FRAME_EXT}"
elif echo "${FIRST_FRAME}" | grep -qE '^[0-9]{4}\.' ; then
    INPUT_PATTERN="${FRAMES_DIR}/%04d.${FRAME_EXT}"
else
    INPUT_PATTERN="${FRAMES_DIR}/%06d.${FRAME_EXT}"
fi

FIRST_NUM=$(echo "${FIRST_FRAME}" | sed 's/\..*//')
START_NUM=$((10#${FIRST_NUM}))

echo "Frame pattern: ${INPUT_PATTERN}"
echo "Starting frame number: ${START_NUM}"
echo ""

ffmpeg -y \
    -start_number "${START_NUM}" \
    -framerate "${FPS}" \
    -i "${INPUT_PATTERN}" \
    ${VFRAMES_OPT} \
    -vf "pad=${ALIGNED_W}:${ALIGNED_H}:0:0:black" \
    -pix_fmt yuv420p \
    -f rawvideo \
    "${OUTPUT_YUV}"

OUTPUT_SIZE=$(stat -c%s "${OUTPUT_YUV}" 2>/dev/null || stat -f%z "${OUTPUT_YUV}")
FRAME_SIZE=$(( ALIGNED_W * ALIGNED_H * 3 / 2 ))
ACTUAL_FRAMES=$(( OUTPUT_SIZE / FRAME_SIZE ))

echo ""
echo "======================================================"
echo "Conversion Complete"
echo "======================================================"
echo "Output:      ${OUTPUT_YUV}"
echo "Resolution:  ${ALIGNED_W}x${ALIGNED_H}"
echo "Frames:      ${ACTUAL_FRAMES}"
echo "File size:   $(( OUTPUT_SIZE / 1024 / 1024 )) MB"
echo ""

META_FILE="${OUTPUT_YUV%.yuv}.meta"
cat > "${META_FILE}" << EOF
source_dir=${FRAMES_DIR}
output_yuv=${OUTPUT_YUV}
width=${ALIGNED_W}
height=${ALIGNED_H}
original_width=${WIDTH}
original_height=${HEIGHT}
chroma_format=420
bit_depth=8
fps=${FPS}
n_frames=${ACTUAL_FRAMES}
ctu_size=${CTU_SIZE}
EOF

echo "Metadata written to: ${META_FILE}"
echo "======================================================"
