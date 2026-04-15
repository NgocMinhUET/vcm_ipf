#!/bin/bash
# =============================================================================
# Phase 2 Environment Setup
# =============================================================================
# Sets up the complete environment for VTM integration experiments.
# Run this ONCE on the server before starting Phase 2 experiments.
#
# Usage:
#   bash scripts/setup_env.sh
# =============================================================================

set -euo pipefail

BASE_DIR="${HOME}/Minh/ipf"
PHASE2_DIR="${BASE_DIR}/phase2"
VTM_DIR="${BASE_DIR}/vtm"
DATA_DIR="${BASE_DIR}/datasets"
YUV_DIR="${DATA_DIR}/yuv"

echo "======================================================"
echo "Phase 2 Environment Setup"
echo "======================================================"

echo "[1/6] Creating directory structure..."
mkdir -p "${PHASE2_DIR}"
mkdir -p "${VTM_DIR}"
mkdir -p "${YUV_DIR}"
mkdir -p "${BASE_DIR}/phase2_outputs"

echo "[2/6] Checking system prerequisites..."
echo "  Python:  $(python3 --version 2>&1 || echo 'NOT FOUND')"
echo "  CMake:   $(cmake --version 2>&1 | head -1 || echo 'NOT FOUND')"
echo "  g++:     $(g++ --version 2>&1 | head -1 || echo 'NOT FOUND')"
echo "  ffmpeg:  $(ffmpeg -version 2>&1 | head -1 || echo 'NOT FOUND')"
echo "  git:     $(git --version 2>&1 || echo 'NOT FOUND')"

echo "[3/6] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
    numpy \
    scipy \
    pandas \
    matplotlib \
    pyyaml \
    pydantic \
    typer \
    rich \
    tqdm \
    ultralytics \
    opencv-python-headless

echo "[4/6] Building VTM..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "${VTM_DIR}/VVCSoftware_VTM/build/source/App/EncoderApp/EncoderApp" ]; then
    bash "${SCRIPT_DIR}/build_vtm.sh"
else
    echo "  VTM already built, skipping..."
fi

echo "[5/6] Converting MOT17 sequences to YUV..."
MOT17_DIR="${DATA_DIR}/MOT17"
SEQUENCES=("MOT17-02-DPM" "MOT17-04-DPM" "MOT17-09-DPM" "MOT17-11-DPM" "MOT17-13-DPM")

for SEQ in "${SEQUENCES[@]}"; do
    OUTPUT="${YUV_DIR}/${SEQ}.yuv"
    if [ -f "${OUTPUT}" ]; then
        echo "  ${SEQ}.yuv already exists, skipping..."
        continue
    fi
    FRAMES_DIR="${MOT17_DIR}/${SEQ}/img1"
    if [ -d "${FRAMES_DIR}" ]; then
        echo "  Converting ${SEQ}..."
        bash "${SCRIPT_DIR}/frames_to_yuv.sh" \
            "${FRAMES_DIR}" "${OUTPUT}" 1920 1080 200 30
    else
        echo "  WARNING: ${FRAMES_DIR} not found, skipping ${SEQ}"
    fi
done

echo "[6/6] Writing environment manifest..."
ENV_MANIFEST="${PHASE2_DIR}/env_manifest.md"
cat > "${ENV_MANIFEST}" << MANIFEST
# Phase 2 Environment Manifest

## Date
$(date -Iseconds)

## System
- OS: $(uname -srm)
- CPU: $(nproc) cores
- RAM: $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo 'unknown')
- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')

## Software Versions
- Python: $(python3 --version 2>&1)
- CMake: $(cmake --version 2>&1 | head -1)
- g++: $(g++ --version 2>&1 | head -1)
- ffmpeg: $(ffmpeg -version 2>&1 | head -1)
- CUDA: $(nvcc --version 2>&1 | tail -1 || echo 'N/A')

## VTM
- Tag: VTM-23.4
- Encoder: ${VTM_DIR}/VVCSoftware_VTM/build/source/App/EncoderApp/EncoderApp
- Decoder: ${VTM_DIR}/VVCSoftware_VTM/build/source/App/DecoderApp/DecoderApp

## Directories
- Phase 1: ${BASE_DIR}/phase1
- Phase 2: ${PHASE2_DIR}
- VTM:     ${VTM_DIR}
- Data:    ${DATA_DIR}
- YUV:     ${YUV_DIR}
- Outputs: ${BASE_DIR}/phase2_outputs

## Python Packages
$(pip list 2>/dev/null | grep -E "numpy|scipy|pandas|matplotlib|pyyaml|pydantic|typer|rich|tqdm|ultralytics|opencv" || echo "unable to list")
MANIFEST

echo ""
echo "======================================================"
echo "Phase 2 Environment Setup Complete"
echo "======================================================"
echo "Environment manifest: ${ENV_MANIFEST}"
echo ""
echo "Next steps:"
echo "  1. Apply VTM patch:  bash scripts/apply_vtm_patch.sh"
echo "  2. Run pilot:        bash scripts/run_pilot.sh"
echo "======================================================"
