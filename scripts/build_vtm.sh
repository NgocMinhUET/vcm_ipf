#!/bin/bash
# =============================================================================
# Build VTM (VVC Test Model) from source
# =============================================================================
# Downloads and builds the JVET VVC reference software (VTM).
# The build produces two key binaries:
#   - EncoderApp: VVC encoder
#   - DecoderApp: VVC decoder
#
# Prerequisites:
#   - cmake >= 3.13
#   - g++ >= 7.0 (C++14 support)
#   - make or ninja
#   - git
#
# Usage:
#   bash scripts/build_vtm.sh [vtm_tag]
# Example:
#   bash scripts/build_vtm.sh VTM-23.4
#   bash scripts/build_vtm.sh              # defaults to VTM-23.4
# =============================================================================

set -euo pipefail

VTM_TAG="${1:-VTM-23.4}"
VTM_REPO="https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git"
INSTALL_DIR="${HOME}/Minh/ipf/vtm"
BUILD_DIR="${INSTALL_DIR}/VVCSoftware_VTM/build"

echo "======================================================"
echo "VTM Build Script"
echo "======================================================"
echo "Tag:         ${VTM_TAG}"
echo "Install dir: ${INSTALL_DIR}"
echo "======================================================"

mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

if [ ! -d "VVCSoftware_VTM" ]; then
    echo "[1/4] Cloning VTM repository..."
    git clone --depth 1 --branch "${VTM_TAG}" "${VTM_REPO}"
else
    echo "[1/4] VTM source already exists, checking tag..."
    cd VVCSoftware_VTM
    CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "unknown")
    if [ "${CURRENT_TAG}" != "${VTM_TAG}" ]; then
        echo "  Current: ${CURRENT_TAG}, requested: ${VTM_TAG}"
        echo "  Fetching and checking out requested tag..."
        git fetch --depth 1 origin "refs/tags/${VTM_TAG}:refs/tags/${VTM_TAG}" 2>/dev/null || true
        git checkout "${VTM_TAG}"
    fi
    cd ..
fi

echo "[2/4] Configuring CMake (Release build)..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}/install"

echo "[3/4] Building (this may take 10-20 minutes)..."
NPROC=$(nproc 2>/dev/null || echo 4)
make -j"${NPROC}"

echo "[4/4] Verifying build..."
# VTM cmake places binaries in build/bin/umake/<compiler>/<arch>/release/
# Use find to locate them robustly regardless of gcc version string.
ENCODER=$(find "${BUILD_DIR}" -name "EncoderApp" -type f 2>/dev/null | sort | tail -1)
DECODER=$(find "${BUILD_DIR}" -name "DecoderApp" -type f 2>/dev/null | sort | tail -1)

if [ -z "${ENCODER}" ] || [ ! -f "${ENCODER}" ]; then
    echo "ERROR: EncoderApp not found under ${BUILD_DIR}"
    echo "Expected location: ${BUILD_DIR}/bin/umake/<gcc-version>/x86_64/release/EncoderApp"
    exit 1
fi
if [ -z "${DECODER}" ] || [ ! -f "${DECODER}" ]; then
    echo "ERROR: DecoderApp not found under ${BUILD_DIR}"
    exit 1
fi

echo ""
echo "======================================================"
echo "VTM Build Successful!"
echo "======================================================"
echo "Encoder: ${ENCODER}"
echo "Decoder: ${DECODER}"
echo ""
echo "Version check:"
"${ENCODER}" --help 2>&1 | head -5 || true
echo ""
echo "To use with IPF Phase 2, set in your config:"
echo "  encoder_path: ${ENCODER}"
echo "  decoder_path: ${DECODER}"
echo "======================================================"
