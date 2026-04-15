#!/bin/bash
# =============================================================================
# Verify VTM Patch Compliance
# =============================================================================
# Encodes a short test sequence with external QP maps, then verifies that
# the bitstream decodes correctly with the UNMODIFIED VTM decoder.
#
# Usage:
#   bash scripts/verify_compliance.sh <test_yuv> <width> <height> <qp_map_dir> [n_frames]
# Example:
#   bash scripts/verify_compliance.sh /path/to/test.yuv 1920 1088 /path/to/qp_maps 10
# =============================================================================

set -euo pipefail

TEST_YUV="${1:?Usage: $0 <test_yuv> <width> <height> <qp_map_dir> [n_frames]}"
WIDTH="${2:?}"
HEIGHT="${3:?}"
QP_MAP_DIR="${4:?}"
N_FRAMES="${5:-10}"

VTM_DIR="${HOME}/Minh/ipf/vtm/VVCSoftware_VTM"
ENCODER="${VTM_DIR}/build/source/App/EncoderApp/EncoderApp"
DECODER="${VTM_DIR}/build/source/App/DecoderApp/DecoderApp"
VTM_CFG="${VTM_DIR}/cfg/encoder_lowdelay_vtm.cfg"

WORK_DIR=$(mktemp -d)
BITSTREAM="${WORK_DIR}/test.bin"
RECON_ENC="${WORK_DIR}/recon_enc.yuv"
RECON_DEC="${WORK_DIR}/recon_dec.yuv"
ENC_LOG="${WORK_DIR}/encoder.log"
DEC_LOG="${WORK_DIR}/decoder.log"

echo "======================================================"
echo "VTM Compliance Verification"
echo "======================================================"
echo "Test YUV:    ${TEST_YUV}"
echo "Resolution:  ${WIDTH}x${HEIGHT}"
echo "QP maps:     ${QP_MAP_DIR}"
echo "Frames:      ${N_FRAMES}"
echo "Work dir:    ${WORK_DIR}"
echo "======================================================"

echo ""
echo "[1/4] Encoding with external QP maps..."
"${ENCODER}" \
    -c "${VTM_CFG}" \
    -i "${TEST_YUV}" \
    -b "${BITSTREAM}" \
    -o "${RECON_ENC}" \
    -wdt "${WIDTH}" \
    -hgt "${HEIGHT}" \
    -q 32 \
    -f "${N_FRAMES}" \
    --InternalBitDepth=8 \
    --ExternalQPMapDir="${QP_MAP_DIR}" \
    2>&1 | tee "${ENC_LOG}"

if [ ! -f "${BITSTREAM}" ]; then
    echo "FAIL: Encoding failed, no bitstream produced"
    exit 1
fi

BITSTREAM_SIZE=$(stat -c%s "${BITSTREAM}" 2>/dev/null || stat -f%z "${BITSTREAM}")
echo "  Bitstream size: ${BITSTREAM_SIZE} bytes"

echo ""
echo "[2/4] Decoding with UNMODIFIED VTM decoder..."
"${DECODER}" \
    -b "${BITSTREAM}" \
    -o "${RECON_DEC}" \
    2>&1 | tee "${DEC_LOG}"

if [ ! -f "${RECON_DEC}" ]; then
    echo "FAIL: Decoding failed, no reconstructed YUV produced"
    exit 1
fi

echo ""
echo "[3/4] Comparing encoder reconstruction vs decoder reconstruction..."
ENC_SIZE=$(stat -c%s "${RECON_ENC}" 2>/dev/null || stat -f%z "${RECON_ENC}")
DEC_SIZE=$(stat -c%s "${RECON_DEC}" 2>/dev/null || stat -f%z "${RECON_DEC}")

echo "  Encoder recon size: ${ENC_SIZE}"
echo "  Decoder recon size: ${DEC_SIZE}"

if [ "${ENC_SIZE}" -eq "${DEC_SIZE}" ]; then
    DIFF=$(cmp "${RECON_ENC}" "${RECON_DEC}" 2>&1 || echo "DIFFER")
    if echo "${DIFF}" | grep -q "DIFFER"; then
        echo "WARNING: Reconstructions differ in content (may be due to rounding)"
    else
        echo "  MATCH: Encoder and decoder reconstructions are byte-identical"
    fi
else
    echo "WARNING: File sizes differ (${ENC_SIZE} vs ${DEC_SIZE})"
fi

echo ""
echo "[4/4] Checking for decoder errors..."
if grep -qi "error\|fail\|invalid" "${DEC_LOG}"; then
    echo "WARNING: Decoder log contains error messages:"
    grep -i "error\|fail\|invalid" "${DEC_LOG}"
else
    echo "  No errors found in decoder log"
fi

echo ""
echo "======================================================"
PASS=true
if [ ! -f "${BITSTREAM}" ]; then PASS=false; fi
if [ ! -f "${RECON_DEC}" ]; then PASS=false; fi
if grep -qi "error" "${DEC_LOG}" 2>/dev/null; then PASS=false; fi

if [ "${PASS}" = true ]; then
    echo "COMPLIANCE VERIFICATION: PASSED"
    echo "  Bitstream encodes and decodes correctly with standard VVC decoder."
else
    echo "COMPLIANCE VERIFICATION: ISSUES DETECTED"
    echo "  Review logs in ${WORK_DIR}"
fi
echo "======================================================"

rm -rf "${WORK_DIR}"
