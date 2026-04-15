#!/bin/bash
# =============================================================================
# Phase 2 Full Pipeline — One-Stop Execution Script
# =============================================================================
# Runs all Phase 2 steps in order:
#   Step 1: Apply VTM patch (ExternalQPMapDir)
#   Step 2: Run Phase 1b multi-sequence (generate QP maps for MOT17-02/04/09)
#   Step 3: Convert MOT17 sequences to YUV 4:2:0
#   Step 4: Run Phase 2 pilot encoding (4 QP × 5 methods × 3 sequences)
#   Step 5: BD-Rate analysis and paper-ready tables
#
# Usage:
#   bash scripts/run_phase2_full.sh [--skip-patch] [--skip-phase1b] [--skip-yuv]
#
# Prerequisites (must already exist):
#   - VTM built at ~/Minh/ipf/vtm/VVCSoftware_VTM
#   - Phase 1 code at ~/Minh/ipf/phase1 with ipf_v2.yaml config
#   - MOT17 dataset at ~/Minh/ipf/datasets/MOT17
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PHASE1_DIR="${HOME}/Minh/ipf/phase1"

SKIP_PATCH=0
SKIP_PHASE1B=0
SKIP_YUV=0
DEVICE="${DEVICE:-cuda:0}"

for arg in "$@"; do
    case "${arg}" in
        --skip-patch)   SKIP_PATCH=1 ;;
        --skip-phase1b) SKIP_PHASE1B=1 ;;
        --skip-yuv)     SKIP_YUV=1 ;;
    esac
done

log() { echo ""; echo "======================================================"; echo "$1"; echo "======================================================"; }

# ---------------------------------------------------------------------------
log "[STEP 1/5] Apply VTM ExternalQPMapDir Patch"
# ---------------------------------------------------------------------------
if [ "${SKIP_PATCH}" -eq 1 ]; then
    echo "  Skipped (--skip-patch)"
else
    cd "${PHASE2_DIR}"
    bash scripts/apply_vtm_patch.sh

    # Verify --ExternalQPMapDir flag exists
    ENCODER=$(find "${HOME}/Minh/ipf/vtm/VVCSoftware_VTM" -name "EncoderApp" -type f 2>/dev/null | sort | tail -1)
    if "${ENCODER}" --help 2>&1 | grep -qi "ExternalQPMapDir"; then
        echo "  VTM patch verified: --ExternalQPMapDir flag present."
    else
        echo "  WARNING: --ExternalQPMapDir not found in encoder help."
        echo "  EncAppCfg.cpp patch may have failed silently."
        echo "  Encoding will still work via g_ipfQPReader init from compressSlice."
    fi
fi

# ---------------------------------------------------------------------------
log "[STEP 2/5] Phase 1b — Generate QP Maps for 3 Sequences"
# ---------------------------------------------------------------------------
if [ "${SKIP_PHASE1B}" -eq 1 ]; then
    echo "  Skipped (--skip-phase1b)"
else
    if [ ! -d "${PHASE1_DIR}" ]; then
        echo "ERROR: Phase 1 directory not found at ${PHASE1_DIR}"
        echo "Please clone/pull phase1 branch first."
        exit 1
    fi
    cd "${PHASE1_DIR}"
    bash scripts/run_multi_sequence.sh "${DEVICE}" 200
    echo "  Phase 1b complete. QP maps in ~/ipf_outputs/multi_seq_*/"
fi

# ---------------------------------------------------------------------------
log "[STEP 3/5] Convert MOT17 Sequences to YUV 4:2:0"
# ---------------------------------------------------------------------------
if [ "${SKIP_YUV}" -eq 1 ]; then
    echo "  Skipped (--skip-yuv)"
else
    cd "${PHASE2_DIR}"
    bash scripts/prepare_yuv.sh \
        "${HOME}/Minh/ipf/datasets/MOT17" \
        "${HOME}/Minh/ipf/datasets/yuv" \
        200
fi

# ---------------------------------------------------------------------------
log "[STEP 4/5] Phase 2 Pilot Encoding (60 runs)"
# ---------------------------------------------------------------------------
cd "${PHASE2_DIR}"
export PYTHONPATH="${PHASE2_DIR}/src:${PYTHONPATH:-}"

echo "  Starting pilot: 4 QP × 5 methods × 3 sequences = 60 VTM encodes"
echo "  Estimated time: 2-6 hours depending on server speed"
echo "  Log: ${PHASE2_DIR}/pilot_run.log"
echo ""

bash scripts/run_pilot.sh configs/pilot.yaml

# ---------------------------------------------------------------------------
log "[STEP 5/5] BD-Rate Analysis and Paper Tables"
# ---------------------------------------------------------------------------
PILOT_OUT="${HOME}/Minh/ipf/phase2_outputs/pilot_v1"

if [ ! -d "${PILOT_OUT}" ]; then
    echo "ERROR: Pilot output not found at ${PILOT_OUT}"
    exit 1
fi

python -m phase2.analysis.results_aggregator \
    --experiment-dir "${PILOT_OUT}" \
    --output-dir "${PILOT_OUT}/analysis"

echo ""
echo "======================================================"
echo "Phase 2 Complete!"
echo "======================================================"
echo "Pilot results:  ${PILOT_OUT}/"
echo "BD-Rate tables: ${PILOT_OUT}/analysis/"
echo ""
echo "Download results:"
echo "  scp -r user@server:${PILOT_OUT}/analysis/ ./phase2_results/"
echo "======================================================"
