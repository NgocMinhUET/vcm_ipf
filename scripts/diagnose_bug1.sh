#!/bin/bash
# Diagnose BUG #1: M0=M1=M4=M5=M6 (external QP map not active).
#
# Run this on the server:
#     bash scripts/diagnose_bug1.sh

set -u

VTM_DIR="${HOME}/Minh/ipf/vtm/VVCSoftware_VTM"
ENC="${VTM_DIR}/bin/umake/gcc-11.4/x86_64/release/EncoderApp"
PHASE1_OUT="${HOME}/ipf_outputs"

print_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# ---------------------------------------------------------------------
print_header "1. Phase 1 QP map directories"
# ---------------------------------------------------------------------
# Each method must have one .txt or .bin per encoded frame in qp_vtm/.
ANY_MISSING=0
for seq in MOT17-04-DPM MOT17-02-DPM MOT17-09-DPM; do
    for m in M1 M4 M5 M6; do
        d="${PHASE1_OUT}/multi_seq_${seq}/${m}/qp_vtm"
        if [ -d "$d" ]; then
            n=$(ls "$d" 2>/dev/null | wc -l)
            example=$(ls "$d" 2>/dev/null | head -n 1)
            echo "  [OK] $seq/$m: $n files in $d  (example: $example)"
        else
            echo "  [MISSING] $seq/$m: directory does not exist: $d"
            ANY_MISSING=1
        fi
    done
done

if [ "$ANY_MISSING" -eq 1 ]; then
    echo ""
    echo "  >>> SOME QP MAP DIRECTORIES ARE MISSING."
    echo "  >>> Phase 1b multi-sequence run must be re-executed for those methods."
    echo "  >>> Check ~/Minh/ipf/phase1/scripts/run_multi_sequence.sh"
fi

# ---------------------------------------------------------------------
print_header "2. VTM binary location & timestamp"
# ---------------------------------------------------------------------
if [ -x "$ENC" ]; then
    echo "  Binary: $ENC"
    stat -c "  Modified: %y" "$ENC"
    stat -c "  Size: %s bytes" "$ENC"
else
    echo "  [ERROR] EncoderApp not found at $ENC"
    echo "  Searching alternative locations..."
    find "$VTM_DIR" -name "EncoderApp" -type f 2>/dev/null
fi

# ---------------------------------------------------------------------
print_header "3. Does the binary recognise --ExternalQPMapDir?"
# ---------------------------------------------------------------------
if [ -x "$ENC" ]; then
    HELP_OUT=$("$ENC" --help 2>&1)
    if echo "$HELP_OUT" | grep -qi "ExternalQPMap"; then
        echo "  [OK] --ExternalQPMapDir option appears in --help output:"
        echo "$HELP_OUT" | grep -i "ExternalQPMap" | sed 's/^/      /'
    else
        echo "  [FAIL] --ExternalQPMapDir is NOT registered in the binary."
        echo "  This means either:"
        echo "    a) The patch was never applied"
        echo "    b) The patch was applied but not rebuilt (incremental cache)"
        echo "    c) EncAppCfg.cpp registration silently failed"
        echo ""
        echo "  Quick fix: cd ~/Minh/ipf/phase2 && bash scripts/apply_vtm_patch.sh"
    fi
fi

# ---------------------------------------------------------------------
print_header "4. C++ source patch markers"
# ---------------------------------------------------------------------
ENC_CFG_H="${VTM_DIR}/source/Lib/EncoderLib/EncCfg.h"
ENC_SLICE_CPP="${VTM_DIR}/source/Lib/EncoderLib/EncSlice.cpp"
ENC_APP_CFG_CPP="${VTM_DIR}/source/App/EncoderApp/EncAppCfg.cpp"

n1=$(grep -c "m_externalQPMapDir" "$ENC_CFG_H" 2>/dev/null || echo 0)
n2=$(grep -c "g_ipfQPReader" "$ENC_SLICE_CPP" 2>/dev/null || echo 0)
n3=$(grep -c "ExternalQPMapDir" "$ENC_APP_CFG_CPP" 2>/dev/null || echo 0)

echo "  EncCfg.h        : 'm_externalQPMapDir' occurrences: $n1   (expect >= 3)"
echo "  EncSlice.cpp    : 'g_ipfQPReader'      occurrences: $n2   (expect >= 2)"
echo "  EncAppCfg.cpp   : 'ExternalQPMapDir'   occurrences: $n3   (expect >= 1)"

if [ "$n1" -lt 3 ] || [ "$n2" -lt 2 ] || [ "$n3" -lt 1 ]; then
    echo ""
    echo "  >>> PATCH IS INCOMPLETE in the source tree."
    echo "  >>> Re-run: bash scripts/apply_vtm_patch.sh"
fi

# ---------------------------------------------------------------------
print_header "5. Binary vs. source timestamp"
# ---------------------------------------------------------------------
if [ -x "$ENC" ] && [ -f "$ENC_SLICE_CPP" ]; then
    bin_ts=$(stat -c %Y "$ENC")
    src_ts=$(stat -c %Y "$ENC_SLICE_CPP")
    if [ "$bin_ts" -lt "$src_ts" ]; then
        echo "  [WARNING] Binary is OLDER than EncSlice.cpp source."
        echo "    Binary modified : $(stat -c %y "$ENC")"
        echo "    Source modified : $(stat -c %y "$ENC_SLICE_CPP")"
        echo "  >>> Source has been edited since the last build."
        echo "  >>> Rebuild: cd $VTM_DIR && make -C build -j\$(nproc) all"
    else
        echo "  [OK] Binary is newer than source."
    fi
fi

# ---------------------------------------------------------------------
print_header "6. End-to-end test: encode 5 frames with and without QP map"
# ---------------------------------------------------------------------
TMP_QP_DIR=$(mktemp -d)
echo "  Generating dummy QP map directory at $TMP_QP_DIR..."
# Create one QP map per frame: 8 columns x 9 rows of CTUs at 1920x1152.
# Use QP=15 everywhere — this should produce a SUBSTANTIALLY different bitrate
# than uniform QP=32 if the patch is active.
for i in $(seq 0 4); do
    fname=$(printf "%05d.txt" "$i")
    > "${TMP_QP_DIR}/${fname}"
    for r in $(seq 1 9); do
        for c in $(seq 1 15); do printf "15 " >> "${TMP_QP_DIR}/${fname}"; done
        printf "15\n" >> "${TMP_QP_DIR}/${fname}"
    done
done

YUV="${HOME}/Minh/ipf/datasets/yuv/MOT17-04-DPM.yuv"
CFG="${VTM_DIR}/cfg/encoder_lowdelay_vtm.cfg"

if [ ! -f "$YUV" ] || [ ! -f "$CFG" ] || [ ! -x "$ENC" ]; then
    echo "  [SKIP] missing prerequisite (YUV / cfg / encoder)"
else
    # We bypass the python wrapper to keep the test minimal.
    BASE_CMD="$ENC -c $CFG -i $YUV -wdt 1920 -hgt 1152 -q 32 -f 5 -fr 30 --InternalBitDepth=8"

    echo ""
    echo "  Test A: uniform QP=32 (no external map)"
    SIZE_A=$($BASE_CMD -b /tmp/test_a.bin -o /tmp/test_a.yuv > /tmp/test_a.log 2>&1; \
             stat -c %s /tmp/test_a.bin 2>/dev/null || echo 0)
    echo "    bitstream size: $SIZE_A bytes"

    echo ""
    echo "  Test B: external QP=15 everywhere (--ExternalQPMapDir=$TMP_QP_DIR)"
    SIZE_B=$($BASE_CMD --ExternalQPMapDir=$TMP_QP_DIR -b /tmp/test_b.bin -o /tmp/test_b.yuv > /tmp/test_b.log 2>&1; \
             stat -c %s /tmp/test_b.bin 2>/dev/null || echo 0)
    echo "    bitstream size: $SIZE_B bytes"

    if [ "$SIZE_A" -gt 0 ] && [ "$SIZE_B" -gt 0 ]; then
        ratio=$(awk "BEGIN { printf \"%.2f\", $SIZE_B / $SIZE_A }")
        echo ""
        echo "  Size ratio B/A = $ratio"
        if [ "$(echo "$ratio > 1.5" | bc -l 2>/dev/null)" = "1" ]; then
            echo "  [OK] B is much larger — external QP map IS being honoured."
        else
            echo "  [BAD] B ≈ A — external QP map is BEING IGNORED by the binary."
            echo "  This confirms BUG #1.  Patch needs to be re-applied AND rebuilt."
        fi
    fi
fi

rm -rf "$TMP_QP_DIR"

echo ""
echo "============================================================"
echo "  Diagnostic complete."
echo "  Send the full output above so the next fix can be targeted."
echo "============================================================"
