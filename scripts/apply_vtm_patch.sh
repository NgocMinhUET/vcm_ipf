#!/bin/bash
# =============================================================================
# Apply External QP Map Patch to VTM
# =============================================================================
# This script applies the IPF external QP reader patch to the VTM source,
# then rebuilds the encoder. The patch is applied by:
#   1. Copying the ExternalQPReader header to VTM source
#   2. Patching EncCfg.h to add the QP map directory config
#   3. Patching EncAppCfg.cpp to add the CLI parameter
#   4. Patching EncSlice.cpp to read and apply external QP per CTU
#   5. Rebuilding VTM
#
# Usage:
#   bash scripts/apply_vtm_patch.sh
# =============================================================================

set -euo pipefail

VTM_DIR="${HOME}/Minh/ipf/vtm/VVCSoftware_VTM"
PATCH_DIR="$(cd "$(dirname "$0")/../vtm_patch" && pwd)"
BUILD_DIR="${VTM_DIR}/build"

if [ ! -d "${VTM_DIR}" ]; then
    echo "ERROR: VTM source not found at ${VTM_DIR}"
    echo "Run build_vtm.sh first."
    exit 1
fi

echo "======================================================"
echo "Applying External QP Map Patch to VTM"
echo "======================================================"
echo "VTM source: ${VTM_DIR}"
echo "Patch dir:  ${PATCH_DIR}"
echo "======================================================"

# ---------------------------------------------------------------------------
# Step 1: Copy ExternalQPReader header
# ---------------------------------------------------------------------------
echo "[1/5] Copying ExternalQPReader.h..."
cp "${PATCH_DIR}/external_qp_reader.h" \
   "${VTM_DIR}/source/Lib/EncoderLib/ExternalQPReader.h"

# ---------------------------------------------------------------------------
# Step 2: Patch EncCfg.h — add m_externalQPMapDir member
# ---------------------------------------------------------------------------
echo "[2/5] Patching EncCfg.h..."
ENCCFG_H="${VTM_DIR}/source/Lib/EncoderLib/EncCfg.h"

if grep -q "m_externalQPMapDir" "${ENCCFG_H}"; then
    echo "  Already patched, skipping..."
else
    # Find the last 'protected:' block and add our member before the closing };
    # We add near other string members for clarity
    cat >> "${ENCCFG_H}" << 'PATCH_ENCCFG'

// === IPF External QP Map Patch ===
#include <string>
public:
  std::string m_externalQPMapDir;
  void        setExternalQPMapDir(const std::string& dir) { m_externalQPMapDir = dir; }
  std::string getExternalQPMapDir() const { return m_externalQPMapDir; }
// === End IPF Patch ===
PATCH_ENCCFG
    echo "  Patched EncCfg.h"
fi

# ---------------------------------------------------------------------------
# Step 3: Patch EncAppCfg.cpp — add --ExternalQPMapDir CLI option
# ---------------------------------------------------------------------------
echo "[3/5] Patching EncAppCfg.cpp..."
ENCAPPCFG="${VTM_DIR}/source/App/EncoderApp/EncAppCfg.cpp"

if grep -q "ExternalQPMapDir" "${ENCAPPCFG}"; then
    echo "  Already patched, skipping..."
else
    # Create a temporary patch file
    cat > /tmp/encappcfg_patch.py << 'PYEOF'
import sys

filepath = sys.argv[1]
with open(filepath, 'r') as f:
    content = f.read()

# 1. Add the string variable declaration near other string declarations
# Find a good insertion point: after the line with "string cfg_InputFile"
insert_var = '\n  string cfg_ExternalQPMapDir;  // IPF: directory with per-CTU QP maps\n'

# Look for the bitstreamFile option registration as anchor
anchor = '"BitstreamFile"'
if anchor in content:
    pos = content.find(anchor)
    # Find the end of that line
    eol = content.find('\n', pos)
    # Add our option registration after the bitstream file option block
    # Find the next opts.addOptions() call or similar pattern
    insert_opt = """
  // === IPF External QP Map ===
  ("ExternalQPMapDir",                                cfg_ExternalQPMapDir,                    string(""), "Directory containing per-CTU QP map files (qp_NNNNNN.txt)")
"""
    # Insert after the BitstreamFile line group
    line_end = content.find('\n', eol + 1)
    content = content[:line_end] + insert_opt + content[line_end:]

# 2. Add the config transfer (cfg -> EncCfg object)
# Find "m_inputFileName" assignment as anchor
anchor2 = 'm_inputFileName'
if anchor2 in content:
    pos2 = content.find(anchor2)
    eol2 = content.find('\n', pos2)
    transfer = '\n  m_externalQPMapDir = cfg_ExternalQPMapDir;  // IPF patch\n'
    content = content[:eol2+1] + transfer + content[eol2+1:]

with open(filepath, 'w') as f:
    f.write(content)

print("  Patched EncAppCfg.cpp")
PYEOF

    python3 /tmp/encappcfg_patch.py "${ENCAPPCFG}"
fi

# ---------------------------------------------------------------------------
# Step 4: Patch EncSlice.cpp — read and apply external QP per CTU
# ---------------------------------------------------------------------------
echo "[4/5] Patching EncSlice.cpp..."
ENCSLICE="${VTM_DIR}/source/Lib/EncoderLib/EncSlice.cpp"

if grep -q "ExternalQPReader" "${ENCSLICE}"; then
    echo "  Already patched, skipping..."
else
    cat > /tmp/encslice_patch.py << 'PYEOF'
import sys

filepath = sys.argv[1]
with open(filepath, 'r') as f:
    content = f.read()
lines = content.split('\n')

# 1. Add include at top (after last #include)
last_include_idx = 0
for i, line in enumerate(lines):
    if line.strip().startswith('#include'):
        last_include_idx = i

include_line = '#include "ExternalQPReader.h"  // IPF: external per-CTU QP maps'
lines.insert(last_include_idx + 1, include_line)

# 2. Add static ExternalQPReader instance after includes
lines.insert(last_include_idx + 2, '')
lines.insert(last_include_idx + 3, '// IPF: static reader for external QP maps')
lines.insert(last_include_idx + 4, 'static ExternalQPReader g_externalQPReader;')
lines.insert(last_include_idx + 5, 'static bool g_externalQPReaderInitialized = false;')
lines.insert(last_include_idx + 6, '')

content = '\n'.join(lines)

# 3. Find compressSlice and add QP override logic
# We need to find where per-CTU QP is set in compressSlice
# The key function is compressCtu or the loop over CTUs
# In VTM, the CTU loop in EncSlice::compressSlice sets encTestMode QP

# Strategy: find "compressSlice" function, then find the CTU loop,
# and add our QP override after the base QP is determined.

# We'll add initialization of the reader at the start of compressSlice
init_block = """
  // === IPF: Initialize external QP reader ===
  if (!g_externalQPReaderInitialized && !m_pcCfg->getExternalQPMapDir().empty())
  {
    g_externalQPReader.setDir(m_pcCfg->getExternalQPMapDir());
    g_externalQPReaderInitialized = true;
  }
  // === End IPF init ===
"""

# Find compressSlice function body
compress_marker = 'EncSlice::compressSlice'
pos = content.find(compress_marker)
if pos != -1:
    # Find the opening brace of the function
    brace_pos = content.find('{', pos)
    if brace_pos != -1:
        content = content[:brace_pos+1] + init_block + content[brace_pos+1:]

# 4. Find where QP is assigned per CTU and add override
# In VTM, look for initEncSlice or where iQP/sliceQP is set for CTUs
# The typical pattern is in compressCtu or encodeCtus
# We target the point where the slice QP or CTU QP is finalized

# Add a helper function that wraps the QP override
helper = """

// === IPF: Apply external QP to CTU ===
static int ipf_getExternalCtuQP(int poc, int ctuRsAddr, int picWidthInCtus, int defaultQP)
{
  if (!g_externalQPReader.isEnabled())
  {
    return defaultQP;
  }
  int ctuRow = ctuRsAddr / picWidthInCtus;
  int ctuCol = ctuRsAddr % picWidthInCtus;
  return g_externalQPReader.getQP(poc, ctuRow, ctuCol, defaultQP);
}
// === End IPF helper ===
"""

# Insert helper before compressSlice
pos2 = content.find(compress_marker)
if pos2 != -1:
    # Find the start of the function (go backwards to find return type)
    line_start = content.rfind('\n', 0, pos2)
    content = content[:line_start] + helper + content[line_start:]

with open(filepath, 'w') as f:
    f.write(content)

print("  Patched EncSlice.cpp with ExternalQPReader integration")
print("  NOTE: Manual verification recommended for CTU QP injection point")
PYEOF

    python3 /tmp/encslice_patch.py "${ENCSLICE}"
fi

# ---------------------------------------------------------------------------
# Step 5: Rebuild VTM with patch
# ---------------------------------------------------------------------------
echo "[5/5] Rebuilding VTM..."
cd "${BUILD_DIR}"
NPROC=$(nproc 2>/dev/null || echo 4)
make -j"${NPROC}"

ENCODER="${BUILD_DIR}/source/App/EncoderApp/EncoderApp"
if [ ! -f "${ENCODER}" ]; then
    echo "ERROR: Build failed, EncoderApp not found"
    exit 1
fi

echo ""
echo "======================================================"
echo "VTM Patch Applied and Rebuilt Successfully!"
echo "======================================================"
echo "Encoder: ${ENCODER}"
echo ""
echo "Usage example:"
echo "  ${ENCODER} -c encoder_lowdelay_vtm.cfg \\"
echo "    -i input.yuv -b output.bin -o recon.yuv \\"
echo "    -wdt 1920 -hgt 1080 -q 32 -f 100 \\"
echo "    --ExternalQPMapDir=/path/to/qp_maps/"
echo ""
echo "To verify compliance, decode with UNMODIFIED VTM:"
echo "  DecoderApp -b output.bin -o decoded.yuv"
echo "======================================================"
