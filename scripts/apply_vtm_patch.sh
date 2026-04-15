#!/bin/bash
# =============================================================================
# Apply External QP Map Patch to VTM
# =============================================================================
# Applies the IPF external per-CTU QP reader patch to VTM source and rebuilds.
#
# Changes:
#   1. EncCfg.h         — adds m_externalQPMapDir member INSIDE the class
#   2. EncAppCfg.cpp    — adds --ExternalQPMapDir CLI option
#   3. EncSlice.cpp     — reads QP maps and overrides per-CTU QP in the
#                         compressSlice CTU loop (before compressCtu call)
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
echo "======================================================"

# ---------------------------------------------------------------------------
# Reset any previous (broken) patch to get a clean slate
# ---------------------------------------------------------------------------
echo "[0/5] Resetting modified VTM source files to original state..."
cd "${VTM_DIR}"
git checkout source/Lib/EncoderLib/EncCfg.h 2>/dev/null || true
git checkout source/Lib/EncoderLib/EncSlice.cpp 2>/dev/null || true
git checkout source/App/EncoderApp/EncAppCfg.cpp 2>/dev/null || true
rm -f source/Lib/EncoderLib/ExternalQPReader.h
echo "  Reset complete."

# ---------------------------------------------------------------------------
# Step 1: Copy ExternalQPReader header
# ---------------------------------------------------------------------------
echo "[1/5] Copying ExternalQPReader.h..."
cp "${PATCH_DIR}/external_qp_reader.h" \
   "${VTM_DIR}/source/Lib/EncoderLib/ExternalQPReader.h"
echo "  Done."

# ---------------------------------------------------------------------------
# Step 2: Patch EncCfg.h — add m_externalQPMapDir INSIDE the class body
# ---------------------------------------------------------------------------
echo "[2/5] Patching EncCfg.h (inserting member inside class)..."
ENCCFG_H="${VTM_DIR}/source/Lib/EncoderLib/EncCfg.h"

python3 << PYEOF
import re, sys

path = "${ENCCFG_H}"
with open(path, 'r') as f:
    content = f.read()

# We insert the member and accessors just before the LAST '};\n' in the file,
# which closes the EncCfg class definition.
ipf_block = """
protected:
  // === IPF: External per-CTU QP map directory ===
  std::string   m_externalQPMapDir;
public:
  void          setExternalQPMapDir(const std::string& d) { m_externalQPMapDir = d; }
  std::string   getExternalQPMapDir()               const { return m_externalQPMapDir; }
  // === End IPF ===
"""

# Find the last occurrence of "};" which closes the EncCfg class.
last_close = content.rfind('};')
if last_close == -1:
    print("ERROR: Could not find closing '}; in EncCfg.h", file=sys.stderr)
    sys.exit(1)

# Sanity check: make sure we have not already patched.
if 'm_externalQPMapDir' in content:
    print("  Already patched, skipping.")
    sys.exit(0)

# Insert the IPF block before the last '};'
content = content[:last_close] + ipf_block + '\n' + content[last_close:]

with open(path, 'w') as f:
    f.write(content)

print("  EncCfg.h patched OK — member inserted inside class body.")
PYEOF

# ---------------------------------------------------------------------------
# Step 3: Patch EncAppCfg.cpp — add --ExternalQPMapDir CLI option
# ---------------------------------------------------------------------------
echo "[3/5] Patching EncAppCfg.cpp..."
ENCAPPCFG="${VTM_DIR}/source/App/EncoderApp/EncAppCfg.cpp"

python3 << PYEOF
import sys

path = "${ENCAPPCFG}"
with open(path, 'r') as f:
    content = f.read()

if 'ExternalQPMapDir' in content:
    print("  Already patched, skipping.")
    sys.exit(0)

# --- (a) Register the option ---
# Find the "BitstreamFile" registration as an anchor (it always exists).
anchor_opt = '"BitstreamFile"'
pos = content.find(anchor_opt)
if pos == -1:
    print("WARNING: Could not find BitstreamFile anchor in EncAppCfg.cpp", file=sys.stderr)
    sys.exit(0)
eol = content.find('\n', pos)
# Insert on the next line after the BitstreamFile group.
eol2 = content.find('\n', eol + 1)
opt_line = '\n  ("ExternalQPMapDir",  m_externalQPMapDir, std::string(""), "Directory with per-CTU QP maps (qp_NNNNNN.txt)")\n'
content = content[:eol2 + 1] + opt_line + content[eol2 + 1:]

# --- (b) Wire to EncCfg ---
# Find the place where other m_xxx members are transferred to the encoder.
# A reliable anchor is "m_inputFileName" assignment.
anchor_cfg = 'm_inputFileName'
pos2 = content.find(anchor_cfg)
if pos2 != -1:
    eol3 = content.find('\n', pos2)
    wire_line = '\n  m_cEncLib.setExternalQPMapDir( m_externalQPMapDir );  // IPF\n'
    content = content[:eol3 + 1] + wire_line + content[eol3 + 1:]

# --- (c) Add the string member declaration ---
# Find "string m_bitstreamFileName" as anchor for member variables.
anchor_mem = 'm_bitstreamFileName'
pos3 = content.find(anchor_mem)
if pos3 != -1:
    eol4 = content.find('\n', pos3)
    mem_line = '\n  std::string  m_externalQPMapDir;  // IPF external QP map directory\n'
    content = content[:eol4 + 1] + mem_line + content[eol4 + 1:]

with open(path, 'w') as f:
    f.write(content)

print("  EncAppCfg.cpp patched OK.")
PYEOF

# ---------------------------------------------------------------------------
# Step 4: Patch EncSlice.cpp — add include, global reader, and CTU QP override
# ---------------------------------------------------------------------------
echo "[4/5] Patching EncSlice.cpp..."
ENCSLICE="${VTM_DIR}/source/Lib/EncoderLib/EncSlice.cpp"

python3 << PYEOF
import sys, re

path = "${ENCSLICE}"
with open(path, 'r') as f:
    content = f.read()

if 'ExternalQPReader' in content:
    print("  Already patched, skipping.")
    sys.exit(0)

lines = content.split('\n')

# --- (a) Add #include after the last #include in the file ---
last_inc = 0
for i, ln in enumerate(lines):
    if ln.strip().startswith('#include'):
        last_inc = i
lines.insert(last_inc + 1, '#include "ExternalQPReader.h"  // IPF')
lines.insert(last_inc + 2, '')
lines.insert(last_inc + 3, '// IPF: singleton QP map reader (initialised once per encode session)')
lines.insert(last_inc + 4, 'static ExternalQPReader g_ipfQPReader;')
lines.insert(last_inc + 5, 'static bool             g_ipfQPReaderInit = false;')
lines.insert(last_inc + 6, '')
content = '\n'.join(lines)

# --- (b) Initialise reader at start of compressSlice ---
marker = 'EncSlice::compressSlice'
pos = content.find(marker)
if pos == -1:
    print("ERROR: compressSlice not found in EncSlice.cpp", file=sys.stderr)
    sys.exit(1)
open_brace = content.find('{', pos)
init_code = r"""
  // === IPF: initialise external QP reader ===
  if (!g_ipfQPReaderInit && !m_pcCfg->getExternalQPMapDir().empty())
  {
    g_ipfQPReader.setDir(m_pcCfg->getExternalQPMapDir());
    g_ipfQPReaderInit = true;
  }
  // === End IPF init ===
"""
content = content[:open_brace + 1] + init_code + content[open_brace + 1:]

# --- (c) Inject per-CTU QP override before every compressCtu() call ---
# Pattern: 'm_pcCuEncoder->compressCtu('  appears in the CTU loop.
# We insert the QP override block immediately before each such call.
ctu_qp_override = (
    '  // === IPF: per-CTU QP override from external map ===\n'
    '  if (g_ipfQPReader.isEnabled())\n'
    '  {\n'
    '    int _ctuRow = ctuRsAddr / cs.pcv->widthInCtus;\n'
    '    int _ctuCol = ctuRsAddr % cs.pcv->widthInCtus;\n'
    '    int _extQP  = g_ipfQPReader.getQP(cs.slice->getPOC(), _ctuRow, _ctuCol, actualQP[CHANNEL_TYPE_LUMA]);\n'
    '    actualQP[CHANNEL_TYPE_LUMA] = _extQP;\n'
    '    prevQP  [CHANNEL_TYPE_LUMA] = _extQP;\n'
    '  }\n'
    '  // === End IPF CTU QP override ===\n'
)

# Replace the first (and usually only) compressCtu call in compressSlice.
# We locate the call and inject just before it.
call_pattern = re.compile(r'( +)(m_pcCuEncoder->compressCtu\s*\()')
match = call_pattern.search(content, open_brace)
if match:
    insert_pos = match.start()
    content = content[:insert_pos] + ctu_qp_override + content[insert_pos:]
    print("  EncSlice.cpp: CTU QP override injected before compressCtu().")
else:
    print("  WARNING: compressCtu() call not found; CTU QP override NOT injected.")
    print("  The encoder will build but external QP maps will have no effect.")
    print("  Manual inspection of EncSlice.cpp required.")

with open(path, 'w') as f:
    f.write(content)

print("  EncSlice.cpp patched OK.")
PYEOF

# ---------------------------------------------------------------------------
# Step 5: Rebuild VTM
# ---------------------------------------------------------------------------
echo "[5/5] Rebuilding VTM..."
cd "${BUILD_DIR}"
NPROC=$(nproc 2>/dev/null || echo 4)
make -j"${NPROC}"

ENCODER="${BUILD_DIR}/source/App/EncoderApp/EncoderApp"
if [ ! -f "${ENCODER}" ]; then
    echo ""
    echo "BUILD FAILED. Review errors above."
    exit 1
fi

echo ""
echo "======================================================"
echo "VTM Patch Applied and Rebuilt Successfully!"
echo "======================================================"
echo "Encoder: ${ENCODER}"
echo ""
echo "Test with external QP maps:"
echo "  ${ENCODER} -c <cfg> -i input.yuv -b out.bin -o recon.yuv \\"
echo "    -wdt 1920 -hgt 1088 -q 32 -f 100 \\"
echo "    --ExternalQPMapDir=/path/to/qp_maps/"
echo ""
echo "Compliance check (decode with UNMODIFIED decoder):"
echo "  DecoderApp -b out.bin -o decoded.yuv"
echo "======================================================"
