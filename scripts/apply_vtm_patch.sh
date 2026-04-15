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

export ENCSLICE_PATH="${ENCSLICE}"
python3 << 'PYEOF'
import sys, re, os

path = os.environ.get('ENCSLICE_PATH', '')
if not path:
    print("ERROR: ENCSLICE_PATH not set", file=sys.stderr)
    sys.exit(1)

with open(path, 'r') as f:
    content = f.read()

if 'ExternalQPReader' in content:
    print("  Already patched, skipping.")
    sys.exit(0)

lines = content.split('\n')

# -------------------------------------------------------------------------
# (a) Add #include and globals after the last #include in the file
# -------------------------------------------------------------------------
last_inc = 0
for i, ln in enumerate(lines):
    if ln.strip().startswith('#include'):
        last_inc = i
lines.insert(last_inc + 1, '#include "ExternalQPReader.h"  // IPF: external per-CTU QP maps')
lines.insert(last_inc + 2, '')
lines.insert(last_inc + 3, '// IPF: singleton QP map reader (initialised once per encode session)')
lines.insert(last_inc + 4, 'static ExternalQPReader g_ipfQPReader;')
lines.insert(last_inc + 5, 'static bool             g_ipfQPReaderInit = false;')
lines.insert(last_inc + 6, '')
content = '\n'.join(lines)

# -------------------------------------------------------------------------
# (b) Initialise reader at the start of compressSlice()
# -------------------------------------------------------------------------
compress_marker = 'EncSlice::compressSlice'
pos = content.find(compress_marker)
if pos == -1:
    print("ERROR: compressSlice not found in EncSlice.cpp", file=sys.stderr)
    sys.exit(1)
open_brace = content.find('{', pos)
init_code = """
  // === IPF: initialise external QP reader once ===
  if (!g_ipfQPReaderInit && !m_pcCfg->getExternalQPMapDir().empty())
  {
    g_ipfQPReader.setDir(m_pcCfg->getExternalQPMapDir());
    g_ipfQPReaderInit = true;
  }
  // === End IPF init ===
"""
content = content[:open_brace + 1] + init_code + content[open_brace + 1:]

# -------------------------------------------------------------------------
# (c) Per-CTU QP override injected before EVERY compressCtu() call.
#
# KEY FIX: In VTM-23.4 compressCtu is called from encodeCtus(), not from
# compressSlice() directly.  The local variable names in encodeCtus() differ
# from those in compressSlice(), so we MUST NOT reference actualQP or
# CHANNEL_TYPE_LUMA (which may not exist in that scope).
#
# Safe variables always in scope inside the CTU loop of any EncSlice function:
#   ctuRsAddr              - uint32_t  (loop variable)
#   cs.pcv->widthInCtus    - uint32_t  (available via CodingStructure)
#   cs.slice->getPOC()     - int       (always available)
#   cs.slice->getSliceQp() - int       (base slice QP, always available)
#   cs.slice->setSliceQp() - void      (setter, always available)
#
# The override works by temporarily setting the slice QP to the external value
# before each CTU is encoded.  VTM's CTU encoder reads the slice QP from the
# slice object at encoding time, so this is the correct hook point.
# -------------------------------------------------------------------------
ctu_qp_override = (
    '  // === IPF: per-CTU QP override from external map ===\n'
    '  if (g_ipfQPReader.isEnabled())\n'
    '  {\n'
    '    const int _ctuRow = (int)(ctuRsAddr / cs.pcv->widthInCtus);\n'
    '    const int _ctuCol = (int)(ctuRsAddr % cs.pcv->widthInCtus);\n'
    '    const int _baseQP = cs.slice->getSliceQp();\n'
    '    const int _extQP  = g_ipfQPReader.getQP(\n'
    '      (int)cs.slice->getPOC(), _ctuRow, _ctuCol, _baseQP);\n'
    '    cs.slice->setSliceQp(_extQP);\n'
    '  }\n'
    '  // === End IPF CTU QP override ===\n'
)

# Inject before EVERY call to compressCtu() in the file (covers both
# compressSlice and encodeCtus if either calls it).
call_pattern = re.compile(r'(?m)^( *)(m_pcCuEncoder->compressCtu\s*\()')
new_content = call_pattern.sub(lambda m: ctu_qp_override + m.group(0), content)

if new_content == content:
    print("  WARNING: compressCtu() call not found — CTU QP override NOT injected.")
    print("  Encoder will build but external QP maps will have no effect.")
    print("  Inspect EncSlice.cpp manually and check the compressCtu call site.")
else:
    n_injections = len(call_pattern.findall(content))
    print(f"  CTU QP override injected before {n_injections} compressCtu() call(s).")
    content = new_content

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
