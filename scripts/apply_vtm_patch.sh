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
git checkout source/App/EncoderApp/EncAppCfg.h 2>/dev/null || true
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
ENCAPPCFG_H="${VTM_DIR}/source/App/EncoderApp/EncAppCfg.h"

export ENCAPPCFG_PATH="${ENCAPPCFG}"
export ENCAPPCFG_H_PATH="${ENCAPPCFG_H}"
python3 << 'PYEOF'
import sys, re, os

cpp_path = os.environ.get('ENCAPPCFG_PATH', '')
hdr_path = os.environ.get('ENCAPPCFG_H_PATH', '')

# -----------------------------------------------------------------------
# (a) EncAppCfg.h — add std::string m_externalQPMapDir member
# -----------------------------------------------------------------------
if hdr_path and os.path.isfile(hdr_path):
    with open(hdr_path, 'r') as f:
        hdr = f.read()
    if 'm_externalQPMapDir' not in hdr:
        # Anchor: the m_bitstreamFileName member is always in EncAppCfg.h
        anchors_h = ['m_bitstreamFileName', 'm_inputFileName', 'm_reconFileName']
        anchor_pos = -1
        for a in anchors_h:
            anchor_pos = hdr.find(a)
            if anchor_pos != -1:
                break
        if anchor_pos != -1:
            eol = hdr.find('\n', anchor_pos)
            hdr = hdr[:eol + 1] + '  std::string  m_externalQPMapDir;  // IPF: per-CTU QP map directory\n' + hdr[eol + 1:]
            with open(hdr_path, 'w') as f:
                f.write(hdr)
            print("  EncAppCfg.h: m_externalQPMapDir member added.")
        else:
            print("  WARNING: EncAppCfg.h anchor not found; member not added.")
    else:
        print("  EncAppCfg.h: already patched.")

# -----------------------------------------------------------------------
# (b) EncAppCfg.cpp — register CLI option and wire to EncCfg
# -----------------------------------------------------------------------
with open(cpp_path, 'r') as f:
    content = f.read()

if 'ExternalQPMapDir' in content:
    print("  EncAppCfg.cpp: already patched, skipping.")
    sys.exit(0)

# --- Register the CLI option ---
# Try multiple anchors for the option-registration block.
# VTM-23.4 uses po::Options with ("Name", variable, default, "description") syntax.
opt_anchors = [
    '"BitstreamFile"',   # VTM <= 18
    '"BitstreamFile,b"', # Some versions include shorthand
    '"InputFile"',
    '"InputFile,i"',
    '"ReconFile"',
    '"ReconFile,o"',
]
opt_pos = -1
for anchor in opt_anchors:
    opt_pos = content.find(anchor)
    if opt_pos != -1:
        print(f"  Found CLI option anchor: {anchor}")
        break

if opt_pos == -1:
    # Last resort: find any ("...", m_inputFileName, ...) line
    m = re.search(r'\("[\w,]+",\s*m_inputFileName\b', content)
    if m:
        opt_pos = m.start()
        print("  Found CLI option anchor via m_inputFileName regex.")

if opt_pos == -1:
    print("  WARNING: Could not locate option-registration block in EncAppCfg.cpp.", file=sys.stderr)
    print("  The --ExternalQPMapDir flag will NOT be available on the command line.", file=sys.stderr)
    print("  Continuing — QP maps can still be injected via env var in the Python wrapper.", file=sys.stderr)
else:
    # Insert new option after the anchor line (skip 1 line to stay in the same block)
    eol = content.find('\n', opt_pos)
    eol2 = content.find('\n', eol + 1)
    opt_line = '\n  ("ExternalQPMapDir",  m_externalQPMapDir, std::string(""), "IPF: directory with per-CTU QP map files (qp_NNNNNN.txt)")\n'
    content = content[:eol2 + 1] + opt_line + content[eol2 + 1:]
    print("  CLI option --ExternalQPMapDir registered.")

# --- Wire m_externalQPMapDir to EncCfg (setExternalQPMapDir) ---
# Anchor: look for the xInitLibCfg function or m_inputFileName assignment to EncLib
wire_anchors = [
    'm_cEncLib.setInputFileName',
    'm_cEncLib.setInputFile',
    'xInitLibCfg',
]
wired = False
for wa in wire_anchors:
    wp = content.find(wa)
    if wp != -1:
        eol_w = content.find('\n', wp)
        wire_line = '\n  m_cEncLib.setExternalQPMapDir( m_externalQPMapDir );  // IPF\n'
        content = content[:eol_w + 1] + wire_line + content[eol_w + 1:]
        print(f"  Wired m_externalQPMapDir to EncCfg via anchor: {wa}")
        wired = True
        break

if not wired:
    # Fallback: find any m_cEncLib.set... call
    m2 = re.search(r'(m_cEncLib\.set\w+\([^;]+;\n)', content)
    if m2:
        insert_at = m2.end()
        wire_line = '  m_cEncLib.setExternalQPMapDir( m_externalQPMapDir );  // IPF\n'
        content = content[:insert_at] + wire_line + content[insert_at:]
        print("  Wired m_externalQPMapDir via fallback m_cEncLib.set pattern.")
    else:
        print("  WARNING: Could not wire m_externalQPMapDir to EncCfg.", file=sys.stderr)
        print("  ExternalQPMapDir option registered but won't be forwarded to encoder.", file=sys.stderr)

with open(cpp_path, 'w') as f:
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
lines.insert(last_inc + 2, '#include <cstdlib>               // IPF: std::getenv')
lines.insert(last_inc + 3, '#include <cmath>                 // IPF: std::pow for lambda')
lines.insert(last_inc + 4, '')
lines.insert(last_inc + 5, '// IPF: singleton QP map reader (initialised once per encode session)')
lines.insert(last_inc + 6, 'static ExternalQPReader g_ipfQPReader;')
lines.insert(last_inc + 7, 'static bool             g_ipfQPReaderInit = false;')
lines.insert(last_inc + 8, '// IPF: per-slice save/restore state for the restore-before-next pattern')
lines.insert(last_inc + 9, 'static int              g_ipfPrevQP  = -1;   // -1 = no pending restore')
lines.insert(last_inc + 10, 'static double           g_ipfPrevLam = -1.0;')
lines.insert(last_inc + 11, '')
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
  // === IPF: initialise external QP reader once per process ===
  if (!g_ipfQPReaderInit)
  {
    // Primary: CLI option wired through EncCfg (requires anchor match in EncAppCfg.cpp).
    std::string _ipfDir = m_pcCfg->getExternalQPMapDir();
    // Reliable fallback: environment variable set by the Python pipeline.
    if (_ipfDir.empty())
    {
      const char* _envDir = std::getenv("VTM_EXTERNAL_QP_DIR");
      if (_envDir && _envDir[0] != '\\0') { _ipfDir = std::string(_envDir); }
    }
    if (!_ipfDir.empty())
    {
      g_ipfQPReader.setDir(_ipfDir);
      std::cout << "[IPF] ExternalQPReader enabled, reading from: " << _ipfDir << std::endl;
    }
    g_ipfQPReaderInit = true;
  }
  // Reset per-slice save/restore state at the start of every new slice.
  // setUpLambda() will set the correct base QP+lambda immediately after,
  // so any stale values from the previous slice are harmless here.
  g_ipfPrevQP  = -1;
  g_ipfPrevLam = -1.0;
  // === End IPF init ===
"""
content = content[:open_brace + 1] + init_code + content[open_brace + 1:]

# -------------------------------------------------------------------------
# (c) Per-CTU QP + lambda override using "restore-before-next" pattern.
#
# WHY LAMBDA MATTERS: setSliceQp() alone changes the quantisation level but
# NOT the lambda (rate-distortion multiplier).  Lambda is set once per slice
# by setUpLambda() and controls all coding decisions (partition, mode, etc.).
# Without updating the lambda, M4 encoding decisions are identical to M0 —
# only the bitstream QP signalling changes, producing ~0% bitrate difference.
#
# FIX: Before each CTU we restore the previous CTU's saved QP+lambda (so VTM
# post-CTU processing always sees the correct base values), then immediately
# set the external QP+lambda for the current CTU.  The restore-before-next
# approach avoids needing a second injection point after compressCtu().
#
# Lambda formula: lambda = 0.57 * 2^((QP-12)/3)  (VTM simplified formula)
# This is consistent with VTM's own calculation and ensures proper RD.
#
# m_pcRdCost is a member of EncSlice (declared in EncSlice.h) so it is
# always accessible from the CTU loop regardless of which EncSlice function
# calls compressCtu().
# -------------------------------------------------------------------------
ctu_qp_override = (
    '  // === IPF: per-CTU QP+lambda override (restore-before-next) ===\n'
    '  if (g_ipfPrevQP >= 0)\n'
    '  {\n'
    '    cs.slice->setSliceQp(g_ipfPrevQP);\n'
    '    m_pcRdCost->setLambda(g_ipfPrevLam, cs.slice->getSPS()->getBitDepths());\n'
    '    g_ipfPrevQP = -1;\n'
    '  }\n'
    '  if (g_ipfQPReader.isEnabled())\n'
    '  {\n'
    '    const int _ctuRow = (int)(ctuRsAddr / cs.pcv->widthInCtus);\n'
    '    const int _ctuCol = (int)(ctuRsAddr % cs.pcv->widthInCtus);\n'
    '    const int _curQP  = cs.slice->getSliceQp();\n'
    '    const int _extQP  = g_ipfQPReader.getQP(\n'
    '      (int)cs.slice->getPOC(), _ctuRow, _ctuCol, _curQP);\n'
    '    if (_extQP != _curQP)\n'
    '    {\n'
    '      g_ipfPrevQP  = _curQP;\n'
    '      g_ipfPrevLam = m_pcRdCost->getLambda();\n'
    '      cs.slice->setSliceQp(_extQP);\n'
    '      m_pcRdCost->setLambda(\n'
    '        0.57 * std::pow(2.0, (_extQP - 12.0) / 3.0),\n'
    '        cs.slice->getSPS()->getBitDepths());\n'
    '    }\n'
    '  }\n'
    '  // === End IPF CTU QP+lambda override ===\n'
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

# VTM cmake outputs binaries to VTM_DIR/bin/umake/<gcc>/<arch>/release/
# (NOT under BUILD_DIR). Search the VTM source root directory.
ENCODER=$(find "${VTM_DIR}" -name "EncoderApp" -type f 2>/dev/null | sort | tail -1)
DECODER=$(find "${VTM_DIR}" -name "DecoderApp" -type f 2>/dev/null | sort | tail -1)

if [ -z "${ENCODER}" ] || [ ! -f "${ENCODER}" ]; then
    echo ""
    echo "BUILD FAILED: EncoderApp binary not found under ${VTM_DIR}"
    echo "Check compiler errors above."
    exit 1
fi

echo ""
echo "======================================================"
echo "VTM Patch Applied and Rebuilt Successfully!"
echo "======================================================"
echo "Encoder: ${ENCODER}"
echo "Decoder: ${DECODER}"
echo ""
echo "Update your phase2 config (configs/pilot.yaml) vtm section:"
echo "  encoder_path: ${ENCODER}"
echo "  decoder_path: ${DECODER}"
echo ""
echo "Test with external QP maps:"
echo "  ${ENCODER} -c <cfg> -i input.yuv -b out.bin -o recon.yuv \\"
echo "    -wdt 1920 -hgt 1088 -q 32 -f 100 \\"
echo "    --ExternalQPMapDir=/path/to/qp_maps/"
echo ""
echo "Compliance check (decode with UNMODIFIED decoder):"
echo "  ${DECODER} -b out.bin -o decoded.yuv"
echo "======================================================"
