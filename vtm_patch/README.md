# VTM External QP Map Patch

## Overview

This patch adds support for per-CTU external QP maps to VTM (VVC Test Model).
It allows reading pre-computed QP values from text files and applying them
at the CTU level during encoding, which is essential for ROI-aware video
coding research.

## Patch Details

**Modified files** (3 files, ~120 lines changed):

1. `source/App/EncoderApp/EncAppCfg.cpp` -- Adds `--ExternalQPMapDir` CLI parameter
2. `source/Lib/EncoderLib/EncCfg.h` -- Adds storage for the QP map directory path
3. `source/Lib/EncoderLib/EncSlice.cpp` -- Reads QP map files and applies per-CTU QP

## QP Map File Format

Files are named `qp_NNNNNN.txt` where NNNNNN is the zero-padded frame index.
Each file contains one header line and then QP values in raster-scan CTU order:

```
# frame=0 rows=8 cols=15
32 32 30 28 22 22 22 28 30 32 32 32 32 32 32
32 30 28 24 22 22 22 24 28 30 32 32 32 32 32
...
```

This format is produced by `phase1/src/phase1/export/qp_exporter.py`.

## How to Apply

```bash
bash scripts/apply_vtm_patch.sh
```

## Compliance

- The patch only overrides the base QP decision per CTU
- All other coding tools (transforms, entropy coding, in-loop filters) remain standard
- Bitstreams produced are **fully VVC-compliant** and can be decoded by any
  standard VVC decoder (including the unmodified VTM DecoderApp)
- This is verified by decoding the patched encoder's output with the unmodified decoder

## Standard Practice

Externally-controlled CTU-level QP is a well-established technique in VCM
(Video Coding for Machines) research. See:
- MPEG VCM (ISO/IEC 23094-4) reference software
- Multiple JVET contributions on ROI-aware encoding
