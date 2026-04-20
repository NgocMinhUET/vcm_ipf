#!/bin/bash
# =============================================================================
# Run Phase 2 Pilot Experiment
# =============================================================================
# Executes the full encoding pipeline for the pilot experiment grid.
#
# Prerequisites:
#   1. Phase 1 multi-sequence run completed (QP maps generated)
#   2. VTM built and patched
#   3. MOT17 YUV files prepared
#
# Usage:
#   bash scripts/run_pilot.sh [config]
# Example:
#   bash scripts/run_pilot.sh configs/pilot.yaml
# =============================================================================

set -euo pipefail

CONFIG="${1:-configs/pilot.yaml}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Derive a tier name from the config filename (smoke / quick / pilot / ...)
TIER_NAME="$(basename "${CONFIG}" .yaml)"
LOG_FILE="${TIER_NAME}_run.log"

echo "======================================================"
echo "Phase 2 Encoding Run — tier: ${TIER_NAME}"
echo "======================================================"
echo "Config:  ${CONFIG}"
echo "Log:     ${LOG_FILE}"
echo "======================================================"

cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

python -c "
import logging, sys
from phase2.core.config import load_phase2_config
from phase2.pipeline.encode_pipeline import EncodingPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('${LOG_FILE}', encoding='utf-8'),
    ],
)

cfg = load_phase2_config('${CONFIG}')
pipeline = EncodingPipeline(cfg)
results = pipeline.run_all()

n_ok = sum(1 for r in results if r.encode and r.encode.get('success'))
print(f'\n{cfg.experiment_id} complete: {n_ok}/{len(results)} successful runs')
print(f'Results: {pipeline.output_dir}')
"

# Read experiment_id back from the YAML so we can print the correct path.
EXP_ID=$(python -c "
from phase2.core.config import load_phase2_config
print(load_phase2_config('${CONFIG}').experiment_id)
")
OUTPUT_DIR="$(python -c "
from phase2.core.config import load_phase2_config
from pathlib import Path
print(Path(load_phase2_config('${CONFIG}').output_dir).expanduser() / '${EXP_ID}')
")"

echo ""
echo "======================================================"
echo "Encoding Run Complete (tier: ${TIER_NAME})"
echo "======================================================"
echo "Results at: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Quick visual check:"
echo "       cat ${OUTPUT_DIR}/experiment_table.txt"
echo ""
echo "  2. Verify pipeline correctness (smoke test):"
echo "       - bitrate(M0) != bitrate(M4)            ← BUG #1 fix"
echo "       - psnr_y_full != psnr_y_roi             ← BUG #2 fix"
echo "       - mAP50 > 0                             ← BUG #3 fix"
echo ""
echo "  3. BD-Rate analysis (only meaningful when QPs >= 4 and methods >= 2):"
echo "       python -m phase2.analysis.results_aggregator \\"
echo "         --experiment-dir ${OUTPUT_DIR}"
echo "======================================================"
