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

echo "======================================================"
echo "Phase 2 Pilot Experiment"
echo "======================================================"
echo "Config: ${CONFIG}"
echo "======================================================"

cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

python -c "
import logging
import sys
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pilot_run.log', encoding='utf-8'),
    ]
)

from phase2.core.config import load_phase2_config
from phase2.pipeline.encode_pipeline import EncodingPipeline

cfg = load_phase2_config('${CONFIG}')
pipeline = EncodingPipeline(cfg)
results = pipeline.run_all()

n_ok = sum(1 for r in results if r.encode and r.encode.get('success'))
print(f'\nPilot complete: {n_ok}/{len(results)} successful runs')
print(f'Results: {pipeline.output_dir}')
"

echo ""
echo "======================================================"
echo "Pilot Experiment Complete"
echo "======================================================"
echo "Results at: ~/Minh/ipf/phase2_outputs/pilot_v1/"
echo ""
echo "Next step: run BD-Rate analysis"
echo "  python -m phase2.analysis.results_aggregator \\"
echo "    --experiment-dir ~/Minh/ipf/phase2_outputs/pilot_v1"
echo "======================================================"
