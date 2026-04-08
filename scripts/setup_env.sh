#!/bin/bash
# =============================================================================
# Environment setup script for IPF Phase 1
# =============================================================================
# Usage:
#   bash scripts/setup_env.sh
# =============================================================================

set -euo pipefail

ENV_NAME="ipf"
PYTHON_VERSION="3.10"

echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "Installing project in editable mode..."
pip install -e .

echo ""
echo "=============================================="
echo "Environment ready!"
echo "Activate with: conda activate ${ENV_NAME}"
echo "=============================================="
