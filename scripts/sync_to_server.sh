#!/bin/bash
# =============================================================================
# Sync local phase1 code to server (run from project root on local machine)
# Usage: bash scripts/sync_to_server.sh
# =============================================================================

SERVER="guest@100.104.64.97"
REMOTE_DIR="/home/guest/Minh/ipf/phase1"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Syncing: $LOCAL_DIR → $SERVER:$REMOTE_DIR"

rsync -avz --progress \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.egg-info/' \
  --exclude='.git/' \
  --exclude='outputs/' \
  --exclude='mock_data/frames/' \
  --exclude='mock_data/*.mp4' \
  --exclude='mock_data/*.avi' \
  "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/"

echo "Sync complete!"
