#!/bin/bash
# =============================================================================
# Sync local phase2 code to server
# Usage: bash scripts/sync_to_server.sh
# =============================================================================

SERVER="guest@100.104.64.97"
REMOTE_DIR="/home/guest/Minh/ipf/phase2"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Syncing: $LOCAL_DIR -> $SERVER:$REMOTE_DIR"

rsync -avz --progress \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.egg-info/' \
  --exclude='.git/' \
  --exclude='outputs/' \
  "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/"

echo "Sync complete!"
echo ""
echo "Also syncing phase1 (cross-sequence analysis)..."

PHASE1_LOCAL="$(cd "$LOCAL_DIR/../phase1" && pwd)"
PHASE1_REMOTE="/home/guest/Minh/ipf/phase1"

rsync -avz --progress \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.egg-info/' \
  --exclude='.git/' \
  --exclude='outputs/' \
  --exclude='mock_data/frames/' \
  --exclude='mock_data/*.mp4' \
  --exclude='mock_data/*.avi' \
  "$PHASE1_LOCAL/" "$SERVER:$PHASE1_REMOTE/"

echo "All syncs complete!"
