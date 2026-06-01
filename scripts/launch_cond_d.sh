#!/usr/bin/env bash
# Condition D: Rule-based ESS (CPU only, no GPU needed)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
source venv/bin/activate
mkdir -p logs

export CUDA_VISIBLE_DEVICES=""

python scripts/run_full_pipeline.py \
  --condition D \
  --seeds 42,123,7 \
  --rounds 30 \
  --agents 500 \
  2>&1 | tee "logs/cond_d_$(date +%Y%m%d_%H%M%S).log"
