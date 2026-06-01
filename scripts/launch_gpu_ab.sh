#!/usr/bin/env bash
# 10-seed A/B LLM comparison (GPU, pinned to device 0)
# Pre-registered confirmatory run: N=500, T=30, seeds 1-10
# Produces bootstrap 95% CIs and formal Mann-Whitney U for H1-H8.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
source venv/bin/activate
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0

python scripts/run_experiment_matrix.py --include-llm \
  --conditions A B \
  --seeds 1..10 \
  --rounds 30 \
  --agents 500 \
  --skip-existing \
  2>&1 | tee "logs/gpu_ab_n500_$(date +%Y%m%d_%H%M%S).log"
