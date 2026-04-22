#!/usr/bin/env bash
# 10-seed A/B LLM comparison (GPU, pinned to device 0)
set -euo pipefail
cd /mnt/sdb1/workspace/lucastourinho/SyntheticSocieties
source venv/bin/activate
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0

python scripts/run_experiment_matrix.py --include-llm \
  --conditions A B \
  --seeds 1..10 \
  --rounds 30 \
  --agents 50 \
  --skip-existing \
  2>&1 | tee "logs/gpu_ab_n50_$(date +%Y%m%d_%H%M%S).log"
