#!/usr/bin/env bash
# GPU job queue runner.
# Usage: bash run_queue.sh [GPU_INDEX]
# Default GPU=0. Run in two terminals with GPU=0 and GPU=1 for dual-GPU parallelism.

set -euo pipefail

PROJECT_DIR="/mnt/sdb1/workspace/lucastourinho/SyntheticSocieties"
VENV="$PROJECT_DIR/venv/bin/activate"
LOGS_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOGS_DIR"

# ── Job queue ─────────────────────────────────────────────────────────────────
# Format: "CMD | logfile"  (logfile saved under logs/)
JOBS=(
  "scripts/run_experiment_matrix.py --skip-existing | matrix_run.txt"
  "scripts/run_padded_control.py --seeds 42,123,7 --agents 500 --rounds 30 | padded_control.txt"
  "scripts/run_cross_model_comparison.py --models qwen2.5-7b gpt4o-mini | cross_model.txt"
)

QUEUE_FILE="/tmp/bgf_gpu_job_queue.txt"
LOCK_FILE="/tmp/bgf_gpu_job_queue.lock"

# Populate queue only if it doesn't already exist
if [ ! -f "$QUEUE_FILE" ]; then
  printf "%s\n" "${JOBS[@]}" > "$QUEUE_FILE"
fi

run_job() {
  local GPU=$1
  local CMD=$2
  local LOG=$3
  CMD=$(echo "$CMD" | xargs)  # trim whitespace
  LOG=$(echo "$LOG" | xargs)  # trim whitespace

  echo "[START] GPU=$GPU  CMD=$CMD  LOG=$LOGS_DIR/$LOG"
  cd "$PROJECT_DIR"
  source "$VENV"
  CUDA_VISIBLE_DEVICES=$GPU python3 -u $CMD 2>&1 | tee "$LOGS_DIR/$LOG"
  echo "[DONE]  GPU=$GPU  CMD=$CMD"
}

worker() {
  local GPU=$1

  while true; do
    # Atomic pop: grab and remove the first line under a file lock
    local JOB
    JOB=$(
      flock -x "$LOCK_FILE" bash -c "
        head -n1 '$QUEUE_FILE' 2>/dev/null || true
        tail -n +2 '$QUEUE_FILE' > '${QUEUE_FILE}.tmp' 2>/dev/null || true
        mv '${QUEUE_FILE}.tmp' '$QUEUE_FILE' 2>/dev/null || true
      "
    )

    if [ -z "$JOB" ]; then
      echo "GPU $GPU: queue empty, exiting."
      break
    fi

    CMD=$(echo "$JOB" | cut -d'|' -f1)
    LOG=$(echo "$JOB" | cut -d'|' -f2)

    run_job "$GPU" "$CMD" "$LOG"
  done
}

worker "${1:-0}"
