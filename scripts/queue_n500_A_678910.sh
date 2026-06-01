#!/usr/bin/env bash
# Queue: waits for one of the 3 currently-running n500 matrix jobs to
# exit, then launches A cond seeds 6-10 on the GPU it vacated.
# (GPU 1 is intentionally NOT claimed — released to another user.)
set -u
cd /mnt/sdb1/workspace/lucastourinho/SyntheticSocieties
source venv/bin/activate

# PID -> GPU index mapping (as of 2026-05-25 22:37)
declare -A PID2GPU=( [2761558]=0 [2761578]=2 [2761588]=3 )

echo "=== queued n500 cond=A seeds=6,7,8,9,10  ($(date)) ==="
echo "waiting for one of PIDs ${!PID2GPU[*]} to exit..."

while true; do
  for pid in "${!PID2GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      gpu="${PID2GPU[$pid]}"
      echo "=== PID $pid exited; claiming GPU $gpu ($(date)) ==="
      sleep 10  # let CUDA memory release
      export CUDA_VISIBLE_DEVICES="$gpu"
      export BGF_MAX_BATCH_SIZE=1
      export PYTHONHASHSEED=0
      exec python scripts/run_experiment_matrix.py \
        --include-llm --conditions A --seeds 6,7,8,9,10 \
        --rounds 30 --agents 500 --id-suffix n500 --skip-existing
    fi
  done
  sleep 60
done
