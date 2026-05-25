#!/usr/bin/env bash
# Queue: waits until ALL GPU matrix runs (A 1-5, B 1-5, B 6-10, and the
# A 6-10 queued behind them) have exited, then launches Condition C
# (generative_agents) n=500 seeds 1-10 on the first non-reserved GPU.
# GPU 1 is excluded — it was released to another user.
set -u
cd /mnt/sdb1/workspace/lucastourinho/SyntheticSocieties
source venv/bin/activate

RESERVED_GPU=1

echo "=== queued n500 cond=C seeds=1..10  ($(date)) ==="
echo "waiting until no '--include-llm' matrix python processes remain..."

while true; do
  if pgrep -af 'python .*run_experiment_matrix.py.*--include-llm' >/dev/null; then
    sleep 120
    continue
  fi
  # also wait until the A6-10 queue tmux session is gone (script may be
  # between PID-exit and child-launch)
  if tmux has-session -t queue-n500-A-678910 2>/dev/null; then
    sleep 60
    continue
  fi
  break
done

echo "=== all GPU matrix runs finished ($(date)) ==="
sleep 15

# pick first GPU != RESERVED_GPU with <1000 MiB used
target=""
while IFS=, read -r idx mem; do
  idx="${idx// /}"; mem="${mem// /}"; mem="${mem%MiB}"
  [ "$idx" = "$RESERVED_GPU" ] && continue
  if [ "${mem:-99999}" -lt 1000 ]; then target="$idx"; break; fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)

if [ -z "$target" ]; then
  echo "no free GPU after wait — falling back to GPU 0"
  target=0
fi

echo "=== launching cond=C on GPU $target ($(date)) ==="
export CUDA_VISIBLE_DEVICES="$target"
export BGF_MAX_BATCH_SIZE=1
export PYTHONHASHSEED=0
exec python scripts/run_experiment_matrix.py \
  --include-llm --conditions C --seeds 1,2,3,4,5,6,7,8,9,10 \
  --rounds 30 --agents 500 --id-suffix n500 --skip-existing
