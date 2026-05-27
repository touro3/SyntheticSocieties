#!/usr/bin/env bash
# Sequentially run Condition D n=100 seeds 4-10 (rule_based_ess, no LLM, CPU-only).
set -u
cd /mnt/sdb1/workspace/lucastourinho/SyntheticSocieties
source venv/bin/activate

LOGDIR=experiments/_gap_fill_d_n100
mkdir -p "$LOGDIR"

# Force CPU-only — no LLM, no point burning GPU contention.
export CUDA_VISIBLE_DEVICES=""

for s in 4 5 6 7 8 9 10; do
  cfg="configs/gap_fill_d_n100/mx_D_s${s}.yaml"
  log="$LOGDIR/mx_D_s${s}.log"
  echo "[$(date -Is)] START mx_D_s${s}"
  python scripts/run_config_simulation.py --config "$cfg" > "$log" 2>&1
  rc=$?
  echo "[$(date -Is)] mx_D_s${s} exited rc=$rc"
done
echo "[$(date -Is)] ALL D n=100 SEEDS DONE"
