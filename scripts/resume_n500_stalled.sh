#!/usr/bin/env bash
# Resume the four mid-flight N=500 LLM-scale cells from their on-disk
# checkpoint.json files (kernel.load_checkpoint). Runs serially to avoid
# GPU memory contention from device_map=auto across simultaneous Mistral-7B
# instances.
#
# Cells (snapshot 2026-05-26 17:42 CEST):
#   mx_A_n500_s1  → round 16 / 30
#   mx_B_n500_s1  → round 16 / 30
#   mx_B_n500_s6  → round 16 / 30
#   mx_A_n500_s6  → round  8 / 30
#
# Run from repo root inside the venv.
set -u
cd /mnt/sdb1/workspace/lucastourinho/SyntheticSocieties
source venv/bin/activate

LOGDIR=experiments/_resume_logs_n500
mkdir -p "$LOGDIR"

CELLS=(mx_A_n500_s1 mx_B_n500_s1 mx_B_n500_s6 mx_A_n500_s6)

for cell in "${CELLS[@]}"; do
  echo "============================================================"
  echo "[$(date -Is)] RESUME $cell"
  echo "  config:    experiments/$cell/config.yaml"
  echo "  log:       $LOGDIR/$cell.log"
  echo "============================================================"
  python scripts/run_config_simulation.py \
      --config "experiments/$cell/config.yaml" \
      --resume "$cell" \
      > "$LOGDIR/$cell.log" 2>&1
  rc=$?
  echo "[$(date -Is)] $cell exited rc=$rc"
  if [ $rc -ne 0 ]; then
    echo "[$(date -Is)] tail of $cell.log:"
    tail -40 "$LOGDIR/$cell.log"
  fi
done

echo "[$(date -Is)] ALL FOUR CELLS DONE"
