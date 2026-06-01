#!/usr/bin/env bash
# Serial resume of the 4 stalled N=500 LLM cells on GPU 0 only.
# Serial chosen because prior parallel attempts were SIGKILL'd (rc=137).
# GPU 2 is in use by another user — do not touch.
set -u
cd /mnt/sdb1/workspace/lucastourinho/SyntheticSocieties
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
LOGDIR=experiments/_resume_logs_n500
mkdir -p "$LOGDIR"
# Order: highest progress first so a re-crash costs the least re-work
CELLS=(mx_A_n500_s1 mx_B_n500_s1 mx_B_n500_s6 mx_A_n500_s6)
for cell in "${CELLS[@]}"; do
  logfile="$LOGDIR/${cell}.log"
  [ -f "$logfile" ] && mv "$logfile" "${logfile}.prev4" 2>/dev/null || true
  echo "============================================================" >> "$LOGDIR/serial.log"
  echo "[$(date -Is)] RESUME $cell on GPU 0" >> "$LOGDIR/serial.log"
  python scripts/run_config_simulation.py \
      --config "experiments/$cell/config.yaml" \
      --resume "$cell" \
      > "$logfile" 2>&1
  rc=$?
  echo "[$(date -Is)] $cell rc=$rc" >> "$LOGDIR/serial.log"
done
echo "[$(date -Is)] all 4 done" >> "$LOGDIR/serial.log"
