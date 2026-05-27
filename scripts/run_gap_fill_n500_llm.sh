#!/usr/bin/env bash
# Sequentially run the 16 missing n=500 LLM cells (Conditions A + B, seeds 2-5, 7-10).
# Requires ~24-40 GPU-hours on a P100 16GB at max_batch_size=2.
#
# DO NOT launch this while another LLM run is using GPU 1 (mistral-7b 4-bit takes ~11-14 GB).
# Run after scripts/resume_n500_stalled.sh has fully completed.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_gap_fill_n500_llm.sh \
#       > experiments/_gap_fill_n500_llm/runner.log 2>&1 &
set -u
cd /mnt/sdb1/workspace/lucastourinho/SyntheticSocieties
source venv/bin/activate

LOGDIR=experiments/_gap_fill_n500_llm
mkdir -p "$LOGDIR"

# Cell order: interleave A/B so partial results still let us compare conditions.
CELLS=(
  mx_A_n500_s2 mx_B_n500_s2
  mx_A_n500_s3 mx_B_n500_s3
  mx_A_n500_s4 mx_B_n500_s4
  mx_A_n500_s5 mx_B_n500_s5
  mx_A_n500_s7 mx_B_n500_s7
  mx_A_n500_s8 mx_B_n500_s8
  mx_A_n500_s9 mx_B_n500_s9
  mx_A_n500_s10 mx_B_n500_s10
)

for cell in "${CELLS[@]}"; do
  cfg="configs/gap_fill_n500_llm/${cell}.yaml"
  log="$LOGDIR/${cell}.log"
  echo "============================================================"
  echo "[$(date -Is)] START $cell"
  echo "  config: $cfg"
  echo "  log:    $log"
  echo "============================================================"
  python scripts/run_config_simulation.py --config "$cfg" > "$log" 2>&1
  rc=$?
  echo "[$(date -Is)] $cell exited rc=$rc"
  if [ $rc -ne 0 ]; then
    echo "[$(date -Is)] tail of $cell.log:"
    tail -30 "$log"
  fi
done
echo "[$(date -Is)] ALL 16 n=500 LLM CELLS DONE"
