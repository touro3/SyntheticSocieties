#!/usr/bin/env bash
# Launch the 4 n=500 resume runs sequentially, with a gap between each,
# so they don't all hit CUDA init at the same instant.
# Order: highest-progress first so a bad-run-early failure costs least work.
set -u
cd /mnt/sdb1/workspace/lucastourinho/SyntheticSocieties
source venv/bin/activate
mkdir -p experiments/_resume_logs_n500
GAP_SECONDS=${GAP_SECONDS:-120}

launch() {
    local gpu="$1" cell="$2"
    local logfile="experiments/_resume_logs_n500/${cell}.log"
    # rotate old log
    [ -f "$logfile" ] && mv "$logfile" "${logfile}.prev3" 2>/dev/null || true
    echo "[$(date -Is)] launching $cell on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python scripts/run_config_simulation.py \
        --config "experiments/${cell}/config.yaml" \
        --resume "${cell}" \
        > "$logfile" 2>&1 &
    local pid=$!
    echo "[$(date -Is)] $cell pid=$pid"
    # wait for model load to finish before launching next
    sleep "$GAP_SECONDS"
}

launch 1 mx_B_n500_s1   # highest progress (round 21)
launch 2 mx_B_n500_s6   # round 16
launch 3 mx_A_n500_s1   # round 23
launch 0 mx_A_n500_s6   # round 8

echo "[$(date -Is)] all 4 launched"
