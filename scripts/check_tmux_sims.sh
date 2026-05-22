#!/bin/bash
# Snapshot of tmux simulation sessions: tail of each pane, GPU state,
# running pipeline processes, and fresh heartbeats.
#
# Usage:
#   ./scripts/check_tmux_sims.sh                   # print to stdout
#   ./scripts/check_tmux_sims.sh -o sim_check.log  # write to log file
#   ./scripts/check_tmux_sims.sh -t                # append to logs/tmux_check_<ts>.log
#   ./scripts/check_tmux_sims.sh -n 30             # show last 30 lines per session (default 15)
#   ./scripts/check_tmux_sims.sh -m 10             # heartbeats updated within 10 min (default 5)
#   ./scripts/check_tmux_sims.sh -w 60             # repeat every 60s until Ctrl-C

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT=""
TAIL_LINES=15
HEARTBEAT_MIN=5
WATCH=0

while getopts "o:tn:m:w:h" opt; do
  case "$opt" in
    o) OUT="$OPTARG" ;;
    t) mkdir -p logs && OUT="logs/tmux_check_$(date +%Y%m%d_%H%M%S).log" ;;
    n) TAIL_LINES="$OPTARG" ;;
    m) HEARTBEAT_MIN="$OPTARG" ;;
    w) WATCH="$OPTARG" ;;
    h)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "Unknown flag. -h for help." >&2; exit 2 ;;
  esac
done

emit() {
  if [ -n "$OUT" ]; then
    "$@" | tee -a "$OUT"
  else
    "$@"
  fi
}

snapshot() {
  if [ -n "$OUT" ]; then : > "$OUT"; fi

  {
    echo "============================================================"
    echo "  TMUX SIMULATION SNAPSHOT — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  host=$(hostname)  cwd=$(pwd)"
    echo "============================================================"
    echo

    echo "── TMUX SESSIONS ──"
    if ! tmux ls 2>/dev/null; then
      echo "  (no tmux server running)"
    fi
    echo

    echo "── GPU STATE ──"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
                 --format=csv,noheader 2>&1 | sed 's/^/  /'
    else
      echo "  nvidia-smi not available"
    fi
    echo

    echo "── PIPELINE PROCESSES ──"
    PS_OUT=$(ps -eo pid,etime,pcpu,pmem,cmd 2>/dev/null \
      | grep -E "run_experiment_matrix|run_full_pipeline|run_config_simulation" \
      | grep -v grep || true)
    if [ -z "$PS_OUT" ]; then
      echo "  (none)"
    else
      echo "$PS_OUT" | awk '{
        pid=$1; etime=$2; cpu=$3; mem=$4;
        cmd=""; for (i=5;i<=NF;i++) cmd=cmd" "$i;
        if (length(cmd) > 180) cmd=substr(cmd,1,177)"...";
        printf "  pid=%-7s etime=%-12s cpu=%-5s mem=%-5s%s\n", pid, etime, cpu, mem, cmd;
      }'
    fi
    echo

    echo "── PER-SESSION TAIL (last $TAIL_LINES lines) ──"
    SESSIONS=$(tmux ls 2>/dev/null | cut -d: -f1)
    if [ -z "$SESSIONS" ]; then
      echo "  (no sessions)"
    else
      for s in $SESSIONS; do
        echo
        echo "  ===== $s ====="
        tmux capture-pane -t "$s" -p 2>/dev/null | tail -n "$TAIL_LINES" | sed 's/^/    /'
      done
    fi
    echo

    echo "── FRESH HEARTBEATS (updated within ${HEARTBEAT_MIN} min) ──"
    NOW=$(date +%s)
    FOUND=0
    while IFS= read -r f; do
      [ -z "$f" ] && continue
      FOUND=1
      mtime=$(stat -c %Y "$f")
      age=$((NOW - mtime))
      exp=$(basename "$(dirname "$f")")
      content=$(cat "$f" 2>/dev/null | tr -d '\n' | head -c 240)
      printf "  [%-40s age=%4ds]  %s\n" "$exp" "$age" "$content"
    done < <(find experiments -maxdepth 2 -name heartbeat.json -mmin -"$HEARTBEAT_MIN" 2>/dev/null | sort)
    if [ "$FOUND" -eq 0 ]; then
      echo "  (no heartbeats updated within ${HEARTBEAT_MIN} min)"
    fi
    echo

    echo "── RECENTLY COMPLETED EXPERIMENTS (mtime < ${HEARTBEAT_MIN} min) ──"
    FOUND=0
    while IFS= read -r d; do
      [ -z "$d" ] && continue
      # Only count as "complete" if it has metrics.json or summary.json
      if [ -f "$d/metrics.json" ] || [ -f "$d/summary.json" ]; then
        FOUND=1
        printf "  %s\n" "$d"
      fi
    done < <(find experiments -maxdepth 1 -type d -mmin -"$HEARTBEAT_MIN" 2>/dev/null | sort)
    if [ "$FOUND" -eq 0 ]; then
      echo "  (none in window)"
    fi
    echo

    echo "── DISK USAGE ──"
    df -h experiments 2>/dev/null | sed 's/^/  /' | head -2
    EXP_SIZE=$(du -sh experiments 2>/dev/null | cut -f1)
    EXP_COUNT=$(find experiments -maxdepth 1 -type d | wc -l)
    echo "  experiments/: ${EXP_SIZE:-?}  ($((EXP_COUNT - 1)) runs)"
    echo
  } | { [ -n "$OUT" ] && tee -a "$OUT" || cat; }
}

if [ "$WATCH" -gt 0 ] 2>/dev/null; then
  while true; do
    snapshot
    echo "(next refresh in ${WATCH}s — Ctrl-C to stop)"
    sleep "$WATCH"
  done
else
  snapshot
  if [ -n "$OUT" ]; then
    echo
    echo "Wrote snapshot to: $OUT"
  fi
fi
