#!/usr/bin/env bash
# ============================================================
# gpu_wait_and_launch.sh — GPU availability watcher + auto-launcher
#
# Polls nvidia-smi at a configurable interval. When ALL visible GPUs
# have free memory above a threshold (default: 10 GiB) AND utilisation
# is below a ceiling (default: 15%), the queued job fires automatically.
#
# Default job: the cross-cultural LLM validation pipeline that the
# Phase 27 checklist marks PENDING:
#
#   bash scripts/pipeline_cross_cultural.sh --include-llm --n-seeds 10
#
# Override the job, thresholds, or GPU list via env vars or CLI flags.
#
# Usage examples:
#   # Default — watch GPU 0, launch cross-cultural LLM pipeline
#   nohup bash scripts/gpu_wait_and_launch.sh &
#
#   # Custom job, custom GPU, custom thresholds
#   GPU_INDICES="0,1" FREE_MEM_MB=12000 MAX_UTIL=10 \
#     bash scripts/gpu_wait_and_launch.sh \
#       -- python scripts/run_experiment_matrix.py --include-llm \
#          --conditions A B --seeds 1..10 --rounds 30 --agents 500
#
#   # Dry-run (prints what it would do, then exits)
#   bash scripts/gpu_wait_and_launch.sh --dry-run
#
# Env vars (all optional):
#   GPU_INDICES   — comma-separated GPU indices to watch  (default: 0)
#   FREE_MEM_MB   — min free VRAM in MiB per GPU          (default: 10240 ≈ 10 GiB)
#   MAX_UTIL      — max GPU utilisation % to consider idle (default: 15)
#   POLL_SECS     — seconds between nvidia-smi polls      (default: 60)
#   COOLDOWN_SECS — extra wait after threshold met         (default: 30)
#   MAX_WAIT_MINS — give up after this many minutes        (default: 1440 = 24 h)
# ============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
GPU_INDICES="${GPU_INDICES:-0}"
FREE_MEM_MB="${FREE_MEM_MB:-10240}"
MAX_UTIL="${MAX_UTIL:-15}"
POLL_SECS="${POLL_SECS:-60}"
COOLDOWN_SECS="${COOLDOWN_SECS:-30}"
MAX_WAIT_MINS="${MAX_WAIT_MINS:-1440}"
DRY_RUN=false

# ── Parse flags ───────────────────────────────────────────────────────────────
JOB_CMD=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY_RUN=true; shift ;;
    --gpu)      GPU_INDICES="$2"; shift 2 ;;
    --mem)      FREE_MEM_MB="$2"; shift 2 ;;
    --util)     MAX_UTIL="$2"; shift 2 ;;
    --poll)     POLL_SECS="$2"; shift 2 ;;
    --cooldown) COOLDOWN_SECS="$2"; shift 2 ;;
    --max-wait) MAX_WAIT_MINS="$2"; shift 2 ;;
    --)         shift; JOB_CMD=("$@"); break ;;
    *)          JOB_CMD=("$@"); break ;;
  esac
done

# Default job: the cross-cultural LLM pipeline (Phase 27 pending item)
if [[ ${#JOB_CMD[@]} -eq 0 ]]; then
  JOB_CMD=(bash scripts/pipeline_cross_cultural.sh --include-llm --n-seeds 10)
fi

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/gpu_watcher_$(date +%Y%m%d_%H%M%S).log"

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*" | tee -a "$LOG_FILE"
}

log "=== gpu_wait_and_launch.sh ==="
log "Repo:        $REPO_DIR"
log "GPU indices: $GPU_INDICES"
log "Free thresh: ${FREE_MEM_MB} MiB"
log "Util ceil:   ${MAX_UTIL}%"
log "Poll every:  ${POLL_SECS}s"
log "Cooldown:    ${COOLDOWN_SECS}s"
log "Max wait:    ${MAX_WAIT_MINS} min"
log "Job:         ${JOB_CMD[*]}"
log "Dry run:     $DRY_RUN"
log "Log file:    $LOG_FILE"
log ""

if $DRY_RUN; then
  log "[DRY-RUN] Would poll GPU(s) $GPU_INDICES until free ≥ ${FREE_MEM_MB} MiB"
  log "[DRY-RUN] and util ≤ ${MAX_UTIL}%, then run:"
  log "[DRY-RUN]   ${JOB_CMD[*]}"
  exit 0
fi

# ── GPU check function ───────────────────────────────────────────────────────
check_gpus_idle() {
  # Returns 0 (true) if ALL watched GPUs meet the idle criteria.
  # Returns 1 (false) if any GPU is busy or nvidia-smi fails.
  local idx free util
  IFS=',' read -ra INDICES <<< "$GPU_INDICES"
  for idx in "${INDICES[@]}"; do
    idx="$(echo "$idx" | xargs)"  # trim whitespace

    # Query free memory and utilisation for this specific GPU
    local gpu_info
    gpu_info=$(nvidia-smi --id="$idx" \
      --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null) || return 1

    free=$(echo "$gpu_info" | awk -F',' '{print int($1)}')
    util=$(echo "$gpu_info" | awk -F',' '{print int($2)}')

    if [[ "$free" -lt "$FREE_MEM_MB" ]]; then
      log "  GPU $idx: free=${free} MiB < ${FREE_MEM_MB} MiB — busy"
      return 1
    fi
    if [[ "$util" -gt "$MAX_UTIL" ]]; then
      log "  GPU $idx: util=${util}% > ${MAX_UTIL}% — busy"
      return 1
    fi
    log "  GPU $idx: free=${free} MiB, util=${util}% — idle ✓"
  done
  return 0
}

# ── Main polling loop ────────────────────────────────────────────────────────
ELAPSED=0
MAX_WAIT_SECS=$((MAX_WAIT_MINS * 60))

log "Starting GPU watch loop…"

while true; do
  if [[ "$ELAPSED" -ge "$MAX_WAIT_SECS" ]]; then
    log "⚠  Max wait time exceeded (${MAX_WAIT_MINS} min). Giving up."
    exit 1
  fi

  log "Polling GPU(s) $GPU_INDICES (elapsed: $((ELAPSED / 60))m)…"

  if check_gpus_idle; then
    log "All GPU(s) idle! Waiting ${COOLDOWN_SECS}s cooldown…"
    sleep "$COOLDOWN_SECS"

    # Re-check after cooldown to avoid transient dips
    if check_gpus_idle; then
      log "✅ GPU(s) confirmed idle after cooldown. Launching job."
      break
    else
      log "GPU(s) went busy during cooldown. Resuming poll."
    fi
  fi

  sleep "$POLL_SECS"
  ELAPSED=$((ELAPSED + POLL_SECS))
done

# ── Activate venv and launch ─────────────────────────────────────────────────
log ""
log "=============================================="
log "LAUNCHING: ${JOB_CMD[*]}"
log "=============================================="

# Set CUDA_VISIBLE_DEVICES to the watched GPUs
export CUDA_VISIBLE_DEVICES="$GPU_INDICES"
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Activate venv if present (the job script may also do this)
if [[ -f "$REPO_DIR/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_DIR/venv/bin/activate"
fi

# Run the job, tee-ing output to both terminal and log
JOB_LOG="$LOG_DIR/gpu_job_$(date +%Y%m%d_%H%M%S).log"
log "Job output → $JOB_LOG"

{
  "${JOB_CMD[@]}" 2>&1 | tee "$JOB_LOG"
  JOB_EXIT=${PIPESTATUS[0]}
} || JOB_EXIT=$?

log ""
if [[ "${JOB_EXIT:-1}" -eq 0 ]]; then
  log "✅ Job completed successfully (exit 0)."
else
  log "❌ Job failed with exit code ${JOB_EXIT:-unknown}."
fi

exit "${JOB_EXIT:-1}"
