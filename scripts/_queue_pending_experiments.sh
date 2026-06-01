#!/usr/bin/env bash
# =============================================================================
# Queue: pending paper experiments, runs after the N=500 LLM A/B sweep finishes
#
# Order (sequential, single-GPU to avoid the 13:25 simultaneous-init wedge):
#   1. Memory ablation (LLM policy)         → H8, §8.5  (~3 h)
#   2. Cross-cultural LLM (10 seeds)        → H9, §8.3  (~6–10 h)
#   3. Padded prompt control (N=100, 3 seeds) → AE1 closure, §8.1.4(ii), §7.9 (~6 h)
#   4. Cross-model comparison (Qwen + GPT-4o-mini)  → H7 cross-family, §6.6  (~4 h)
#
# Each step writes to its own logfile under experiments/_queue_logs/.
# Failures are logged but do not halt the queue (one bad run shouldn't block
# the others — they answer independent hypotheses).
#
# Trigger: this script blocks until the 4 N=500 cells (mx_{A,B}_n500_s{1,6})
# all have run_state.status ∈ {complete, failed}.
# =============================================================================
set -u
cd /mnt/sdb1/workspace/lucastourinho/SyntheticSocieties
source venv/bin/activate

# Pick GPU 0 for all queued runs. If a future change wants parallelism, override
# QUEUE_GPU per-job from the dispatcher.
QUEUE_GPU="${QUEUE_GPU:-0}"
LOGDIR="experiments/_queue_logs"
mkdir -p "$LOGDIR"
QLOG="$LOGDIR/queue.log"

log() { echo "[$(date -Is)] $*" | tee -a "$QLOG"; }

# ── Wait for the N=500 sweep to finish ─────────────────────────────────────
log "queue armed — waiting for N=500 LLM A/B sweep to complete"
N500_CELLS=(mx_A_n500_s1 mx_A_n500_s6 mx_B_n500_s1 mx_B_n500_s6)

n500_all_done() {
    for c in "${N500_CELLS[@]}"; do
        # Still running if a process exists for it
        if pgrep -f "run_config_simulation.*${c}" >/dev/null 2>&1; then
            return 1
        fi
    done
    return 0
}

while ! n500_all_done; do
    sleep 300  # 5-min poll; runs take many hours
done

log "N=500 sweep processes all gone — starting queue"
sleep 30  # give the OS a moment to release GPU memory

# ── 1. Memory ablation (LLM policy) ────────────────────────────────────────
log "[1/4] memory ablation (LLM policy) → $LOGDIR/memory_ablation.log"
CUDA_VISIBLE_DEVICES="$QUEUE_GPU" bash scripts/run_memory_ablation_llm.sh \
    > "$LOGDIR/memory_ablation.log" 2>&1 \
    && log "[1/4] memory ablation OK" \
    || log "[1/4] memory ablation FAILED rc=$?"

# ── 2. Cross-cultural LLM (10 seeds) ───────────────────────────────────────
log "[2/4] cross-cultural LLM (10 seeds) → $LOGDIR/cross_cultural.log"
CUDA_VISIBLE_DEVICES="$QUEUE_GPU" bash scripts/pipeline_cross_cultural.sh \
    --include-llm --n-seeds 10 \
    > "$LOGDIR/cross_cultural.log" 2>&1 \
    && log "[2/4] cross-cultural OK" \
    || log "[2/4] cross-cultural FAILED rc=$?"

# ── 3. Padded-prompt control at N=100 (AE1 closure) ────────────────────────
# Using N=100 not N=500 to keep queue duration reasonable. The paper's §8.1.4
# explicitly says "N=100 or N=500 is still pending"; N=100 is sufficient to
# discriminate "BRM trend is a length artefact" from "BRM trend survives".
log "[3/4] padded prompt control N=100, 3 seeds → $LOGDIR/padded_control.log"
CUDA_VISIBLE_DEVICES="$QUEUE_GPU" python scripts/run_padded_control.py \
    --seeds 42,123,7 --agents 100 --rounds 30 \
    > "$LOGDIR/padded_control.log" 2>&1 \
    && log "[3/4] padded control OK" \
    || log "[3/4] padded control FAILED rc=$?"

# ── 4. Cross-model comparison (Qwen + GPT-4o-mini) ─────────────────────────
log "[4/4] cross-model comparison (Qwen2.5-7B + GPT-4o-mini) → $LOGDIR/cross_model.log"
CUDA_VISIBLE_DEVICES="$QUEUE_GPU" python scripts/run_cross_model_comparison.py \
    --models qwen2.5-7b gpt4o_mini \
    > "$LOGDIR/cross_model.log" 2>&1 \
    && log "[4/4] cross-model OK" \
    || log "[4/4] cross-model FAILED rc=$?"

log "queue finished"
