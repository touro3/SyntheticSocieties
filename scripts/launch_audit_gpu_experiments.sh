#!/usr/bin/env bash
# =============================================================================
# GPU Experiment Launcher — Audit-Required Runs
# =============================================================================
#
# This script launches the GPU-blocked experiments identified in the paper
# audit (2026-05-19) as tmux sessions. Each experiment runs independently.
#
# PREREQUISITES:
#   - CUDA-capable GPU with >= 16 GB VRAM (Tesla P100/V100/A100)
#   - Mistral-7B-Instruct-v0.3 model available (HuggingFace cache or local)
#   - Python venv activated with project dependencies
#   - tmux installed
#
# USAGE:
#   chmod +x scripts/launch_audit_gpu_experiments.sh
#   bash scripts/launch_audit_gpu_experiments.sh
#
# Each session can be attached with: tmux attach -t <session-name>
# Monitor all sessions: tmux ls
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "============================================="
echo "  BGF Audit — GPU Experiment Launcher"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
echo ""

# -------------------------------------------------------------------------
# 1. CRITICAL — Memory Ablation LLM Re-run (Audit A.9)
#    Replaces the mock-policy ablation data with real LLM inference.
#    Table 7 in paper.md is currently "pre-registered prediction" only.
#    ~6-8 GPU-hours
# -------------------------------------------------------------------------
SESSION_1="audit-memory-ablation"
echo "[1/4] Launching: ${SESSION_1} (audit row A.9 — ~6-8 GPU-h)"
tmux new-session -d -s "${SESSION_1}" -c "${PROJECT_ROOT}" \
  "echo '=== Memory Ablation LLM Re-run (A.9) ===' && \
   echo 'Start: '$(date) && \
   bash scripts/run_memory_ablation_llm.sh 2>&1 | tee logs/audit_memory_ablation_$(date +%Y%m%d_%H%M%S).log && \
   echo '' && echo 'Done: '$(date) && \
   echo 'Next: run python3 scripts/analyze_memory_ablation.py to regenerate analysis/tables/memory_ablation.json' && \
   bash -i" || echo "  ⚠ Session ${SESSION_1} may already exist (tmux ls to check)"

# -------------------------------------------------------------------------
# 2. CRITICAL — Padded Prompt Control at Primary Scale (Audit C.5)
#    Isolates prompt-length confound. N=500, T=30, 3 seeds.
#    ~6 GPU-hours
# -------------------------------------------------------------------------
SESSION_2="audit-padded-control"
echo "[2/4] Launching: ${SESSION_2} (audit row C.5 — ~6 GPU-h)"
tmux new-session -d -s "${SESSION_2}" -c "${PROJECT_ROOT}" \
  "echo '=== Padded Prompt Control N=500 T=30 (C.5) ===' && \
   echo 'Start: '$(date) && \
   python3 scripts/run_padded_control.py \
     --n-agents 500 \
     --n-rounds 30 \
     --seeds 42 43 44 \
     2>&1 | tee logs/audit_padded_control_$(date +%Y%m%d_%H%M%S).log && \
   echo '' && echo 'Done: '$(date) && \
   bash -i" || echo "  ⚠ Session ${SESSION_2} may already exist"

# -------------------------------------------------------------------------
# 3. HIGH — Cross-Cultural LLM-Scale Replication (Audit A.1/A.2/A.3)
#    Current results are dry_run rule-based proxy only.
#    Validates the Spearman ρ=1.000 claim under actual LLM inference.
#    ~12-18 GPU-hours (6 clusters × 3 seeds)
# -------------------------------------------------------------------------
SESSION_3="audit-cross-cultural-llm"
echo "[3/4] Launching: ${SESSION_3} (audit rows A.1-A.3 — ~12-18 GPU-h)"
tmux new-session -d -s "${SESSION_3}" -c "${PROJECT_ROOT}" \
  "echo '=== Cross-Cultural LLM Replication (A.1-A.3) ===' && \
   echo 'Start: '$(date) && \
   bash scripts/pipeline_cross_cultural.sh --include-llm --n-seeds 3 \
     2>&1 | tee logs/audit_cross_cultural_llm_$(date +%Y%m%d_%H%M%S).log && \
   echo '' && echo 'Done: '$(date) && \
   bash -i" || echo "  ⚠ Session ${SESSION_3} may already exist"

# -------------------------------------------------------------------------
# 4. HIGH — Re-run phase_c_comparison T=30 Pilot (Audit A.10/A.11/C3)
#    The original experiment directory is MISSING from disk.
#    This re-runs the canonical N=50, T=30 pilot to recover the raw
#    event logs needed to recompute B_RLHF from the action triplet.
#    ~2-3 GPU-hours
# -------------------------------------------------------------------------
SESSION_4="audit-phase-c-rerun"
echo "[4/4] Launching: ${SESSION_4} (audit rows A.10-A.14 — ~2-3 GPU-h)"
tmux new-session -d -s "${SESSION_4}" -c "${PROJECT_ROOT}" \
  "echo '=== Phase C T=30 Pilot Re-run (A.10-A.14) ===' && \
   echo 'Start: '$(date) && \
   echo 'Running Condition A (ungrounded)...' && \
   python3 -m simulation.runner \
     --experiment-id phase_c_comparison_A \
     --n-agents 50 \
     --n-rounds 30 \
     --seed 42 \
     --policy llm \
     --grounding none \
     --output-dir experiments/phase_c_comparison_A_s42 \
     2>&1 | tee logs/audit_phase_c_A_$(date +%Y%m%d_%H%M%S).log && \
   echo 'Running Condition B (grounded)...' && \
   python3 -m simulation.runner \
     --experiment-id phase_c_comparison_B \
     --n-agents 50 \
     --n-rounds 30 \
     --seed 42 \
     --policy llm \
     --grounding ess \
     --output-dir experiments/phase_c_comparison_B_s42 \
     2>&1 | tee logs/audit_phase_c_B_$(date +%Y%m%d_%H%M%S).log && \
   echo '' && echo 'Done: '$(date) && \
   echo 'Next steps:' && \
   echo '  1. python3 scripts/compute_paper_numbers.py  (refresh paper_numbers.json)' && \
   echo '  2. python3 -c \"from metrics.behavioral_realism import compute_brlhf; ...\"  (recompute B_RLHF from raw action triplet)' && \
   bash -i" || echo "  ⚠ Session ${SESSION_4} may already exist"

echo ""
echo "============================================="
echo "  All sessions launched!"
echo "============================================="
echo ""
echo "Monitor sessions:"
echo "  tmux ls                              # list all sessions"
echo "  tmux attach -t audit-memory-ablation # attach to session"
echo "  Ctrl-B D                             # detach from session"
echo ""
echo "After completion, run these post-processing steps:"
echo ""
echo "  # 1. Regenerate memory ablation table"
echo "  python3 scripts/analyze_memory_ablation.py"
echo ""
echo "  # 2. Recompute paper numbers (picks up new phase_c data)"
echo "  python3 scripts/compute_paper_numbers.py"
echo ""
echo "  # 3. Recompute B_RLHF from raw action triplets"
echo "  python3 -c \""
echo "    import json"
echo "    from metrics.behavioral_realism import compute_brlhf"
echo "    # Load events.jsonl from phase_c re-runs"
echo "    # Compute B_RLHF = TV(pi, pi_uniform) from the actual (work, save, coop) counts"
echo "    print('Update paper.md §6.1, abstract, and Figure 2 caption with corrected values')"
echo "  \""
echo ""
echo "  # 4. Update paper.md with corrected B_RLHF values"
echo "  #    Replace 0.712/0.420 with the recomputed values"
echo "  #    Update the reduction percentage in the abstract"
echo ""
echo "Estimated total GPU time: ~26-35 hours (runs in parallel across 4 tmux sessions)"
echo "With a single GPU: run sessions sequentially by waiting for each to finish."
