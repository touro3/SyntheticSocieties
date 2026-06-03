#!/usr/bin/env bash
# =============================================================================
# run_memory_ablation_llm.sh — Re-run memory ablation with REAL LLM policy
#
# This script re-runs all 24 memory ablation experiments (M0–M3 × 2 conditions
# × 3 seeds) using the actual LLM policy. The existing on-disk experiments used
# policy=mock and produce no persona fidelity variation.
#
# Paper claim: M0 fidelity=0.609 → M3 fidelity=0.742 (monotonic under grounding)
# Status: UNVERIFIED — needs this run to substantiate Table 7 and Figure 15
#
# Requirements: GPU with ≥4 GB VRAM, Mistral-7B-Instruct-v0.3 cached
# Estimated runtime: ~2–4 hours on single P100 (4-bit quantization)
#
# Usage:
#   source venv/bin/activate
#   bash scripts/run_memory_ablation_llm.sh [--dry-run]
#
# Output: experiments/ablation_M{0-3}_{grounded,ungrounded}_s{42,123,7}/
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SEEDS=(42 123 7)
MEMORY_LEVELS=(0 1 2 3)
MEMORY_CONFIGS=(
    "configs/memory_ablation/m0_no_memory.yaml"
    "configs/memory_ablation/m1_window_only.yaml"
    "configs/memory_ablation/m2_window_archive.yaml"
    "configs/memory_ablation/m3_full.yaml"
)
MEMORY_NAMES=("M0_no_memory" "M1_window" "M2_archive" "M3_full")

N_AGENTS=20
N_ROUNDS=10   # Short horizon to keep runtime manageable per seed; extend to 30 for final

echo "========================================================"
echo "  BGF Memory Ablation Re-run — LLM Policy"
echo "  Levels: M0–M3 | Seeds: ${SEEDS[*]} | N=${N_AGENTS} T=${N_ROUNDS}"
echo "  $(date)"
echo "========================================================"

for level_idx in "${!MEMORY_LEVELS[@]}"; do
    level="${MEMORY_LEVELS[$level_idx]}"
    cfg="${MEMORY_CONFIGS[$level_idx]}"
    name="${MEMORY_NAMES[$level_idx]}"

    for condition in grounded ungrounded; do
        # Map condition to the new A/B/C/D pipeline flags
        if [[ "$condition" == "grounded" ]]; then
            condition_flag="B"  # Condition B is Empirical/Grounded
        else
            condition_flag="A"  # Condition A is Synthetic/Ungrounded
        fi

        for seed in "${SEEDS[@]}"; do
            exp_id="ablation_${name}_${condition}_s${seed}"
            out_dir="experiments/${exp_id}"

            echo ""
            echo "── M${level} | ${condition} | seed=${seed} → ${exp_id}"

            if [[ -d "$out_dir" && -f "$out_dir/metrics.json" ]]; then
                echo "   SKIP: already complete"
                continue
            fi

            # CORRECTED COMMAND BLOCK:
            cmd=(
                python scripts/run_full_pipeline.py
                --condition "$condition_flag"
                --llm-ablation-level "$level"
                --seeds "$seed"
                --agents "$N_AGENTS"
                --rounds "$N_ROUNDS"
                --include-llm
                --skip-existing
            )

            if $DRY_RUN; then
                echo "   DRY-RUN: ${cmd[*]}"
            else
                echo "   Running: ${cmd[*]}"
                "${cmd[@]}" 2>&1 | tee "logs/${exp_id}.log"
                echo "   Done: $(date)"
            fi
        done
    done
done

echo ""
echo "========================================================"
echo "  All memory ablation runs complete."
echo "  Analyze results:"
echo "    python scripts/analyze_memory_ablation.py"
echo "    python scripts/plot_memory_ablation_heatmap.py"
echo "========================================================"
