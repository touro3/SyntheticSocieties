#!/usr/bin/env bash
# =============================================================================
# run_memory_ablation_llm.sh — Memory ablation with REAL LLM policy (v2)
#
# Tests H8: persona fidelity monotonic in memory depth (M0 → M3).
# Runs 24 cells: M0–M3 × grounded/ungrounded × seeds {42, 123, 7}, N=20, T=10.
#
# BUGS PATCHED (2026-06-03, see §8.5.1 of paper.md):
#   Bug A: ablation.mode=no_rag was silently ignored in policy.type=llm branch.
#          Fix: run_config_simulation.py now reads ablation.mode and strips
#          graph_rag/sql_rag from LLMPolicy when mode=no_rag.
#   Bug B: memory.level config was not propagated to HierarchicalMemory.__init__.
#          Fix: population/generator.py now reads config["memory"]["level"] and
#          passes it to _build_memory(), so M0 agents truly have no memory
#          context shown to the LLM.
#
# Output: experiments/ablation_M{0-3}_{grounded,ungrounded}_s{42,123,7}_v2/
#         (v2 suffix distinguishes from the invalid 2026-06-03 run)
#
# Usage:
#   source venv/bin/activate
#   bash scripts/run_memory_ablation_llm.sh [--dry-run] [--skip-existing]
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=false
SKIP_EXISTING=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]]      && DRY_RUN=true
    [[ "$arg" == "--skip-existing" ]] && SKIP_EXISTING=true
done

PYTHON="${PYTHON:-venv/bin/python}"
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
N_ROUNDS=10

echo "========================================================"
echo "  BGF Memory Ablation — LLM Policy (v2, bugs patched)"
echo "  Levels: M0–M3 | Seeds: ${SEEDS[*]} | N=${N_AGENTS} T=${N_ROUNDS}"
echo "  $(date)"
echo "========================================================"

TOTAL=0
SKIPPED=0
FAILED=0

for level_idx in "${!MEMORY_LEVELS[@]}"; do
    level="${MEMORY_LEVELS[$level_idx]}"
    cfg="${MEMORY_CONFIGS[$level_idx]}"
    name="${MEMORY_NAMES[$level_idx]}"

    for condition in grounded ungrounded; do
        for seed in "${SEEDS[@]}"; do
            exp_id="ablation_${name}_${condition}_s${seed}_v2"
            out_dir="experiments/${exp_id}"
            TOTAL=$((TOTAL + 1))

            echo ""
            echo "── M${level} | ${condition} | seed=${seed} → ${exp_id}"

            # Skip if summary.json already written (previous complete v2 run)
            if $SKIP_EXISTING && [[ -f "$out_dir/summary.json" ]]; then
                echo "   SKIP: summary.json exists"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            # Build override list
            overrides=(
                "project.experiment_id=${exp_id}"
                "project.seed=${seed}"
                "simulation.population_size=${N_AGENTS}"
                "simulation.rounds=${N_ROUNDS}"
                "policy.type=llm"
            )

            # Bug A fix: pass ablation.mode=no_rag for ungrounded arm.
            # run_config_simulation.py now routes this to graph_rag=None, sql_rag=None.
            if [[ "$condition" == "ungrounded" ]]; then
                overrides+=("ablation.mode=no_rag")
            fi

            # Bug B fix: memory.level is already in the yaml config (m0=0, m1=1, etc.)
            # and population/generator.py now reads it — no additional override needed.

            cmd=(
                "$PYTHON" scripts/run_config_simulation.py
                --config "$cfg"
                "${overrides[@]}"
            )

            if $DRY_RUN; then
                echo "   DRY-RUN: ${cmd[*]}"
            else
                echo "   Running: ${cmd[*]}"
                mkdir -p logs
                if "${cmd[@]}" 2>&1 | tee "logs/${exp_id}.log"; then
                    echo "   Done: $(date)"
                else
                    echo "   FAILED: see logs/${exp_id}.log"
                    FAILED=$((FAILED + 1))
                fi
            fi
        done
    done
done

echo ""
echo "========================================================"
echo "  Memory ablation v2 complete."
echo "  Total: ${TOTAL} | Skipped: ${SKIPPED} | Failed: ${FAILED}"
echo "  Analyze results:"
echo "    python scripts/analyze_memory_ablation.py"
echo "========================================================"
