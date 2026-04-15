#!/usr/bin/env bash
# Phase 17 — Cross-Cultural ESS Validation Pipeline
# Runs 3 cluster simulations (Nordic / Southern / Eastern), computes the
# trust-cooperation Pearson r and Spearman rho, and generates a scatter plot.
#
# Usage:
#   bash pipeline_cross_cultural.sh                              # mock policy, 20 agents, 10 rounds
#   bash pipeline_cross_cultural.sh --dry-run                    # 5 agents, 3 rounds, synthetic data
#   bash pipeline_cross_cultural.sh --include-llm                # LLM policy, 1 seed (GPU required)
#   bash pipeline_cross_cultural.sh --include-llm --n-seeds 3    # LLM policy, 3 seeds (GPU required)
#   bash pipeline_cross_cultural.sh --include-llm --n-seeds 10   # LLM policy, 10 seeds (GPU required)
#
# Outputs:
#   analysis/cross_cultural_results.json
#   analysis/tables/cross_cultural_correlation.csv
#   analysis/figures/cross_cultural_validation.png
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source venv/bin/activate

python scripts/run_cross_cultural.py "$@"
python scripts/plot_cross_cultural_validation.py

echo ""
echo "Cross-cultural validation complete."
echo "  Results: analysis/cross_cultural_results.json"
echo "  Table:   analysis/tables/cross_cultural_correlation.csv"
echo "  Figure:  analysis/figures/cross_cultural_validation.png"
