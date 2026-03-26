#!/bin/bash
set -e
set -o pipefail

echo "======================================================"
echo " STARTING BAD APPLE EXPERIMENT PIPELINE"
echo "======================================================"

source venv/bin/activate
export PYTHONPATH="."

echo ""
echo "[1/2] Running the Adversarial Simulations (500 Agents, 5% Bad Apples)..."
python scripts/run_bad_apple.py --pop-size 500 --rounds 30 --injection-rate 0.05

echo ""
echo "[2/2] Generating Resilience and Vulnerability Plots..."
python scripts/plot_bad_apple.py

echo ""
echo "✅ BAD APPLE PIPELINE COMPLETED SUCCESSFULLY!"
echo "📍 Check analysis/figures/bad_apple_resilience.png for the results."
echo "======================================================"
