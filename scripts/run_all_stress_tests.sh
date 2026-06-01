#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "======================================================"
echo "🚀 STARTING THE ULTIMATE BGF STRESS TEST MARATHON"
echo "======================================================"

echo "Running Experiment 1/3: Bad Apple..."
bash scripts/pipeline_bad_apple.sh

echo "Running Experiment 2/3: Macroeconomic Shock..."
bash scripts/pipeline_macro_shock.sh

echo "Running Experiment 3/3: Topological Dictatorship..."
bash scripts/pipeline_topology.sh

echo "======================================================"
echo "🏆 ALL EXPERIMENTS COMPLETED SUCCESSFULLY! TIME TO PUBLISH!"
echo "======================================================"
