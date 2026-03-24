#!/bin/bash
set -e

echo "======================================================"
echo "🚀 STARTING THE ULTIMATE BGF STRESS TEST MARATHON"
echo "======================================================"

echo "Running Experiment 1/3: Bad Apple..."
./pipeline_bad_apple.sh

echo "Running Experiment 2/3: Macroeconomic Shock..."
./pipeline_macro_shock.sh

echo "Running Experiment 3/3: Topological Dictatorship..."
./pipeline_topology.sh

echo "======================================================"
echo "🏆 ALL EXPERIMENTS COMPLETED SUCCESSFULLY! TIME TO PUBLISH!"
echo "======================================================"
