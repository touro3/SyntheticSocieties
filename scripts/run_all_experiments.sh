#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# ==============================================================================
# FULL PAPER REPRODUCTION PIPELINE
#
# Runs all BGF experiments and generates all figures needed for the paper.
# Phases execute sequentially — each depends on artifacts from previous phases.
#
# Prerequisites:
#   - venv activated with all dependencies
#   - Mistral-7B model available (for LLM phases)
#   - artifacts/persona_fidelity/ populated (Phase A output)
#
# Usage:
#   bash scripts/run_all_experiments.sh           # Run everything
#   bash scripts/run_all_experiments.sh --no-llm  # Skip LLM phases (baselines only)
# ==============================================================================

# ---                                                                           
# The Simplest Starting Point                                                   
#                                                                               
# # 1. Activate the virtual environment (always do this first)                  
# source venv/bin/activate                                                      
#                                                                               
# # 2. Verify everything works — should show 503 passed                         
# make test                                                                     
#                                                                               
# ---                                                                           
# Running Simulations (No GPU Required)                                         
#                                                                               
# # Fastest run: 5 rounds, 10 agents, 3 seeds — takes ~30 seconds               
# make reproduce-fast                                                           
#                                                                               
# # Standard run: 30 rounds, 100 agents — takes ~5 minutes                      
# make reproduce-full                                                           
#                                                                               
# # Or call the script directly with custom settings                            
# python scripts/run_full_pipeline.py --rounds 10 --agents 20 --seeds 42,123,7  
#                                                                               
# Results land in experiments/ and plots in analysis/figures/.                  
#                                                                               
# ---                                                                           
# Running the Full Paper Reproduction                                           
#                                                                               
# # One command: checks Python, runs tests, runs pipeline, generates plots
# bash reproduce_paper.sh                                                       
#  
# # Full version (slower, matches paper parameters)                             
# bash reproduce_paper.sh --full                                                
#                                                                               
# ---                                                           
# Running Individual Analyses
#                                                                               
# # Phase 17 — Trust gradient validation (5 minutes, no GPU)
# # Shows that high-trust ESS populations → higher cooperation rates            
# python scripts/run_trust_gradient.py                                          
# python scripts/plot_trust_gradient.py                                         
# # → outputs: analysis/tables/trust_gradient.json                              
# #            analysis/figures/trust_gradient.png          
#                                                                               
# # Phase 18 — Phase transition analysis                    
# # First need experiments to exist, then:                                      
# python scripts/run_phase_transition_sweeps.py --analyze-only                  
# python scripts/plot_phase_transitions.py                                      
# # → outputs: analysis/tables/phase_transitions.json                           
# #            analysis/figures/phase_transitions.png                           
#                                                     
# # Regenerate all plots from existing experiments                              
# make plots
# # or: python scripts/run_full_pipeline.py --plots-only                        
#                                                                               
# ---
# Running With the LLM (Requires GPU)                                           
#                                                                               
# # Full LLM pipeline — Condition A vs Condition B comparison
# python scripts/run_full_pipeline.py --include-llm --rounds 30 --agents 50     
# --seeds 42                                                                    
#                                                                               
# # With ablation ladder (V0–V4 prompt levels)                                  
# python scripts/run_full_pipeline.py --include-llm --run-ablation-ladder
#                                                                               
# # Bad apple experiment (5% adversarial agents)                                
# bash pipeline_bad_apple.sh
#                                                                               
# # Macro shock experiment (50% wealth crash at round 15)                       
# bash pipeline_macro_shock.sh                                                  
#                                                                               
# # Topology comparison (small-world vs random)                                 
# bash pipeline_topology.sh
#                                                                               
# # Full paper reproduction including LLM (~hours)                              
# bash run_all_experiments.sh
#                                                                               
# ---                                                           
# Where Everything Lives
#                                                                               
# experiments/                  ← Each run gets a folder here
#   rule_s42_r10/                                                               
#     rounds.jsonl              ← Every agent decision, every round             
#     metrics.json              ← Final Gini, coop rates, etc.                  
#     config.yaml               ← Exact config used                             
#   llm_s42_r30/                                                                
#     prompts.jsonl             ← Every LLM prompt (for debugging)              
#                                                                               
# analysis/                                                                     
#   figures/                    ← All PNG plots                                 
#     phase_c_macro_comparison.png                          
#     trust_gradient.png                                                        
#     phase_transitions.png
#     grafo_A_ablated.png       ← Network graph, Condition A                    
#     grafo_B_grounded.png      ← Network graph, Condition B                    
#   tables/                                                                     
#     trust_gradient.json       ← Trust gradient raw results                    
#     phase_transitions.json    ← Phase transition sweep results                
#                                                     
# data/                                                                         
#   ess_clean.parquet           ← 866 Austrian ESS respondents
#   empirical_distributions.json                                                
#                                                                               
# ---                                                                           
# Using the New Metrics Directly in Python                                      
#                                                                               
# from metrics.behavioral_realism import compute_brm_jsd,
# compute_rlhf_bias_index                                                       
# from metrics.trust_gradient import compute_trust_recovery_correlation,
# TRUST_GROUPS                                                                  
# from metrics.persona_decay import compute_per_round_persona_fidelity
# from metrics.complexity import fit_phase_transition, fit_power_law            
# from metrics.mediation import compute_mediation_decomposition                 
#                                                                               
# # BRM: how close is simulation to ESS wealth distribution?                    
# brm = compute_brm_jsd(sim_wealth_array, emp_wealth_array) 
# print(f"BRM = {brm:.3f}")  # 1.0 = perfect match                              
#                                                                               
# # RLHF Bias: how far from uniform action distribution?                        
# bias = compute_rlhf_bias_index({"work": 0.1, "save": 0.1, "cooperate": 0.8})  
# print(f"B_RLHF = {bias:.3f}")  # 0.0 = unbiased, 0.667 = fully biased         
#                                                                               
# # Phase transition detection on sweep data                                    
# result = fit_phase_transition(sweep_x, metric_y)                              
# print(f"Inflection at {result['inflection_point']:.2f},                       
# R²={result['r_squared']:.2f}")                                                
#                                                                               
# ---                                                                           
# ---
# Quick Sanity Checks                                                           
#                                                     
# # Verify all imports work
# python -c "                                                                   
# from metrics.behavioral_realism import compute_brm_jsd, 
# compute_rlhf_bias_index                                                       
# from metrics.trust_gradient import TRUST_GROUPS             
# from metrics.complexity import sigmoid, fit_phase_transition
# from metrics.persona_decay import compute_per_round_persona_fidelity          
# from metrics.mediation import compute_mediation_decomposition
# from decision.padded_prompt_builder import build_padded_prompt                
# print('All imports OK')                                                       
# print(f'Trust groups: {[g.name for g in TRUST_GROUPS]}')                      
# "                                                                             
#                                                                               
# # See what experiments exist
# python -c "                                                                   
# from tracker.analytics import ExperimentAnalytics           
# ea = ExperimentAnalytics()                                                    
# print(ea.list_experiments())
# "                                                                             
#                                                     
# ---
# The Full Progression (What Order to Run Things)
#                                                                               
# 1. make test                    ← Always start here
# 2. make reproduce-fast          ← Quick baseline smoke test                   
# 3. python scripts/run_trust_gradient.py   ← Phase 17, GPU-free
# 4. bash pipeline_bad_apple.sh   ← Phase stress test (no LLM)                  
# 5. python scripts/run_full_pipeline.py --include-llm  ← Full paper (GPU)      
# 6. make plots                   ← Regenerate all figures                      
# 7. bash reproduce_paper.sh --full  ← Final reproducibility check


set -e
set -o pipefail

source venv/bin/activate
export PYTHONPATH="."

NO_LLM=false
if [ "$1" = "--no-llm" ]; then
    NO_LLM=true
fi

DIVIDER="======================================================"

echo "$DIVIDER"
echo "  BGF FULL PAPER REPRODUCTION"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "$DIVIDER"

# ── Phase C: Baselines & Framework Validation ────────────────────────────

echo ""
echo "[Phase C] Baselines & Validation..."
echo "$DIVIDER"
bash scripts/pipeline_phase_c.sh
echo ""

# ── Phase D: Large-Scale Simulation (500 agents) ────────────────────────

if [ "$NO_LLM" = false ]; then
    echo "[Phase D] Large-Scale Simulation (500 agents, 30 rounds)..."
    echo "$DIVIDER"
    bash scripts/pipeline_phase_d.sh
    echo ""
fi

# ── Stress Test: Bad Apple Injection ─────────────────────────────────────

if [ "$NO_LLM" = false ]; then
    echo "[Stress Test] Bad Apple Injection (5% adversarial agents)..."
    echo "$DIVIDER"
    bash scripts/pipeline_bad_apple.sh
    echo ""
fi

# ── Stress Test: Macroeconomic Shock ─────────────────────────────────────

if [ "$NO_LLM" = false ]; then
    echo "[Stress Test] Macroeconomic Shock (50% wealth shock at round 15)..."
    echo "$DIVIDER"
    bash scripts/pipeline_macro_shock.sh
    echo ""
fi

# ── Stress Test: Network Topology Comparison ─────────────────────────────

if [ "$NO_LLM" = false ]; then
    echo "[Stress Test] Topology Comparison (small-world vs fully-connected)..."
    echo "$DIVIDER"
    bash scripts/pipeline_topology.sh
    echo ""
fi

# ── Multi-Seed Trajectory Plots ──────────────────────────────────────────

echo "[Analysis] Generating multi-seed trajectory plots..."
python scripts/plot_trajectories_full.py --seeds 42,123,7 2>/dev/null || echo "  (skipped — no experiment data found)"
echo ""

# ── Network Evolution Visualization ──────────────────────────────────────

echo "[Analysis] Generating network evolution plots..."
# Try the most common LLM experiment directory
LLM_EXP=$(ls -d experiments/cmp_llm_s42 2>/dev/null || true)
if [ -n "$LLM_EXP" ]; then
    python scripts/plot_network_evolution.py --experiment "$LLM_EXP" 2>/dev/null || echo "  (skipped)"
else
    echo "  (skipped — no cmp_llm_s42 experiment found)"
fi
echo ""

# ── ESS Data Validation ─────────────────────────────────────────────────

echo "[Validation] Running ESS data integrity checks..."
python scripts/validate_ess_data.py || echo "  (validation reported issues — check output above)"
echo ""

# ── DuckDB Analytics ─────────────────────────────────────────────────────

echo "[Analytics] Running DuckDB experiment queries..."
python -c "from tracker.analytics import run_all_queries; run_all_queries()" 2>/dev/null || echo "  (skipped — experiment_index.parquet not found)"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────

echo "$DIVIDER"
echo "  ALL EXPERIMENTS AND ANALYSES COMPLETED"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Outputs:"
echo "    Experiment data:  experiments/"
echo "    Figures:          analysis/figures/"
echo "    Analytics tables: analysis/tables/"
echo "    Network exports:  analysis/networks/"
echo "$DIVIDER"
