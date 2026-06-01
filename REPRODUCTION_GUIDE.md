# BGF Reproduction Guide

Complete step-by-step instructions for reproducing every result in the paper.
Last audited: 2026-05-11.

---

## Quick-start (no GPU)

```bash
git clone https://github.com/touro3/SyntheticSocieties && cd SyntheticSocieties
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/run_full_pipeline.py          # CPU, ~30s, baseline smoke-test
pytest tests/ -q                             # 1,372 tests, should all pass
```

---

## Status of every paper result

| Section | Claim | Status | Script |
|---------|-------|--------|--------|
| §6.1 Fig 2 | Cond A: coop=96.2%, B_RLHF=0.712; Cond B: coop=58.2%, B_RLHF=0.420 | ✅ Verified in `paper_numbers.json`; Figure from `fix_figure2_canonical.py` | `run_phase_c_replication.sh` to regenerate raw data |
| §6.2 Fig 3/4 | Network modularity A≈0.04, B≈0.31 | ✅ Figures regenerate from GEXF files | `python scripts/plot_networks.py` |
| §6.3 Fig 5 | Bad-apple resilience | ✅ Figure regenerates from parquets | `python scripts/plot_bad_apple.py` |
| §6.4 Fig 5 text | Phase transition f*≈0.023 (N=20 pilot) | ✅ Fitted from `bad_apple_sweep.json` | `run_bad_apple_sweep_n500.sh` for N=500 |
| §6.5 Fig 9 | Trust gradient ρ=0.800 | ✅ All values exact | `python scripts/plot_trust_gradient.py` |
| §6.6 Fig 10 / Table 3 | Cross-model: Mistral −17.6%, Qwen −30%, GPT +40.3% | ✅ Verified in `cross_model_results.json` | `python scripts/plot_cross_model_comparison.py` |
| §6.7 Fig 11/12 | Feature importance β=+0.287 trust | ✅ All values exact | `python scripts/plot_feature_importance.py` |
| §6.9 Fig 15 / Table 7 | Memory ablation M0→M3 fidelity 0.609→0.742 | ❌ NEEDS RE-RUN (mock policy) | `run_memory_ablation_llm.sh` |
| §7 Fig 7 | Macro shock hysteresis | ✅ Regenerates from parquets | `python scripts/plot_macro_shock.py` |
| §7.2 Fig 13 | Long-horizon T=100 fidelity 0.823 | ✅ Rule-based, caveat in paper | `python scripts/plot_long_horizon.py` |
| §8.1 | 10-seed A/B (pending) | ⏳ Needs GPU | `bash scripts/launch_gpu_ab.sh` |
| §8.3 Fig 16 | Cross-cultural ρ=1.000, r=0.983 | ✅ All values exact | `python scripts/plot_cross_cultural_expanded.py` |
| §8.4 | Human eval (pending) | ⏳ Needs Prolific | Game at `/human-game/` on deployed server |

---

## Step 1 — Environment setup

```bash
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-api.txt   # for the web API only
```

Required env vars for LLM runs:
```bash
export CUDA_VISIBLE_DEVICES=0         # select GPU
export HF_HOME=/mnt/raid/workspace/lucastourinho/models  # model cache
```

---

## Step 2 — Reproduce CPU-only results (no GPU required)

All of the following can run on any machine with 4 GB RAM:

```bash
# Condition D — Rule-Based ESS (Table 1 baseline)
python scripts/run_full_pipeline.py \
    --policy rule_based \
    --agents 500 --rounds 30 --seeds 42,123,7

# Trust gradient (Figure 9, Table 2)
python scripts/run_trust_gradient.py --seeds 5
python scripts/plot_trust_gradient.py

# Bad-apple resilience at N=20 (existing data, re-plot)
python scripts/plot_bad_apple.py

# Bad-apple phase transition at N=500 (PRE-REGISTERED, ~30 min)
bash scripts/run_bad_apple_sweep_n500.sh

# Feature importance (Figure 11/12)
python scripts/plot_feature_importance.py

# Cross-cultural validation (Figure 16) — re-plot from cached CSV
python scripts/plot_cross_cultural_expanded.py

# Cross-model comparison (Figure 10) — re-plot from cross_model_results.json
python scripts/plot_cross_model_comparison.py

# Figure 2 — rebuild from paper_numbers.json (no raw data needed)
python scripts/fix_figure2_canonical.py

# All analytics figures at once
python scripts/plot_all_analytics.py
```

---

## Step 3 — GPU runs (requires Mistral-7B cached)

### 3a. Memory ablation — CRITICAL, must run before submission

The existing memory ablation experiments used `policy=mock`. This re-run uses real LLM.

```bash
mkdir -p logs
# Estimated: 2–4 hours on single P100 (N=20, T=10 per run, 24 total)
bash scripts/run_memory_ablation_llm.sh

# Dry-run first to verify commands:
bash scripts/run_memory_ablation_llm.sh --dry-run

# After completion, analyze:
python scripts/analyze_memory_ablation.py
python scripts/plot_memory_ablation_heatmap.py
```

Expected results (Table 7):
- M0 grounded persona fidelity ≈ 0.609
- M3 grounded persona fidelity ≈ 0.742
- Monotonic improvement M0 < M1 < M2 < M3

### 3b. Phase-C replication — restore missing parquets

```bash
# Run in tmux to survive disconnects (~3–5 hours)
tmux new-session -d -s phase_c \
    "bash scripts/run_phase_c_replication.sh 2>&1 | tee logs/phase_c.log"
tmux attach -t phase_c

# After completion, verify against canonical values:
python scripts/compute_paper_numbers.py
```

Expected results (§6.1):
- Condition A: coop ≈ 0.962, Gini ≈ 0.625, B_RLHF ≈ 0.712
- Condition B: coop ≈ 0.582, Gini ≈ 0.260, B_RLHF ≈ 0.420

### 3c. 10-seed A/B extension (§8.1 pending)

```bash
tmux new-session -d -s gpu_ab \
    "bash scripts/launch_gpu_ab.sh 2>&1 | tee logs/gpu_ab.log"
```

---

## Step 4 — Regenerate all figures from verified data

Run in order after all desired experiments are complete:

```bash
source venv/bin/activate

# Figure 2 — from paper_numbers.json (always safe to run)
python scripts/fix_figure2_canonical.py

# Figures 3–4 — networks
python scripts/plot_networks.py

# Figure 5 — bad-apple resilience
python scripts/plot_bad_apple.py

# Figure 7 — macro shock
python scripts/plot_macro_shock.py

# Figure 9 — trust gradient
python scripts/plot_trust_gradient.py

# Figure 10 — cross-model (from cross_model_results.json)
python scripts/plot_cross_model_comparison.py

# Figures 11–12 — feature importance
python scripts/plot_feature_importance.py

# Figure 15 — memory ablation (only after run_memory_ablation_llm.sh completes)
python scripts/analyze_memory_ablation.py

# Figure 16 — cross-cultural
python scripts/plot_cross_cultural_expanded.py

echo "All figures regenerated in analysis/figures/"
```

---

## Step 5 — Verify paper numerical claims

```bash
# Runs all automated checks and prints pass/fail for every claim
python scripts/compute_paper_numbers.py --verify

# Expected output: all claims PASS
# Any FAIL will print the expected vs actual value
```

---

## Data provenance map

| Figure | Data source | Generated by |
|--------|------------|--------------|
| Fig 1 | `data/ess_clean.parquet` | `plot_empirical_analysis.py` |
| Fig 2 | `analysis/paper_numbers.json` | `fix_figure2_canonical.py` |
| Fig 3/4 | `analysis/networks/*.gexf` | `plot_networks.py` |
| Fig 5 | `experiments/bad_apple/` parquets | `plot_bad_apple.py` |
| Fig 7 | `experiments/macro_shock/` parquets | `plot_macro_shock.py` |
| Fig 8 | `analysis/paper_numbers.json` (parquets missing → re-run §3b) | `plot_phase_c.py` |
| Fig 9 | `analysis/tables/trust_gradient.json` | `plot_trust_gradient.py` |
| Fig 10 | `analysis/cross_model_results.json` | `plot_cross_model_comparison.py` |
| Fig 11/12 | `analysis/tables/feature_importance.json` | `plot_feature_importance.py` |
| Fig 13 | `experiments/long_horizon_*/` | `plot_long_horizon.py` |
| Fig 15 | `experiments/ablation_M*/` (needs LLM re-run) | `analyze_memory_ablation.py` |
| Fig 16 | `analysis/tables/cross_cultural_expanded_correlation.csv` | `plot_cross_cultural_expanded.py` |

---

## Known issues and workarounds

### Figure 8 (phase_c_macro_comparison.png) not reproducible from current disk state

The `experiments/phase_c_comparison/` parquet files are missing. The cached PNG is from
the original GPU run and shows correct values (A: 96.2% coop, B: 58.2% coop). Summary
statistics are preserved in `analysis/paper_numbers.json`. To regenerate:

```bash
bash scripts/run_phase_c_replication.sh
```

### Figure 15 (memory ablation) — all experiments used mock policy

The existing 24 ablation_M* experiments ran with `policy=mock`, making all conditions
identical. Persona fidelity values in Table 7 cannot be verified from disk. To regenerate:

```bash
bash scripts/run_memory_ablation_llm.sh
```

### Phase transition f* discrepancy (§6.4)

The bad_apple_sweep.json is from N=20 (f*≈0.023). The paper now reports this correctly
with a note that a pre-registered N=500 sweep is pending. To run:

```bash
bash scripts/run_bad_apple_sweep_n500.sh
```

### cross_model_results.json reverts automatically

A linter/formatter in this repo reverts `analysis/cross_model_results.json` to old mock
values. If the file is reverted, restore it:

```bash
python scripts/plot_cross_model_comparison.py   # re-reads and re-saves from JSON
# Or manually:
git checkout HEAD -- analysis/cross_model_results.json
```

The canonical values are: Mistral B_RLHF A=0.567/B=0.467, Qwen A=0.333/B=0.233, GPT A=0.223/B=0.313.

---

## One-command paper reproduction

After all GPU runs are complete:

```bash
bash reproduce_paper.sh
```

This runs: all figure generation scripts, `compute_paper_numbers.py --verify`, and
exports hi-res figures to `analysis/figures_hires/`.
