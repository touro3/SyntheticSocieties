# Master Guide: SyntheticSocieties

This document provides a comprehensive command-line reference for running simulations, generating research plots, and verifying the system.

---

## 1. Running Simulations

The primary entry point is `run_full_pipeline.py`. It handles experiment registration, data tracking, and plotting.

### Standard Baseline (Fast, No LLM)
Runs 5 rounds with 10 agents to verify the engine:

```bash
venv/bin/python3 scripts/run_full_pipeline.py
```

### LLM-Grounded Research Run (Requires GPU)
Runs a single-seed experiment with Mistral-7B, RAG, and Memory enabled:

```bash
venv/bin/python3 scripts/run_full_pipeline.py --include-llm --rounds 10 --agents 5
```

### Large-Scale Statistical Sweep
Run multiple seeds sequentially to generate confidence intervals:

```bash
venv/bin/python3 scripts/run_full_pipeline.py --seeds 1,2,3,4,5 --include-llm
```

---

## 2. Plotting & Analytics

We use a two-stage plotting system for both individual runs and aggregated trends.

### Generate Multi-Seed Trajectories
Extracts data from `events.jsonl` across all seeds and plots Mean ± 1σ:

```bash
# Validates wealth, stress, action frequencies, and social diversity
venv/bin/python3 scripts/plot_trajectories_full.py --seeds 5
```

### Regenerate All Individual Run Plots
If you have existing data but want new graphics:

```bash
venv/bin/python3 scripts/run_full_pipeline.py --plots-only
```

### Calibration Report
Verify how well the synthetic population matches the European Social Survey (ESS):

```bash
venv/bin/python3 metrics/calibration.py
```

---

## 3. Testing & Verification

### Full Suite (104+ Tests)
Verifies memory, kernel, RAG, and policy logic:

```bash
venv/bin/python3 -m pytest tests/ -v
```

### RAG-Specific Verification
Run these to see the "social context" and SQL injection protection in action:

```bash
# Manual showcase
venv/bin/python3 scripts/test_rag_features.py

# Automated RAG tests
venv/bin/python3 -m pytest tests/test_rag.py -v
```

---

## 4. How to Evaluate Results

### Behavioral Diversity (Entropy)
Check `analysis/figures/diversity_collapse.png`.

- Low Entropy: Agents have converged to a single behavior (e.g., all "save")
- High Entropy: Agents are experimenting with different strategies

### Social Stability (Gini Coefficient)
Check `analysis/figures/wealth_trajectories.png`.

- A rising Gini curve indicates wealth concentration (inequality)
- RAG Goal: Grounded agents (SQL/Graph) should ideally show more informed cooperation that buffers against extreme inequality

### LLM Prompt Inspection
View `analysis/experiments/<experiment_id>/prompts.jsonl` to inspect what context the LLM received. Look for:

- Social Network Context: Who has helped whom
- General Population Trends: ESS peer group trust levels

---

## Project Structure Reference

- `/agents`: Memory, State, and Persona logic
- `/decision`: RAG (SQL/Graph) and LLM backend
- `/metrics`: Trajectory extraction and Gini/Entropy calculation
- `/scripts`: High-level automation and plotting
- `/data`: ESS parquet datasets
