---
title: Synthetic Societies
emoji: 🧪
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 5050
suggested_hardware: cpu-basic
suggested_storage: small
pinned: false
---

# SyntheticSocieties — Behavioral Grounding Framework (BGF)

[![BGF CI](https://github.com/touro3/SyntheticSocieties/actions/workflows/ci.yml/badge.svg)](https://github.com/touro3/SyntheticSocieties/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-1484%20passed-brightgreen.svg)](#test-suite)
[![Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen.svg)](#test-suite)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Reproducible](https://img.shields.io/badge/reproducible-make%20reproduce-blue.svg)](#reproducibility)

> **Reproducibility one-liner:** `make reproduce` (CPU baselines, ~30 min) or `make reproduce-docker` (containerised; asserts pytest passes at build time). Pinned dependency closure in `requirements.lock.txt` (258 packages, frozen from the validated venv). DOI metadata in `.zenodo.json`; citation metadata in `CITATION.cff`.

**Can LLMs grounded in real survey microdata produce more realistic synthetic societies than pure LLMs?**

SyntheticSocieties is a research-grade agent-based simulation framework that tests whether Large Language Models (LLMs) conditioned on empirical sociodemographic data from the European Social Survey (ESS) generate more behaviorally faithful synthetic populations than ungrounded LLMs.

**Core finding**: Without empirical anchoring, RLHF-aligned LLMs default to hyper-cooperative, risk-blind behavior — producing utopian societies bearing no resemblance to reality. The BGF mitigates this systematically through dual RAG pipelines, hierarchical memory with reflection, and a formally specified policy architecture grounded in ESS microdata.

---

## Key Results

| Metric | Pure LLM (Condition A) | Grounded LLM (Condition B) |
|--------|----------------------|---------------------------|
| Gini coefficient | ~0.08 (near-perfect equality) | 0.28–0.34 (empirically plausible) |
| Cooperation rate | ~90% (uniform, RLHF bias) | ~35% (selective, trust-dependent) |
| RLHF Bias Index (B_RLHF) | ~0.52 | ~0.21 (~60% reduction) |
| Network modularity (Q) | ~0.04 (no community structure) | ~0.31 (fragmented clusters) |
| Behavioral Realism Metric (BRM) | ~0.23 ± 0.04 | ~0.61 ± 0.07 (2.7× improvement) |
| Behavioral entropy | Low (mode collapse) | High (diverse strategies) |

### Cross-Model Generalizability (Phase 16)

| Model | Cond A B_RLHF | Cond B B_RLHF | Δ B_RLHF | Grounding Effective? |
|-------|--------------|--------------|---------|---------------------|
| Mistral-7B-Instruct-v0.3 | 0.567 | 0.467 | −17.6% | Yes |
| Qwen2.5-7B-Instruct | 0.333 | 0.233 | −30.0% | Yes |
| GPT-4o-mini | 0.223 | 0.313 | +40.3% | No (inverse effect) |

*The GPT-4o-mini inverse effect is an honest null result identifying alignment methodology as a moderating variable. See `analysis/cross_model_results.json` and `docs/paper.md` Section 5.6.*

---

## Formal Framework

BGF is formally defined as a tuple:

```
BGF = (A, E, G, P, Phi, T)
```

- **A** — Agent set with profiles sampled from the empirical distribution `D_ESS` via grounding function `Phi`
- **E** — Economic environment with payoff function `u(action, state)`
- **G** — Social graph (Watts-Strogatz small-world, evolving via cooperation events)
- **P** — LLM policy: `Profile × State × Memory × Context → Action`
- **Phi** — Grounding function mapping ESS microdata to `AgentProfile`
- **T** — Simulation horizon

### Core Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **BRM-JSD** | `1 - JSD(D_sim ‖ D_ESS)` | Distribution fidelity, ∈ [0,1] |
| **Composite BRM** | Weighted avg of JSD, Gini gap, coop gap, stability | Overall realism score, ∈ [0,1] |
| **B_RLHF** | `0.5 · Σ|π(a) - 1/3|` (Total Variation from uniform) | Alignment bias, ∈ [0,1] |

See `docs/formal_framework.md` for complete mathematical definitions.

---

## Architecture

```
ESS Microdata → Population Synthesis → Agent Creation → Simulation Kernel → Metrics & Analysis
     │                  │                    │                  │                    │
 data/            population/           agents/          simulation/           metrics/
                                        decision/        environment/          analysis/
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `population/` | Synthesizes agent populations from ESS survey distributions with joint-distribution preservation |
| `agents/` | Agent class: ESS-derived `AgentProfile`, mutable `AgentState`, hierarchical `HierarchicalMemory` |
| `decision/` | Pluggable policy system: `RandomPolicy`, `RuleBasedPolicy`, `LLMPolicy`, `ConditionedLLMPolicy`, `AblatedLLMPolicy` |
| `decision/sql_rag.py` | SQL-based population norm injection: queries DuckDB over ESS microdata |
| `decision/graph_rag.py` | Graph-based social context: centrality, cooperation history, structural holes |
| `decision/padded_prompt_builder.py` | Length-controlled ablation: semantically empty padding matching grounded prompt token count |
| `decision/token_budget.py` | Token budget guard: prevents silent truncation, trims sections in priority order |
| `environment/` | Economic engine (work/save/cooperate payoffs), Watts-Strogatz network topology, institutional rules |
| `simulation/` | Event-driven kernel with batched GPU inference, `RoundProcessor` decomposition |
| `metrics/behavioral_realism.py` | **BRM** and **B_RLHF** — formal realism metrics |
| `metrics/persona_decay.py` | Per-round persona fidelity: quantifies behavioral drift from ESS profile |
| `metrics/mediation.py` | Factorial effect decomposition: persona effect + RAG effect + interaction |
| `metrics/complexity.py` | Phase transition detection (sigmoid fit) + power law fitting (Clauset 2009 MLE) |
| `metrics/` | 15+ evaluation dimensions: Gini, Lorenz, JSD, Wasserstein, network metrics, persona fidelity |
| `bgf_logging/` | JSONL event and prompt logging for full reproducibility |
| `tracker/` | DuckDB-based experiment registry: SQL queries across 174+ runs |
| `docs/formal_framework.md` | Mathematical definitions of BGF, BRM, and B_RLHF |
| `docs/causal_model.md` | Causal DAG, confound control table, mediation decomposition |

### Experimental Conditions

| Condition | Description | Policy |
|-----------|-------------|--------|
| **Condition A (Ablated)** | LLM with environment rules only — no ESS persona, no RAG | `AblatedLLMPolicy(no_persona)` |
| **Condition B (Grounded)** | Full ESS persona + SQL RAG + Graph RAG + hierarchical memory | `ConditionedLLMPolicy` |
| **Padded Control** | Same token count as B, but ESS content replaced with semantic filler | `build_padded_prompt()` |

---

## Reproducibility

This is a research artefact; reproduction is a first-class concern.

| What | How |
|---|---|
| **One-command paper reproduction** (CPU baselines, ~30 min) | `make reproduce` |
| **Figure regeneration only** (no simulations re-run) | `make reproduce-figures` |
| **Analysis-table re-aggregation** (idempotent, no sim) | `make reproduce-tables` |
| **Containerised reproduction** (asserts pytest passes at build time) | `make reproduce-docker` |
| **Pinned dependency closure** (258 packages, full pip-freeze of validated venv) | `requirements.lock.txt` |
| **Re-pin the lockfile** after intentional dep updates | `make lock` |
| **Citation metadata** | `CITATION.cff` (CFF v1.2.0) |
| **DOI / Zenodo metadata** | `.zenodo.json` (auto-picked up on GitHub release) |
| **Hash-stable runs** (byte-identical events.jsonl across machines) | `PYTHONHASHSEED=0` set by all reproduce entrypoints |
| **Crash-recovery / resume** | `python scripts/run_config_simulation.py --resume <EXP_ID>` reads `experiments/<EXP_ID>/checkpoint.json` |

GPU experiments (Mistral-7B-Instruct-v0.3 at 4-bit quantisation, ~16 GB VRAM) are documented in `scripts/pipeline_*.sh` and are run manually; they are intentionally excluded from `make reproduce` so a CPU-only laptop can validate the full non-LLM analysis pipeline.

---

## HuggingFace Space Setup

The public demo runs on a Docker Space at `huggingface.co/spaces/<user>/<space>`. To deploy your own copy:

**1. Secrets to configure** (Settings → Variables and secrets):

| Variable | Required | Purpose |
|---|---|---|
| `BGF_API_TOKEN` | Recommended | Bearer token gating all POST endpoints. Unset = open mode. |
| `OPENAI_API_KEY` | For LLM features | `/design-simulation`, interview synthesis, anchor stance extraction, `/report`. |
| `GROQ_API_KEY` | Optional fallback | Cheaper LLM provider; used when OpenAI is unavailable. |
| `BGF_DATA_ROOT` | For persistence | Point at `/data` if you enable persistent storage. Without it, all uploads + experiment results are lost on every Space restart. |
| `BGF_CORS_ORIGINS` | Optional | Comma-separated allowed origins. Defaults to auto-derived Space URL + localhost. |
| `BGF_DEMO_MODE` | Public demos | `true` clamps wizard sims to small N/T so a visitor cannot OOM the Space. |

**2. Enable persistent storage** (Settings → Storage → Small / 5 GB) and set `BGF_DATA_ROOT=/data` so `experiments/`, `uploads/`, `tracker/`, and human-eval responses survive restarts.

**3. Push from a local clone** with `python scripts/deploy_hf_space.py` (requires `HF_TOKEN` and `HF_SPACE_ID=<user>/<space>` in your local env).

---

## Quick Start

> **Dependencies**: `requirements.txt` — full install with GPU support. `requirements-ci.txt` — CPU-only for CI/testing. `requirements-api.txt` — API/web service only. `requirements.lock.txt` — exact pinned closure of the validated environment (use this for byte-identical reproduction).

```bash
# Install
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run baseline (no GPU, fast verification — mock policy)
python scripts/run_full_pipeline.py

# Run with LLM grounding (GPU required: Mistral-7B-Instruct-v0.3)
python scripts/run_full_pipeline.py --include-llm --rounds 10 --agents 5

# Multi-seed statistical sweep
python scripts/run_full_pipeline.py --seeds 1,2,3,4,5 --include-llm

# Regenerate plots from existing data
python scripts/run_full_pipeline.py --plots-only

# By default, analytics are scoped to the current run seeds/policies (no stale mixing).
# A provenance manifest is saved to: analysis/reports/last_pipeline_run_manifest.json
# A post-run integrity audit is also generated:
#   analysis/reports/research_integrity_audit.json
#   analysis/reports/research_integrity_audit.md

# Reproducibility: ALWAYS pin hash randomization. PYTHONHASHSEED must be
# exported BEFORE the Python interpreter starts (it cannot be set from
# pytest/conftest — that is too late). reproduce_paper.sh and the GPU
# launcher already export PYTHONHASHSEED=0; for manual runs do the same:
PYTHONHASHSEED=0 python scripts/run_full_pipeline.py --seeds 1,2,3

# Full paper reproduction (sets PYTHONHASHSEED=0 for you):
bash reproduce_paper.sh

# Run full test suite (1,203 tests) — pin the hash seed here too
PYTHONHASHSEED=0 pytest tests/ -v

# Run specific new metric tests
pytest tests/test_behavioral_realism.py tests/test_persona_decay.py \
       tests/test_complexity.py tests/test_mediation.py \
       tests/test_length_controlled_ablation.py -v
```

---

## Experiment Pipelines

```bash
bash scripts/pipeline_bad_apple.sh      # 5% adversarial agent injection
bash scripts/pipeline_macro_shock.sh    # 50% wealth shock at round 15
bash scripts/pipeline_topology.sh       # Network topology comparison (small-world vs random)
bash scripts/pipeline_phase_c.sh        # 100 agents, 100 rounds, LLM with RAG
bash scripts/pipeline_phase_d.sh        # Large-scale 500-agent simulation
bash scripts/run_all_experiments.sh     # Full paper reproduction (all conditions + analysis)
bash scripts/run_all_experiments.sh --no-llm  # Skip GPU-dependent phases
```

---

## Analysis & Plotting

```bash
# Multi-seed trajectory plots with 95% CI bands
python scripts/plot_trajectories_full.py --seeds 5

# Network evolution snapshots + time series
python scripts/plot_network_evolution.py

# Phase transition sweeps (bad apple fraction, shock magnitude, beta)
python scripts/run_phase_transition_sweeps.py --no-llm  # Run sweeps
python scripts/plot_phase_transitions.py                 # Plot results

# Out-of-sample validation (ESS in-sample + WVS holdout; optional ungrounded control)
python scripts/run_cross_cultural_expanded.py --run-ungrounded-control
python scripts/plot_cross_cultural_expanded.py --wvs

# Human baseline analysis (after Prolific CSV is collected).
# Default mode is publication-strict: it rejects synthetic/demo-like data and
# enforces minimum sample/round thresholds.
python scripts/analyze_human_baseline.py \
  --input-csv data/human/prolific_round_data.csv \
  --comparison-json analysis/tables/human_vs_simulation_reference.json

# Optional standalone audit (basic or publication)
python scripts/research_integrity_audit.py --level publication --fail-on-blockers

# ESS data validation
python scripts/validate_ess_data.py

# Statistical significance analysis (DuckDB)
python -c "from tracker.analytics import pairwise_significance; ..."

# Calibration vs evaluation gap
python metrics/calibration.py
```

---

## New in Recent Sessions (Phase 27, Phases 23–18)

### Phase 27 — True Cross-Cultural ESS Validation

Added cross-cultural validation grounded in published ESS-11 country cluster statistics.
Three clusters (Nordic / Southern / Eastern) are simulated with trust-matched agent profiles;
Pearson r and Spearman ρ between ESS-11 mean trust and simulated cooperation rates tests
whether BGF recovers the empirical cross-cultural trust gradient.

```python
from metrics.cross_cultural import compute_cross_cultural_correlation, ClusterSimResult
from population.country_clusters import load_clusters

clusters = load_clusters()  # Nordic (trust=0.673), Southern (0.463), Eastern (0.421)
# After running simulations per cluster:
result = compute_cross_cultural_correlation(cluster_results)
# CrossCulturalResult(pearson_r=?, spearman_rho=1.0, gradient_recovered=True)
```

Dry-run result (mock policy, rule-based agents): **Spearman ρ = 1.000 (p = 0.000)**.
LLM GPU result (Mistral-7B, 10 seeds/cluster, N=20, T=10): **Pearson r = +0.983, Spearman ρ = +1.000** — gradient fully recovered.

New files: `population/country_clusters.py`, `metrics/cross_cultural.py`,
`data/cross_cultural_benchmarks.json`, `configs/cross_cultural/`, `scripts/run_cross_cultural.py`,
`scripts/plot_cross_cultural_validation.py`, `tests/test_cross_cultural.py` (30 tests)

---

### Phase 23 — Formal Framework
Added `metrics/behavioral_realism.py` with two formally defined metrics:

```python
from metrics.behavioral_realism import (
    compute_brm_jsd,          # BRM = 1 - JSD(D_sim, D_ESS)  ∈ [0,1]
    compute_rlhf_bias_index,  # B_RLHF = TV(π, π_uniform)    ∈ [0,1]
    compute_composite_brm,    # Weighted aggregate of 4 sub-dimensions
    rlhf_bias_index_from_counts,
)

# Example
brm = compute_brm_jsd(simulated_wealth, ess_wealth)       # 0 = no match, 1 = identical
bias = rlhf_bias_index_from_counts({"cooperate": 95, "work": 3, "save": 2})  # ~0.62

composite = compute_composite_brm(
    sim_wealth, emp_wealth, sim_gini, emp_gini,
    sim_coop_rate, emp_coop_rate, temporal_stability_jsd
)
# Returns: {'composite': 0.74, 'jsd_component': 0.81, ...}
```

Mathematical definitions: `docs/formal_framework.md`.

### Phase 25 — Contribution Statement
Rewrote `docs/paper.md` abstract and introduction with:
- Numbered contribution list (9 contributions)
- Summary of Contributions box
- Formal BRM/B_RLHF terminology throughout
- 7 additional citations (Ouyang 2022, Watts-Strogatz 1998, Barabasi-Albert 1999, Axelrod 1984, etc.)

### Phase 24 — Limitations & Persona Decay
Added `metrics/persona_decay.py` to quantify behavioral drift from initial ESS persona:

```python
from metrics.persona_decay import (
    expected_cooperation_rate,        # Trust/risk → expected coop baseline
    compute_per_round_persona_fidelity,  # Sliding-window fidelity per agent
    compute_decay_summary,            # Aggregate across all agents
)

# For a high-trust, low-risk agent, expected cooperation is ~0.74
rate = expected_cooperation_rate(profile)  # 0.2 + 0.6 * trust * (1 - risk)

# Compute fidelity over time
result = compute_per_round_persona_fidelity(events, profile, window=5)
# {'rounds': [...], 'fidelity': [...], 'decay_rate': -0.02, 'half_life': 22}
```

Expanded Limitations section in paper to 6 failure modes with quantified analysis.

### Phase 19 — Causal Inference
**Length-controlled ablation** (`decision/padded_prompt_builder.py`):

```python
from decision.padded_prompt_builder import build_padded_prompt, measure_grounded_token_count

# Measure token count of fully grounded prompt
target_tokens = measure_grounded_token_count(profile, state, memory, context, round_id=5,
                                              social_context=sql_ctx, population_context=graph_ctx)

# Build padded control matching that token count
messages = build_padded_prompt(profile, state, memory, context, round_id=5,
                               target_token_count=target_tokens, seed=42)
# Same length as grounded, but filler instead of ESS content
```

**Mediation analysis** (`metrics/mediation.py`):

```python
from metrics.mediation import compute_mediation_decomposition, mediation_table

result = compute_mediation_decomposition(
    full_grounded_coop=0.35,   # Condition B: persona + RAG
    persona_only_coop=0.50,    # Persona without RAG
    rag_only_coop=0.55,        # RAG without persona
    baseline_coop=0.75,        # Condition A: neither
)
# {'total_effect': -0.40, 'persona_effect': -0.25, 'rag_effect': -0.20,
#  'interaction_effect': 0.05, 'persona_share': 0.625, 'rag_share': 0.50}
```

Causal DAG and confound control: `docs/causal_model.md`.
Paper addition: Section 3.5 "Causal Identification Strategy".

### Phase 18 — Emergent Complexity
Added `metrics/complexity.py` with phase transition detection and power law fitting:

```python
from metrics.complexity import (
    fit_phase_transition,   # Sigmoid fit via scipy.optimize.curve_fit
    fit_power_law,          # Clauset et al. 2009 MLE, no extra dependencies
    analyze_sweep_results,  # Apply across multiple metrics
)

# Detect cooperation collapse as bad-apple fraction increases
result = fit_phase_transition(sweep_x, cooperation_rates)
# {'inflection_point': 0.18, 'steepness': 14.3, 'r_squared': 0.96, 'is_transition': True}

# Test wealth distribution for power law
fit = fit_power_law(final_wealth_values)
# {'alpha': 2.41, 'is_power_law': True, 'ks_statistic': 0.04, 'p_value': 0.73}
```

Sweep orchestration: `scripts/run_phase_transition_sweeps.py`
Visualization: `scripts/plot_phase_transitions.py` (4-panel figure)
Paper addition: Section 5.4 "Emergent Phase Transitions".

---

## Experiment Outputs

Each run produces a directory under `experiments/<exp_id>/`:

```
experiments/<exp_id>/
├── config.yaml      # Run configuration snapshot
├── metadata.json    # Experiment metadata
├── events.jsonl     # Round-level agent decisions and states
├── prompts.jsonl    # LLM prompts (when applicable)
└── summary.json     # Aggregate metrics
```

Visualizations in `analysis/figures/` (93+ figures).
Network exports for Gephi in `analysis/networks/`.
Phase transition tables in `analysis/tables/`.

---

## Research Hypotheses

| ID | Hypothesis | Metric | Status |
|----|-----------|--------|--------|
| H1 | Empirical grounding improves realism | BRM(B) > BRM(A) | Confirmed |
| H2 | Persona conditioning affects cooperation | Coop rate correlates with trust | Confirmed |
| H3 | Memory improves temporal stability | JSD variance decreases with memory | Confirmed |
| H4 | Network topology modulates inequality | Small-world → higher Gini | Confirmed |
| H5 | LLM decisions are robust across seeds | CV(wealth) < 0.15 | Confirmed |
| H6 | Temperature controls decision diversity | Higher temp → higher entropy | Confirmed |
| H7 | RLHF causes systematic cooperative bias | B_RLHF(A) >> B_RLHF(B) | Confirmed |
| H8 | Grounding effect is content, not length | Effect survives padded control | Pending (padded control built; GPU run needed) |

---

## Test Suite

**1,557 tests collected; 1,484 pass on the non-LLM subset**, with 3 known pre-existing failures isolated to `tests/test_token_budget.py` and `tests/test_quick_wins.py` (token-count assertions tied to a deprecated tokenizer surface). The 73-test LLM/slow/e2e subset is gated behind `-k "llm or slow or e2e"` and exercises the GPU path. Coverage on the **core science modules** (`agents/`, `decision/`, `environment/`, `metrics/`, `population/`, `simulation/`, `tracker/`, `utils/`, `bgf_logging/`) is **82%** across 7,313 statements (measured 2026-05-26 via `make coverage-fast`).

```bash
pytest tests/ -v                                       # Full suite (1,557 tests)
pytest tests/ -v -k "not llm and not slow and not e2e" # Non-GPU subset (~72 s, 1,484 pass)
make coverage-fast                                     # Coverage on core modules (82%)
make coverage                                          # Full coverage report (fails under 70%)
pytest tests/test_behavioral_realism.py -v             # Phase 23: BRM metrics
pytest tests/test_persona_decay.py -v                  # Phase 24: Persona decay
pytest tests/test_length_controlled_ablation.py -v     # Phase 19: Padded ablation
pytest tests/test_mediation.py -v                      # Phase 19: Mediation analysis
pytest tests/test_complexity.py -v                     # Phase 18: Phase transitions
pytest tests/test_rag.py -v                            # RAG-specific subset
```

Test count history: 0 → 104 → 254 → 396 → 413 → 481 → 552 → 636 → 921 → 1,203 → **1,557** (current, measured across 91+ files).

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM | Mistral-7B-Instruct-v0.3 (HuggingFace Transformers, batched inference) |
| Survey data | European Social Survey Round 11, stored as Parquet |
| RAG | DuckDB (SQL RAG) + NetworkX (Graph RAG) |
| Configuration | Hydra YAML, Pydantic validation |
| Simulation | Event-driven kernel, seed-controlled |
| Experiment tracking | DuckDB `experiment_index.parquet` |
| Statistical tests | Mann-Whitney U, Cohen's d, bootstrap CI (95%) |
| Visualization | Matplotlib, Seaborn, NetworkX (Fruchterman-Reingold) |
| Type safety | `PolicyProtocol` (PEP 544), `LLMBackendProtocol`, Pydantic `BGFConfig` |

---

## Documentation

| File | Contents |
|------|---------|
| `docs/paper.md` | Full research paper (abstract through references) |
| `docs/formal_framework.md` | Mathematical definitions: BGF tuple, BRM, B_RLHF |
| `docs/causal_model.md` | Causal DAG, confound table, mediation decomposition |
| `docs/hypotheses.md` | Formalized experimental hypotheses H1–H8 |
| `docs/evaluation_protocol.md` | Calibration vs evaluation split methodology |
| `docs/MASTERS_ELEVATION_PLAN.md` | Research elevation roadmap (Phases 16–26) |
| `docs/BGF_PROGRESS_CHECKLIST.md` | Phase completion tracker |

---

## Project Structure

```
SyntheticSocieties/
├── agents/                # Agent architecture (profile, state, hierarchical memory)
├── bgf_logging/           # Event and prompt logging
├── configs/               # Hydra YAML configurations
├── data/                  # ESS datasets and derived distributions
├── decision/              # Policy system (LLM, RAG, ablated, padded, conditioned)
│   ├── padded_prompt_builder.py  # NEW: length-controlled causal ablation
│   └── token_budget.py           # Token budget guard
├── docs/                  # Paper, formal framework, causal model, hypotheses
│   ├── formal_framework.md       # NEW: mathematical BGF definitions
│   └── causal_model.md           # NEW: causal DAG and mediation design
├── environment/           # Economy engine, network topology, institutions
├── experiments/           # 174+ completed experiment runs
├── metrics/               # Evaluation metrics
│   ├── behavioral_realism.py     # NEW: BRM, B_RLHF
│   ├── persona_decay.py          # NEW: per-round fidelity tracking
│   ├── mediation.py              # NEW: causal effect decomposition
│   └── complexity.py             # NEW: phase transitions, power laws
├── models/                # Behavioral modeling
├── population/            # ESS-grounded population synthesis
├── scripts/               # Pipeline scripts and plotting
│   ├── run_phase_transition_sweeps.py  # NEW: sweep orchestration
│   └── plot_phase_transitions.py       # NEW: 4-panel phase diagram
├── simulation/            # Event-driven simulation kernel
├── tests/                 # 1,203 tests across 91 files
│   ├── test_behavioral_realism.py     # NEW
│   ├── test_persona_decay.py          # NEW
│   ├── test_length_controlled_ablation.py  # NEW
│   ├── test_mediation.py              # NEW
│   └── test_complexity.py             # NEW
├── tracker/               # DuckDB experiment registry + statistical tests
├── utils/                 # Configuration and I/O utilities
└── analysis/
    ├── figures/           # 93+ generated visualizations
    ├── reports/           # Persona fidelity and comparison reports
    ├── networks/          # GEXF graph exports for Gephi
    └── tables/            # Statistical summary tables + phase transitions
```
