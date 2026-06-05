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
[![Tests](https://img.shields.io/badge/tests-1578%20passed-brightgreen.svg)](#test-suite)
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

Results from the pre-registered 10-seed confirmatory extension (N=100, T=30, Mistral-7B-Instruct-v0.3).

| Metric | Ungrounded LLM (Condition A) | Grounded LLM (Condition B) | Rule-Based ESS (Condition D) | Eurostat reference |
|--------|------------------------------|---------------------------|-----------------------------|--------------------|
| Cooperation rate | 45.5 ± 4.4% | 46.1 ± 4.2% | 38.6% | 35–65% (lab PGG) |
| Gini coefficient | 0.715 ± 0.034 | 0.718 ± 0.034 | **0.325 ± 0.001** | ~0.31 |
| B_RLHF | ~0.195 | ~0.187 | 0.106 | — |
| BRM (composite) | 0.832 ± 0.022 | **0.848 ± 0.017** | 0.766 ± 0.001 | — |

**Central finding — Φ/P_LLM dissociation:** The rule-based ESS policy (Condition D) achieves Gini = 0.325 squarely within the Eurostat range at zero inference cost. The LLM conditions (A vs B) are statistically indistinguishable on cooperation, Gini, and wealth (MWU p > 0.85 on all three). ESS grounding reaches the LLM prompt but does not produce a measurable behavioural shift at this scale.

**Population-scale cooperation cascade (N=500, T=30 COMPLETE):** At N=500, LLM cooperation reaches 94.0%/96.0% (condA/condB) at R30 terminal (T=30 complete, 2026-06-05). B_RLHF amplifies to 0.607/0.627 — 91–94% of the 2/3 theoretical maximum — with condB peak of 0.633 at R27/R28. Gini reaches 0.9653/0.9695 at terminal, far outside the Eurostat range. Event-aggregate B_RLHF R1–R30: 0.516/0.545, a **2.6–2.8× amplification** over the N=100 baseline (0.195). condA has plateaued (6 consecutive rounds R25–R30: 93.6–94.0%); condB oscillates near attractor. The rule-based baseline at N=500 remains stable at Gini = 0.325.

**Memory ablation — H8 FALSIFIED (N=20, T=10, 24/24 cells, Mistral-7B-Instruct-v0.3, 2026-06-05):** Contrary to the pre-registered prediction of monotone M0→M3 increase, the grounded arm shows a **monotone decrease**: M0G(0.583) > M1G(0.367) = M2G(0.367) = M3G(0.367). The ungrounded arm shows an inverted-U: M0U(0.417) < M1U(0.633) = M2U(0.633) > M3U(0.450±0.000). Full hierarchical memory does not rescue either arm. B_RLHF global minimum at M3G (0.072). M3U convergence to 0.450±0.000 across all 3 seeds demonstrates RLHF attractor stabilisation.

**Bad apple phase transition (N=500, rule-based, 2026-06-05):** f*=0.041 (k=5.2, R²=0.996) at N=500 — still well below the 10–20% evolutionary game theory prediction. **Scale reversal**: at N=500 adversarial injection *decreases* Gini (0.246→0.180) rather than increasing it (N=20: 0.243→0.330). Mechanism: at large N, adversaries suppress cooperation → smaller public-goods pool → less wealth stratification.

### Cross-Cultural Validation

| Cluster | ESS Trust | Simulated Cooperation | WVS Trust |
|---------|-----------|----------------------|-----------|
| Nordic | 0.689 | 25.6% | 68% |
| Northern | 0.634 | 22.3% | 55% |
| Anglo | 0.565 | 19.3% | 43% |
| Western | 0.504 | 18.0% | 37% |
| Southern | 0.455 | 12.5% | 29% |
| Eastern | 0.418 | 11.2% | 24% |

Spearman ρ = +1.000 (exact p ≈ 0.003); WVS Wave 7 replication r = +0.977. Herrmann-Thöni-Gächter (2008) PGG benchmark: ρ = +0.886, p = 0.033 — an independent behavioural benchmark BGF never ingests.

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

The public demo runs on a Docker Space at `huggingface.co/spaces/touro3/synthetic-societies`. To deploy your own copy:

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
| H1 | Empirical grounding improves realism | BRM(B) > BRM(A) | Directional (+0.016, within seed variance); awaits N=500 |
| H2 | Grounding reduces RLHF bias within Mistral-7B | B_RLHF(B) < B_RLHF(A) | **Falsified at N=100** (MWU p = 0.91); N=500 T=30 complete: B_RLHF=0.607/0.627 (condA/B) — both cascade, condB marginally higher (exploratory, 1 seed) |
| H3 | Gini falls within Eurostat range under grounding | Gini(B) ∈ [0.28, 0.38] | Confirmed for rule-based (D); LLM scale Gini ≈ 0.72 (outside range) |
| H4 | Network modularity higher under grounding | Q(B) > Q(A) | Directional (ΔQ = +0.005, p = 0.68); pilot contrast not reproduced at N=100 |
| H5 | Cross-cultural trust gradient recovered | Spearman ρ > 0 (ESS clusters) | **Confirmed** — ρ = +1.000 (rule-based proxy; LLM-scale replication pending) |
| H6 | Topology variation produces phase transitions | R² > 0.85 on sigmoid fit | Confirmed (rule-based sweeps) |
| H7 | RLHF bias exists in instruction-tuned LLMs | B_RLHF > 0 | **Confirmed** — B_RLHF ≈ 0.195 at N=100, amplifies to 0.607/0.627 at N=500 terminal |
| H8 | Memory depth monotonically increases cooperation fidelity (M0→M3) | coop(M3)>coop(M0) under grounding | **FALSIFIED both arms** (v2 24/24 cells, 2026-06-05) — grounded arm monotone decrease M0G(0.583)>M1G=M2G=M3G(0.367); ungrounded arm inverted-U. See §8.5. |
| H9 | Simulated cooperation matches lab PGG benchmark | ρ vs Herrmann 2008 contributions | Confirmed per-test (ρ = +0.886, p = 0.033); not significant under family-wise correction |

---

## Test Suite

**1,578 tests passing** across 130 test files (full suite run 2026-06-05). Coverage on the **core science modules** (`agents/`, `decision/`, `environment/`, `metrics/`, `population/`, `simulation/`, `tracker/`, `utils/`, `bgf_logging/`) is **82%** across 7,313 statements (measured 2026-05-26 via `make coverage-fast`).

```bash
pytest tests/ -v                                       # Full suite (1,578 tests)
pytest tests/ -v -k "not llm and not slow and not e2e" # Non-GPU subset
make coverage-fast                                     # Coverage on core modules (82%)
make coverage                                          # Full coverage report (fails under 70%)
pytest tests/test_behavioral_realism.py -v             # BRM and B_RLHF metrics
pytest tests/test_persona_decay.py -v                  # Persona fidelity/decay
pytest tests/test_length_controlled_ablation.py -v     # Padded prompt ablation
pytest tests/test_mediation.py -v                      # Mediation decomposition
pytest tests/test_complexity.py -v                     # Phase transition detection
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
│   ├── padded_prompt_builder.py  # Length-controlled causal ablation
│   └── token_budget.py           # Token budget guard
├── docs/                  # Paper, formal framework, causal model, hypotheses
│   ├── formal_framework.md       # Mathematical BGF definitions
│   └── causal_model.md           # Causal DAG and mediation design
├── environment/           # Economy engine, network topology, institutions
├── metrics/               # Evaluation metrics (BRM, B_RLHF, Gini, persona decay, complexity)
├── population/            # ESS-grounded population synthesis
├── scripts/               # Pipeline scripts and plotting
├── simulation/            # Event-driven simulation kernel
├── tests/                 # 1,557 tests across 122 files
├── tracker/               # DuckDB experiment registry + statistical tests
├── utils/                 # Configuration and I/O utilities
└── analysis/
    ├── figures/           # 93+ generated visualizations
    ├── reports/           # Persona fidelity and comparison reports
    ├── networks/          # GEXF graph exports for Gephi
    └── tables/            # Statistical summary tables + phase transitions
```
