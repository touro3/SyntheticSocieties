# SyntheticSocieties — Behavioral Grounding Framework (BGF)

**Can LLMs grounded in real census data produce more realistic synthetic personas than pure LLMs?**

SyntheticSocieties is an agent-based simulation framework that tests whether Large Language Models conditioned on empirical socio-economic data (European Social Survey) generate more faithful behavioral proxies of real populations than ungrounded LLMs. The core finding: without empirical anchoring, RLHF-aligned LLMs default to hyper-cooperative, risk-blind behavior — producing utopian societies that bear no resemblance to reality.

---

## Key Results

| Metric | Pure LLM (Condition A) | Grounded LLM (Condition B) |
|--------|----------------------|---------------------------|
| Gini coefficient | ~0.0 (perfect equality) | 0.25–0.40 (empirically plausible) |
| Cooperation rate | ~95% (uniform) | 40–60% (selective, trust-dependent) |
| Network topology | Hyper-connected blob | Fragmented clusters with echo chambers |
| Behavioral entropy | Low (mode collapse) | High (diverse strategies) |

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
| `population/` | Synthesizes agent populations from ESS survey distributions |
| `agents/` | Agent class with profile, state, hierarchical memory |
| `decision/` | Pluggable policy system: random, template, rule-based, LLM, LLM+RAG |
| `environment/` | Economic engine (work/save/cooperate), network topology, institutions |
| `simulation/` | Event-driven kernel with batched GPU inference |
| `metrics/` | 15+ evaluation dimensions: Gini, JSD, entropy, calibration, network metrics |
| `bgf_logging/` | JSONL event and prompt logging for full reproducibility |
| `tracker/` | DuckDB-based experiment registry and cross-experiment analytics |

### Experimental Conditions

- **Condition A (Ablated)** — LLM agents with environment rules but no ESS persona conditioning
- **Condition B (Grounded)** — LLM agents conditioned on full ESS profiles (trust, risk tolerance, demographics)

## Quick Start

```bash
# Install
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run baseline (no GPU, fast verification)
python scripts/run_full_pipeline.py

# Run with LLM grounding (GPU required: Mistral-7B-Instruct-v0.3)
python scripts/run_full_pipeline.py --include-llm --rounds 10 --agents 5

# Multi-seed statistical sweep
python scripts/run_full_pipeline.py --seeds 1,2,3,4,5 --include-llm

# Run tests (104+ tests)
pytest tests/ -v
```

## Experiment Pipelines

```bash
bash pipeline_bad_apple.sh      # 5% adversarial agent injection
bash pipeline_macro_shock.sh    # 50% wealth shock at round 15
bash pipeline_topology.sh       # Network topology comparison
bash pipeline_phase_d.sh        # Large-scale 500-agent simulation
bash run_all_experiments.sh     # Full paper reproduction
```

## Plotting & Analysis

```bash
# Multi-seed trajectory plots (Mean ± 1σ)
python scripts/plot_trajectories_full.py --seeds 5

# Regenerate plots from existing data
python scripts/run_full_pipeline.py --plots-only

# ESS calibration validation
python metrics/calibration.py
```

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

Visualizations land in `analysis/figures/`; network exports for Gephi in `analysis/networks/`.

## Research Hypotheses

| ID | Hypothesis | Metric |
|----|-----------|--------|
| H1 | Empirical grounding improves realism | JSD(LLM, ESS) < JSD(baseline, ESS) |
| H2 | Persona conditioning affects cooperation | Cooperation rate correlates with trust |
| H3 | Memory improves temporal stability | JSD variance decreases with memory |
| H4 | Network topology modulates inequality | Small-world → higher Gini |
| H5 | LLM decisions are robust across seeds | CV(wealth) < 0.15 |
| H6 | Temperature controls decision diversity | Higher temp → higher entropy |

## Technology Stack

- **LLM**: Mistral-7B-Instruct-v0.3 (batched inference via HuggingFace Transformers)
- **Data**: European Social Survey Round 11, stored as Parquet
- **RAG**: SQL-based and graph-based retrieval for empirical context injection
- **Simulation**: Event-driven kernel, Hydra config, DuckDB experiment tracking
- **Visualization**: Matplotlib, Seaborn, NetworkX (Fruchterman-Reingold layout)

## Project Structure

```
SyntheticSocieties/
├── agents/          # Agent architecture (profile, state, memory)
├── bgf_logging/     # Event and prompt logging
├── configs/         # Hydra YAML configurations
├── data/            # ESS datasets and derived distributions
├── decision/        # Policy system (random, template, LLM, RAG)
├── docs/            # Paper draft, hypotheses, evaluation protocol
├── environment/     # Economy engine, network topology, institutions
├── experiments/     # 105 completed experiment runs
├── metrics/         # Evaluation metrics (15+ dimensions)
├── models/          # Behavioral modeling
├── population/      # ESS-grounded population synthesis
├── scripts/         # Pipeline scripts and plotting
├── simulation/      # Event-driven simulation kernel
├── tests/           # Test suite (104+ tests)
├── tracker/         # DuckDB experiment registry
├── utils/           # Configuration and I/O utilities
└── analysis/
    ├── figures/     # 69 generated visualizations
    ├── reports/     # Persona fidelity and comparison reports
    ├── networks/    # GEXF graph exports for Gephi
    └── tables/      # Statistical summary tables
```
