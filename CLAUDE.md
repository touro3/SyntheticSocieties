# CLAUDE.md

## Project Overview

**SyntheticSocieties** is an agent-based economic simulation framework (called BGF — Behavioral Grounding Framework) that tests whether LLMs grounded in empirical socio-economic data (for example from the European Social Survey, ESS) produce more realistic behavior than naive LLMs. Agents make economic decisions (work, save, cooperate, steal) in a game-theoretic setting while evolving across simulation rounds.

## Common Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Run baseline pipeline (no GPU needed: 5 rounds, 10 agents, no LLM)
python scripts/run_full_pipeline.py

# Run with LLM (GPU required: Mistral-7B-Instruct-v0.3)
python scripts/run_full_pipeline.py --include-llm --rounds 10 --agents 5

# Multi-seed statistical sweep
python scripts/run_full_pipeline.py --seeds 1,2,3,4,5 --include-llm

# Regenerate plots from existing experiments
python scripts/run_full_pipeline.py --plots-only

# Run tests
pytest tests/ -v
pytest tests/test_rag.py -v          # RAG-specific subset
python scripts/test_rag_features.py  # Interactive RAG showcase

# ESS calibration validation
python metrics/calibration.py

# Multi-seed aggregation plots (confidence intervals)
python scripts/plot_trajectories_full.py --seeds 5

# Experiment-specific pipelines (run from repo root)
bash pipeline_bad_apple.sh     # 5% adversarial agent injection
bash pipeline_macro_shock.sh   # Economic crisis (50% wealth shock at round 15)
bash pipeline_topology.sh      # Network topology experiments
bash pipeline_phase_c.sh       # 100 agents, 100 rounds, LLM with RAG
bash pipeline_phase_d.sh       # Large-scale 500-agent simulation
bash run_all_experiments.sh    # Full paper reproduction
```

## Architecture

The framework is layered; understanding the data flow is key:

```
ESS Data → Population synthesis → Agent creation → Simulation kernel → Metrics
```

### Core Layers

**`population/`** — Synthesizes realistic agent populations from ESS survey data. `ess_grounding.py` samples empirical distributions; `generator.py` creates agent profiles; `persona_synthesizer.py` generates natural-language persona descriptions.

**`agents/`** — Agent data structures. `agent.py` is the core class; `profile.py` holds demographic/psychological attributes; `state.py` tracks mutable state (wealth, stress); `memory.py` is a hierarchical memory system with a sliding window.

**`decision/`** — Pluggable policy system. All policies implement the same interface:
- `random_policy.py`, `template_policy.py`, `rule_based_policy.py` — Non-LLM baselines
- `llm_policy.py` — LLM-based decisions via `llm_backend.py` / `fast_batched_backend.py`
- `sql_rag.py`, `graph_rag.py` — RAG backends that inject empirical context into prompts
- `prompt_builder.py` — Constructs prompts from agent state, memory, world context
- `output_parser.py` — Parses LLM JSON responses with regex fallback (anti-hallucination)

**`environment/`** — The economic world:
- `economy.py` — `EconomyEngine`: parses agent actions, enforces game-theory rules, handles adversarial agents. Actions: `work` (+8 wealth), `save` (+4), `cooperate` (-3, generates +12/cooperator shared), steal (bad apples take 50% of public pool).
- `network.py` — `NetworkManager`: small-world (Watts-Strogatz) and random (Erdős–Rényi) topologies
- `institutions.py` — Institutional redistribution rules
- `world.py` / `world_state.py` — Global state and agent context assembly

**`simulation/kernel.py`** — Event-driven main loop with batched GPU inference. Coordinates policy calls, economy processing, state updates, and metric collection per round.

**`metrics/`** — 20+ evaluation dimensions: `inequality.py` (Gini coefficient), `calibration.py` (ESS comparison), `distribution.py`, `network_metrics.py`, `persona_fidelity.py`, `trajectories.py`, `macro_metrics.py`, `behavioral_realism.py` (BRM + B_RLHF), `persona_decay.py` (per-round fidelity), `mediation.py` (2×2 factorial decomposition), `complexity.py` (phase transitions + power laws), `trust_gradient.py` (Spearman rank validation), `cross_model.py` (multi-family comparison).

**`tracker/`** — DuckDB-based experiment registry (`experiment_index.parquet`). `analytics.py` enables SQL queries across all experiments.

### Configuration

Hydra-based configs in `configs/`. The default is `configs/base_config.yaml`:
- `policy.type`: `mock | random | template | rule_based | llm | generative_agents`
- `population.source`: `empirical` (ESS) or `synthetic`
- `llm.model_id`: defaults to `mistralai/Mistral-7B-Instruct-v0.3`, cached at `/mnt/raid/workspace/lucastourinho/models`
- `llm.backend_type`: `huggingface` (local GPU) or `openai` (API)
- `network.type`: `random | small_world`
- Cross-model configs: `configs/cross_model/mistral.yaml`, `qwen2.5-7b.yaml`, `gpt4o_mini.yaml`
- Condition C config: `configs/condition_c.yaml` (Generative Agents proxy)

### Experiment Outputs

Each run produces an experiment directory under `experiments/<exp_id>/` containing:
- `rounds.jsonl` — Round-level agent decisions and states
- `prompts.jsonl` — LLM prompts (for debugging/analysis)
- `metrics.json` — Aggregate metrics
- `config.yaml` — Run configuration snapshot

Plots land in `analysis/figures/`; network exports for Gephi in `analysis/networks/`.


## Key Design Patterns

- **Condition A vs B**: "Pure LLM" (no grounding) vs "Grounded LLM" (RAG-injected ESS distributions) is the central experimental contrast throughout the codebase.
- **Bad apple agents**: Set via `is_adversarial=True` on the agent profile. `EconomyEngine.parse_action()` hard-constrains them to steal-only behavior.
- **Batched inference**: `fast_batched_backend.py` handles GPU batching for large populations; `llm_backend.py` is the single-agent fallback.
- **Anti-hallucination**: `output_parser.py` uses strict JSON parsing with regex fallback; `EconomyEngine` maps invalid LLM responses to `work` by default.
- **Memory**: `memory.py` implements a hierarchical memory system with a sliding window and a "long-term" summary slot, allowing agents to recall past events and decisions.   
- **RAG**: `sql_rag.py` and `graph_rag.py` provide SQL and graph-based retrieval for empirical context injection, enabling agents to access real-world socio-economic data.
- **Experiment Tracking**: `tracker/` uses DuckDB to store experiment metadata and results, enabling SQL-based analytics across all runs.
- **Progress Tracking**: `BGF_PROGRESS_CHECKLIST.md` tracks the project's progress against the original research plan, including advanced stress tests for robustness.
- **Testing**: `tests/` contains 552+ unit and integration tests, including RAG-specific tests, cross-model adapter tests, metric validation tests, and shared pytest fixtures in `conftest.py`.
- **Configuration**: `configs/` uses Hydra for flexible configuration management, with `base_config.yaml` as the default and multiple experiment-specific configs.
- **Documentation**: `docs/` contains the paper draft, hypotheses, evaluation protocol, and other research artifacts.
- **Network Analysis**: `analysis/networks/` contains network exports for Gephi and other network analysis tools.
- **Data**: `data/` contains ESS datasets and derived distributions, stored as Parquet for efficient access.
- **Models**: `models/` contains model files, including the Mistral-7B-Instruct-v0.3 model cached at `/mnt/raid/workspace/lucastourinho/models`.
- **Logging**: `bgf_logging/` contains event and prompt logging utilities, enabling full reproducibility of simulation runs.
- **Experiments**: `experiments/` contains 105 completed experiment runs, each with its own configuration, metadata, events, prompts, and metrics.
- **Scripts**: `scripts/` contains pipeline scripts and plotting utilities, including multi-seed trajectory plots and ESS calibration validation.
- **Utilities**: `utils/` contains configuration and I/O utilities, including `config.py` for Hydra configuration and `io_utils.py` for data I/O.
- **Analysis**: `analysis/` contains analysis scripts and visualizations, including multi-seed trajectory plots and ESS calibration validation.

## Key engineering decisions

- **Condition C**: `decision/generative_agents_policy.py` implements a fictional-persona LLM policy (Park et al. 2023 proxy) with no ESS grounding or RAG — enables direct Generative Agents comparison.
- **Cross-model backends**: `decision/model_config.py` (ModelConfig dataclass + `get_backend()` factory) and `decision/openai_backend.py` (OpenAI responses.create API with caching and retry).
- **TDD**: Test-Driven Development throughout — 552+ tests in `tests/` with pytest fixtures in `conftest.py`.
- **Modular Design**: Clear separation between layers. Individual components (policies, metrics, RAG backends) can be swapped without affecting the rest of the system.

