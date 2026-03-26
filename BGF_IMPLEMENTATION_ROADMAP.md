# BGF Implementation Roadmap

This roadmap defines the implementation order of the Behavioral Grounding Framework (BGF).
The goal is to build the system incrementally while preserving reproducibility and experimental rigor.

---

# Phase 1 — Project Infrastructure  Status: Completed

Goal: create the technical foundation of the project.

Tasks:

- repository structure
- Python environment
- requirements.txt
- Hydra configuration system
- logging setup
- experiment directory structure
- experiment tracker initialization

Deliverables:

- working repo
- base config system
- empty experiment tracker

---

# Phase 2 — Empirical Grounding Status: Completed

Goal: connect the simulation to real-world socio-economic data.

Tasks:

- dataset ingestion (ESS / OECD / WVS / World Bank)
- schema harmonization
- variable selection
- dataset versioning
- attribute distributions

Deliverables:

- cleaned dataset
- population attribute schema

---

# Phase 3 — Population Synthesis Status: Completed

Goal: generate a synthetic population grounded in empirical distributions.

Tasks:

- synthetic population generation
- demographic attribute sampling
- joint distribution consistency
- seed-controlled sampling

Deliverables:

- population generator
- reproducible synthetic population

---

# Phase 4 — Agent Architecture Status: Completed

Goal: implement the internal structure of agents.

Components:

- persona attributes
- internal state
- perception module
- decision interface
- state update logic

Deliverables:

- Agent class
- state representation
- perception system

---

# Phase 5 — Simulation Kernel Status: Completed

Goal: implement the event-driven simulation engine.

Tasks:

- simulation loop
- event scheduling
- environment updates
- interaction handling
- round management
- batched GPU inference processing

Deliverables:

- event-driven kernel
- high-scale round execution pipeline

---

# Phase 6 — LLM Decision Interface Status: Completed

Goal: integrate LLM reasoning into agent decisions.

Tasks:

- prompt builder
- persona conditioning
- memory injection
- model inference
- structured output parsing

Deliverables:

- decision interface
- structured action output

---

# Phase 7 — Logging & Reproducibility Status: Completed

Goal: guarantee full reproducibility.

Tasks:

- JSONL/Parquet event logs
- prompt logging
- model output logging
- seed management
- metadata storage
- experiment registration

Deliverables:

- reproducible experiment runs
- experiment metadata files

---

# Phase 8 — Evaluation Metrics Status: Completed

Goal: implement quantitative evaluation.

Metrics:

Inequality
- Gini coefficient
- Lorenz curves

Distribution similarity
- Jensen-Shannon divergence
- KL divergence
- Wasserstein distance

Network structure
- assortativity
- modularity
- diffusion speed
- topological graph rendering

Behavior
- cooperation rate
- defection rate
- temporal stability

Deliverables:

- metrics module
- topological visualizer
- analysis-ready outputs

---

# Phase 9 — Baselines & Ablations Status: Completed

Goal: create comparison agents and ablation conditions.

Baselines:

- random constrained agents
- rule-based utility agents
- socioeconomically grounded behavioral agents

Ablations:

- no persona
- minimal persona
- no memory
- no network
- no institutions

Deliverables:

- baseline simulation results

---

# Phase 10 — Experiments Status: Completed

Goal: run controlled experiments.

Tasks:

- seed sweeps
- prompt perturbation
- model comparison
- population size sweeps
- network topology sweeps
- horizon sweeps

Deliverables:

- experiment dataset
- robustness analysis

---

# Phase 11 — Experiment Tracker Status: Completed

Goal: monitor and analyze all simulation runs.

Components:

- experiment metadata
- metrics database
- DuckDB analytics
- experiment comparison queries

Deliverables:

- experiment_index.parquet
- tracker database

---

# Phase 12 — Paper Writing Status: Completed

Goal: transform results into a research paper.

Sections:

- Abstract
- Introduction
- Related Work
- Methodology
- Experimental Setup
- Results
- Discussion
- Limitations
- Conclusion

Deliverables:

- full research paper

---

# Phase 13 — Senior Architecture Refactor Status: Completed

Goal: elevate codebase from prototype to senior SWE/MLE quality. TDD throughout.

Rationale: the original codebase was functional but had: no formal interfaces (duck typing),
15+ magic numbers, hidden side-effect bugs, 3x duplicated Gini implementations, 5 copies of
system prompts, and zero config validation. This phase addresses all of those systematically.

Sub-phases (each with tests-first):

1. **Domain value objects & type safety**
   - `ProposedAction.action_type` → `Literal["work", "save", "cooperate"]`
   - `AgentState.clamp()` prevents negative wealth, bounds stress
   - `AgentProfile.__post_init__` validates ESS [0,1] fields
   - `WorldState` shock support fields

2. **Policy Protocol (PEP 544)**
   - `PolicyProtocol` — `runtime_checkable` structural subtyping for all 9 policies
   - Zero-migration: existing classes auto-conform

3. **Environment layer consolidation**
   - Fixed `InstitutionManager.execute()` hidden side-effect (direct target mutation)
   - Extracted canonical `GamePayoffs` frozen dataclass
   - Deduplicated `NetworkManager._relabel_graph()`
   - Removed dead `EconomyEngine` from exports

4. **Kernel decomposition**
   - Extracted `RoundProcessor` from god-class `SimulationKernel`
   - Eliminated 95% code duplication between `run_round()` and `run_round_batched()`

5. **LLM policy deduplication**
   - Created `LLMPolicyBase` with shared `_generate_with_retries()`, `_fallback_action()`, `_log_prompt()`
   - All three LLM policies (`LLM`, `Ablated`, `Conditioned`) inherit from base

6. **Prompt builder consolidation**
   - Centralized all 5 system prompts into `decision/system_prompts.py`
   - Single `get_system_prompt(mode)` registry with `KeyError` on typos

7. **Metrics deduplication**
   - Canonical Gini in `metrics/inequality.py`
   - Replaced 3 duplicate implementations in `macro_metrics.py`, `calibration.py`, `plot_policy_comparison_full.py`

8. **Configuration validation**
   - Pydantic `BGFConfig` schema for all YAML config sections
   - Removed hardcoded absolute path from `base_config.yaml`
   - Config validates at load time: unknown policy types, invalid ranges caught early

9. **Test infrastructure cleanup**
   - Removed `sys.path` hacks from all 37 test files
   - `pip install -e .` is the canonical import mechanism
   - Added `conftest.py` shared fixtures

10. **LLM backend Protocol**
    - `LLMBackendProtocol` and `BatchLLMBackendProtocol` with `runtime_checkable`
    - Catches `FastBatchedBackend` signature mismatch at import time

Deliverables:

- 254 tests passing (up from 105)
- Zero `sys.path` lines in tests
- Zero duplicated Gini/prompts/fallback logic
- All type contracts formally specified

---

# Phase 14 — AI/ML Subsystem Improvements Status: Completed

Goal: fix fundamental AI architecture gaps identified during audit.

Sub-phases:

1. **Memory reflection system** (was dead code)
   - `HierarchicalMemory.generate_reflection()` — rule-based summarization of archive
   - Action frequency analysis, partner relationship tracking, recency weighting
   - Cached with dirty-flag invalidation on `add()`
   - Auto-compression of archive every 20 evictions
   - Injected into prompts via `build_memory_block()` as `[Memory summary]` prefix

2. **GraphRAG centrality caching** (was O(VE) per agent per round)
   - `_centrality_cache` with `_cache_dirty` flag
   - Invalidated only on new cooperation edges (work/save don't dirty)
   - For 500 agents x 30 rounds: ~15,000 computations → ~number of topology changes

3. **SQL RAG connection fix**
   - `_connect()` changed from misleading `return True` to `-> None` with proper exceptions
   - Callers use `try/except` for meaningful error messages

4. **Token budget management** (was missing — silent truncation risk)
   - `decision/token_budget.py` — character-based estimator (4 chars ≈ 1 token)
   - `trim_to_budget()` drops sections in priority order: extra → social_context → population_context → halve memory
   - System prompt, persona, and state are never trimmed
   - Wired into `build_prompt()` — every LLM call is now budget-safe

Deliverables:

- 254 tests passing
- Memory reflections operational (was dead code)
- GraphRAG performance: O(1) amortized per context call
- Zero silent truncation risk

---

# Phase 15 — Testing & Statistical Rigor Status: Completed

Goal: fill remaining test gaps, add statistical inference, expand analysis tooling.

Sub-phases:

1. **Persona fidelity tests** — `tests/test_persona_fidelity.py` (22 tests)
   - PCA projection correctness, constant-column handling, composite_score edge cases
   - compute_fidelity_report structure, affine recalibration, write_report_files I/O

2. **Calibration metric tests** — `tests/test_calibration.py` (18 tests)
   - Calibration/evaluation split logic, compute_metrics with real/empty data
   - Gini on uniform/single-agent, gap calculation, report formatting

3. **Ablated prompt construction tests** — `tests/test_ablated_prompts.py` (48 tests)
   - All 6 ablation modes (no_persona, minimal_persona, rich_persona, no_memory, no_network, no_institutions)
   - Cross-cutting checks: 2-message structure, JSON instruction, round_id, state block

4. **Statistical significance in tracker** — `tracker/analytics.py` + `tests/test_statistical_significance.py` (24 tests)
   - `cohens_d()` — pooled-std effect size with zero-variance/small-sample guards
   - `mann_whitney_test()` — two-sided U-test with degenerate-input handling
   - `bootstrap_ci()` — seeded bootstrap confidence intervals
   - `pairwise_significance()` — compare each policy group against reference (LLM)

5. **Network evolution visualization** — `scripts/plot_network_evolution.py` + `tests/test_network_evolution.py` (10 tests)
   - Cumulative cooperation graph construction from events.jsonl
   - Multi-panel snapshot plots at selected rounds
   - Time-series of network metrics (edges, density, clustering, components)

6. **ESS data validation script** — `scripts/validate_ess_data.py` + `tests/test_ess_validation.py` (20 tests)
   - Parquet readability, required columns, [0,1] range checks
   - Missing data rate warnings, dataset_registry.json validation
   - empirical_distributions.json structure checks

7. **Trajectory extraction tests** — `tests/test_trajectories.py` (17 tests)
   - `extract_trajectories()` — event parsing, agent histories, malformed JSON handling
   - `aggregate_seeds()` — multi-seed pooling, round trimming, prefix resolution
   - Pool wealth/stress matrices returned for downstream Gini computation

8. **Dead code removal** — `environment/economy.py` deleted
   - `EconomyEngine` was 98 lines of dead code (imported nowhere, not in `__init__.py`)
   - Only reference was a commented-out import in `run_bad_apple.py`

9. **Per-agent trajectory plots with CI bands** — `scripts/plot_trajectories_full.py` rewritten
   - Removed `sys.path.append` hack
   - Fixed incomplete Gini trajectory (was `pass` placeholder) — now uses canonical `gini_coefficient`
   - Added `plot_per_agent_ci_bands()` — per-agent wealth with 95% CI shading across seeds
   - Extended `aggregate_seeds()` to return `pool_wealth` / `pool_stress` matrices

10. **Expanded `run_all_experiments.sh`** — full paper reproduction pipeline
    - Phase C (baselines), Phase D (500-agent LLM), Bad Apple, Macro Shock, Topology
    - Multi-seed trajectory plots, network evolution visualization
    - ESS data validation, DuckDB analytics
    - `--no-llm` flag to skip GPU-dependent phases

Deliverables:

- 413 tests passing (up from 254)
- Statistical significance functions for paper (Mann-Whitney U, Cohen's d, bootstrap CI)
- Network evolution plots (snapshots + time series)
- Per-agent CI band trajectory plots
- Gini coefficient over rounds (was broken, now functional)
- Data pipeline health checks (ESS validation script)
- One-command paper reproduction (`bash run_all_experiments.sh`)

---

# Phase 16 — Multi-Model Generalizability Study  Status: Completed (infrastructure + dry-run; GPU experiments pending)

Goal: validate that the RLHF cooperative bias is a general phenomenon across LLM families,
not a Mistral-7B artifact.

Tasks:

- `decision/model_config.py` — `ModelConfig` dataclass (model_id, backend_type, dtype,
  quantization, cache_dir, max_agents, max_rounds) + `get_backend()` factory
  Returns `LLMBackend` for HuggingFace, `OpenAIBackend` for OpenAI API.
- Named constructors: `ModelConfig.mistral_7b()`, `ModelConfig.llama3_8b()`, `ModelConfig.gpt4o_mini()`
- `decision/openai_backend.py` — OpenAI chat completions adapter conforming to `LLMBackendProtocol`
  Uses `gpt-4o-mini` by default; reads `OPENAI_API_KEY` from environment
- `metrics/cross_model.py` — `CrossModelResult`, `compute_cross_model_result()`,
  `build_comparison_table()` producing Table 2 with bias reduction percentages
- `scripts/run_cross_model_comparison.py` — runs A vs B for each model; `--dry-run` mode
  for pipeline validation without GPU; saves `analysis/cross_model_results.json`
- `scripts/plot_cross_model_comparison.py` — grouped bar chart: bias index + coop rate per model × condition
- `configs/cross_model/mistral.yaml`, `llama3.yaml`, `gpt4o_mini.yaml`
- `requirements.txt`: added `openai>=1.0.0`

Deliverables:

- `decision/model_config.py` + `decision/openai_backend.py` (26 tests)
- `metrics/cross_model.py`
- `tests/test_model_adapter.py`
- Figure: `analysis/figures/cross_model_bias_comparison.png`
- GPU experiments: run `python scripts/run_cross_model_comparison.py` with GPU available

---

# Phase 17 — Trust-Gradient Sub-Population Validation  Status: Completed

Goal: validate that BGF agents grounded in higher-trust ESS sub-populations produce higher
cooperation rates, confirming that the grounding function Φ genuinely transfers empirical
trust signals to simulated behavior.

Tasks:

- `metrics/trust_gradient.py` — TrustGroup dataclass, trust gradient metrics
- Spearman rank correlation between ESS trust mean and simulated cooperation rate
- Validates: `rank(ESS trust mean) ≈ rank(simulated cooperation rate)`, p < 0.10
- `scripts/run_trust_gradient.py` — parameterized sweep across trust bands
- `scripts/plot_trust_gradient.py` — scatter plot with regression line

Deliverables:

- `metrics/trust_gradient.py` — trust gradient validation (12 tests)
- `tests/test_trust_gradient.py`
- Figure: `analysis/figures/trust_gradient.png`

---

# Phase 18 — Emergent Complexity Analysis  Status: Completed

Goal: identify and characterize phase-transition behavior and power laws in BGF simulations,
establishing structural results beyond "we ran simulations."

Tasks:

- Cooperation phase transition: sweep bad-apple fraction 0–40%, fit logistic curve
- Wealth inequality phase transition: sweep shock magnitude 0–100%
- Network topology phase diagram: Watts-Strogatz β ∈ {0.0, ..., 1.0}
- Power law fitting for wealth distributions (MLE, KS test)
- `metrics/complexity.py` — sigmoid fitting, power law MLE (no external `powerlaw` package)
- `scripts/run_phase_transition_sweeps.py` — parameterized sweep scripts
- `scripts/plot_phase_transitions.py` — 4-panel phase transition figure

Deliverables:

- `metrics/complexity.py` — phase transition and power law fitting (15 tests)
- `tests/test_complexity.py`
- Figure: `analysis/figures/phase_transitions.png`

---

# Phase 19 — Causal Inference and Ablation Formalization  Status: Completed

Goal: formalize the causal chain from ESS grounding → behavior change using an ablation
design that controls for prompt length (addresses the "longer prompt = different behavior"
confound).

Tasks:

- `decision/padded_prompt_builder.py` — length-controlled ablation (padded no-grounding)
  Condition: same token count as grounded, padding is random noise. If effect survives,
  it is the semantic content, not the token length, that matters.
- `metrics/mediation.py` — 2×2 factorial decomposition of total grounding effect into
  persona effect + RAG effect + interaction effect
- `docs/causal_model.md` — DAG of BGF causal claims with confounders and identification
- `tests/test_mediation.py` — mediation decomposition tests
- `tests/test_length_controlled_ablation.py` — padded prompt construction tests

Deliverables:

- `docs/causal_model.md`
- `metrics/mediation.py` (10 tests)
- `decision/padded_prompt_builder.py` (12 tests)
- Section "3.5 Causal Identification Strategy" in paper

---

# Phase 20 — Publication-Quality Figure Export  Status: Completed

Goal: regenerate all key paper figures at publication quality (300 DPI PNG + vector PDF).

Tasks:

- `scripts/export_figures_hires.py` — batch export all figures to `paper/figures/`
  at 300 DPI (PNG) and vector PDF format
- Covers: trajectory plots, network evolution, phase transitions, trust gradient,
  bad apple resilience, ablation comparison
- `Makefile` targets: `make figures`, `make reproduce`, `make reproduce-fast`

Deliverables:

- `scripts/export_figures_hires.py` (8 tests)
- `Makefile` — unified build interface
- `paper/figures/` directory with publication-ready exports

---

# Phase 21 — Comparison to Generative Agents Baseline  Status: Completed

Goal: implement Condition C (fictional-persona LLM, no ESS grounding) to directly compare
BGF against the Park et al. 2023 Generative Agents approach.

Tasks:

- `decision/generative_agents_policy.py` — LLM with fictional narrative backstory,
  no RAG, no ESS-derived attributes. Eight backstory templates deterministically
  assigned per agent_id.
- Three-condition comparison: A (pure LLM), B (BGF grounded), C (fictional persona)
- Expected result: Condition C RLHF bias index closer to A than B, validating that
  empirical grounding (not just persona length) is what reduces alignment tax
- `configs/condition_c.yaml` — experiment config for Condition C

Deliverables:

- `decision/generative_agents_policy.py` (13 tests)
- `configs/condition_c.yaml`
- `tests/test_generative_agents_policy.py`

---

# Phase 22 — Reproducibility Package  Status: Completed

Goal: package BGF for external one-command paper reproduction.

Tasks:

- `reproduce_paper.sh` — annotated script reproducing every experiment (CPU-only mode)
  Maps paper table/figure → script → config → output file
- `Makefile` — `make reproduce`, `make reproduce-fast`, `make test`, `make figures`
- Documents GPU requirements, HuggingFace model IDs, ESS data download procedure

Deliverables:

- `reproduce_paper.sh`
- `Makefile`

---

# Phase 23 — Theoretical Framework Formalization  Status: Completed

Goal: introduce a formal mathematical definition section that allows reviewers to state
"BGF introduces X, which is novel because Y."

Tasks:

- `docs/formal_framework.md` — BGF defined as tuple (A, E, G, P, Φ, T) with formal
  payoff function, action space constraints, graph evolution rule
- `metrics/behavioral_realism.py` — BRM-JSD and RLHF Bias Index (B_RLHF = TV(π, π_uniform))
  Single [0,1] score where 1 = perfect calibration. Central claim: BRM(BGF-B) >> BRM(BGF-A)
- Tests validate BRM properties: monotonicity, boundary conditions, composite weighting

Deliverables:

- `docs/formal_framework.md`
- `metrics/behavioral_realism.py` (15 tests)
- `tests/test_behavioral_realism.py`

---

# Phase 24 — Limitations and Failure Mode Analysis  Status: Completed

Goal: systematically quantify BGF's failure modes, documenting them with the intellectual
honesty that top thesis committees require.

Tasks:

- `metrics/persona_decay.py` — measures behavioral drift from initial persona
  `persona_fidelity(round) = 1 - |actual_coop_rate - expected_coop_rate(profile)`
  Decay rate = linear slope of fidelity over rounds (negative = drifting)
- Quantifies "character capture": agents drift from ESS persona after ~20 rounds
- Tests: edge cases (single round, all-work agents, monotonic fidelity decay)

Deliverables:

- `metrics/persona_decay.py` (12 tests)
- `tests/test_persona_decay.py`
- Updated Limitations section in paper addressing 6 documented failure modes
