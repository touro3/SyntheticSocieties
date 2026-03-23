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
- Jensen–Shannon divergence
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