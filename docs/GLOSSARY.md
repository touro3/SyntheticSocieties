# Glossary

Key terms and abbreviations used throughout the BGF codebase and paper.

---

## Acronyms

| Term | Expansion | Definition |
|------|-----------|------------|
| **BGF** | Behavioral Grounding Framework | The full simulation system: `BGF = (A, E, G, P, Φ, T)` |
| **B_RLHF** | RLHF Bias Index | Total Variation distance between an observed LLM action distribution and a reference distribution; measures cooperative over-bias |
| **BRM** | Behavioral Realism Metric | Composite score ∈ [0,1] aggregating JSD, Gini gap, cooperation accuracy, and temporal stability |
| **ESS** | European Social Survey | Pan-European survey used as empirical grounding data (Round 11, 2022–2023) |
| **JSD** | Jensen-Shannon Divergence | Symmetric, bounded (∈ [0,1]) divergence between two distributions; `H(M) − 0.5H(P) − 0.5H(Q)` |
| **KL** | Kullback-Leibler Divergence | Asymmetric divergence; used internally in JSD |
| **KS** | Kolmogorov-Smirnov test | Two-sample non-parametric test for distributional equality; primary test for behavioral indistinguishability |
| **LLM** | Large Language Model | Autoregressive transformer model used as the agent decision policy |
| **RAG** | Retrieval-Augmented Generation | Injecting retrieved empirical context into LLM prompts at inference time |
| **RLHF** | Reinforcement Learning from Human Feedback | Fine-tuning technique that instills the cooperative bias BGF measures |
| **TV** | Total Variation distance | `TV(P,Q) = 0.5 · Σ|P(a) − Q(a)|`; the distance metric underlying B_RLHF |

---

## Experimental Conditions

| Label | Description |
|-------|-------------|
| **Condition A** | Pure LLM — no ESS grounding, no RAG, no memory |
| **Condition B** | Grounded LLM — ESS persona + SQL RAG + Graph RAG + hierarchical memory |
| **Condition C** | Generative Agents proxy — fictional persona, no ESS grounding (Park et al. 2023 comparison) |
| **Condition D** | Rule-Based ESS — deterministic policy from ESS distributions, no LLM |
| **Padded Control** | Same token count as Condition B, but ESS content replaced with semantic filler |

---

## Core Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| **Gini coefficient** | `(2·Σ(i·x_i) / (n·Σx_i)) − (n+1)/n` | [0, 1] | 0 = perfect equality; 1 = maximum inequality |
| **B_RLHF (uniform)** | `TV(π, π_uniform)` | [0, 0.5] | Upper bound on cooperative bias |
| **B_RLHF (human)** | `TV(π, π_human)` | [0, 1] | Calibrated bias vs. empirical human baseline |
| **BRM-JSD** | `1 − JSD(D_sim ‖ D_ESS)` | [0, 1] | Distribution fidelity; 1.0 = perfect match |
| **Composite BRM** | Weighted avg of sub-metrics | [0, 1] | Overall behavioral realism |
| **Cooperation rate** | `count(cooperate) / total_actions` | [0, 1] | Fraction of cooperative decisions per round |

---

## Architecture Terms

| Term | Location | Role |
|------|----------|------|
| **Grounding function Φ** | `population/ess_grounding.py` | Maps ESS microdata → `AgentProfile` |
| **Dual RAG** | `decision/sql_rag.py`, `decision/graph_rag.py` | Injects peer norms + social context at inference |
| **Hierarchical memory M0–M3** | `agents/memory.py` | Four ablation levels: none → window → archive → full reflection |
| **EconomyEngine** | `environment/economy.py` | Parses actions, enforces payoffs, handles adversarial agents |
| **Action collapse** | `simulation/kernel.py` | When >90% of agents choose the same action; logged as `UserWarning` |
| **Bad apple agent** | `agents/profile.py` | `is_adversarial=True`; constrained to `steal` by `EconomyEngine` |
| **Experiment registry** | `tracker/` | DuckDB-backed store of all 176+ experiment runs |
