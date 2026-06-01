# Behavioral Grounding Framework (BGF)

**One-line summary.** An agent-based simulation platform that tests whether large language models grounded in empirical socio-economic microdata (European Social Survey Round 11) generate more behaviourally realistic synthetic populations than off-the-shelf LLMs.

**Why it exists.** Instruction-tuned LLMs deployed as agents in multi-agent economic simulations exhibit a systematic anomaly — cooperation rates that exceed both Nash equilibrium and laboratory human behaviour, producing "synthetic utopias" that do not resemble observed human populations. BGF quantifies this as the **RLHF Cooperative Bias** (`B_RLHF`) and introduces the **Behavioural Realism Metric** (`BRM ∈ [0, 1]`) as a paired composite measure of distributional fidelity against ESS marginals.

**What it ships.**
- A formal agent-simulation specification `BGF = (A, E, G, P, Φ, T)` with a population-synthesis map `Φ: D_ESS → Profile` preserving joint distributions across 15+ ESS attributes.
- Four experimental conditions: A (pure LLM), B (grounded LLM with ESS-RAG), C (generative-agents proxy), D (deterministic rule-based ESS policy).
- 20+ evaluation modules: distributional similarity (JSD/KL/Wasserstein), network metrics, persona fidelity, trust gradient, cross-cultural validation, mediation analysis.
- Flask + Vue interactive web front-end deployed on HuggingFace Spaces (`touro3/synthetic-societies`) for live agent interviews, anchor queries, and AI-designed scenarios.
- One-command paper reproduction (`bash scripts/reproduce_paper.sh --full`).

**Headline findings (rule-based grounding).**
- Gini coefficient `0.325 ± 0.001` at N=500, T=30 — within Eurostat European empirical range (median ≈ 0.31).
- Cross-cultural cooperation gradient recovered at Spearman ρ = +1.000 across six ESS cultural clusters (Nordic > Southern > Eastern); independently replicated against WVS Wave 7 (r = +0.977) and Herrmann-Thöni-Gächter 2008 per-city PGG contributions (Spearman ρ = +0.886, p = 0.033) — a behavioural benchmark BGF never ingests.

**Repo entry points.**
- `docs/paper.md` — manuscript (1304 lines).
- `CLAUDE.md` — full architecture and command reference.
- `scripts/reproduce_paper.sh` — one-command reproduction.
- `api/app.py` — Flask API serving the Vue SPA.
- `decision/` — pluggable policy system (LLM / RAG / rule-based / generative-agents).
- `metrics/` — 20+ evaluation dimensions.

**Citation.** See `CITATION.cff`. Pre-registration: `docs/hypothesis_preregistration.md`.

**License.** MIT (see `LICENSE`).
