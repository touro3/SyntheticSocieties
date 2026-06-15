# Masterplan — Cultural Evolution of Cooperation in BGF

## Context

The monograph reports several claims the author **could not prove** at single-model / single-generation scope:
- **H7 cross-model** — withdrawn; only Mistral-7B has on-disk audit trail (`results.tex:198–248`).
- **Cross-cultural LLM replication** — pending; gradient confirmed rule-based only (`conclusion.tex:62`).
- **RLHF-attractor null** — LLMs stay pinned to a hyper-cooperative prior; grounding (H2) and memory (H8) do not move the action distribution. No test of whether cooperation *evolves* under selection pressure.

Vallinder & Hughes ([arXiv 2412.10270](https://arxiv.org/abs/2412.10270), AAMAS 2025) show cooperation **culturally evolves across generations** of LLM agents in an iterated Donor Game — and **diverges by base model** (Claude 3.5 Sonnet sustains cooperation; GPT-4o + Gemini 1.5 Flash collapse). This project has ~80% of the infrastructure to replicate *and extend* that result, directly resolving the three unproven claims and adding two novel fusions (grounding × evolution, cross-cultural × evolution) that go beyond the paper.

**Scope decisions (confirmed with user):** full 3-model API arm (Claude 3.5 Sonnet + GPT-4o + Gemini 1.5 Flash); **faithful** pairwise Donor Game (not PGG reuse).

### Reference protocol (Vallinder & Hughes, locked)
- 10 generations × 12 agents × 12 rounds/generation.
- Donor game: endowment 10; donation benefits recipient at **2× cost**.
- Donor sees recipient **reputation** (recent action history).
- Selection: **top 50%** by accumulated resources survive; bottom 50% replaced.
- Inheritance: survivors' **strategy text** is given to new-generation agents, who write their own strategy from it (prompt-text transmission).
- Optional costly punishment variant.
- 5 independent runs (seeds) per model.

## Existing seams to reuse (do not rebuild)
- `scripts/run_cross_cultural_expanded.py` — template for a multi-seed runner: builds agents → network → world → `kernel.run()` → extracts cooperation/gini. Clone its structure for the evolution runner. Cluster trust bands here feed Phase 4.
- `environment/social_dilemmas.py` — `SocialDilemma` ABC + `compute_brlhf()` (TV distance). Add `DonorGame` as a subclass.
- `agents/agent.py`, `agents/memory.py` — `Agent`, `HierarchicalMemory`, `generate_reflection()` (rule-based, no LLM) → distill survivor strategy for inheritance.
- `decision/llm_policy.py` (`propose_action`) — already layers RAG/social/collective context; add an `inherited_strategy` layer + donor-game decision prompt.
- `decision/model_config.py` + `get_backend()` — model-swap factory. `decision/openai_backend.py` already covers **GPT-4o**.
- `simulation/kernel.py` — checkpoint/resume pattern to mirror for long generational runs.

## Build plan

### Phase 0 — Faithful Donor Game substrate
- `environment/social_dilemmas.py`: add `DonorGame(SocialDilemma)` — actions `["donate","keep"]`, nash=`keep`, social_optimum=`donate`, benefit=2×cost, human_baseline from indirect-reciprocity lit. Enables `compute_brlhf` reuse for the donor game.
- **New** `environment/donor_game.py`: pairwise random matching per round, reputation ledger (each agent's last K actions visible to its donor), endowment/payoff bookkeeping, accumulated-resources tracking. Independent of the World kernel (donor game is pairwise, not world-state-based).
- Tests: `tests/test_donor_game.py` — payoff correctness (donate → −cost donor / +2×cost recipient), reputation window, accumulated-resource accounting.

### Phase 1 — Generational loop + strategy inheritance
- **New** `simulation/evolution_kernel.py`: `EvolutionRunner` — for `gen in range(10)`: run 12 donor rounds → rank by accumulated resources → **top 50% survive** → spawn offspring → inject inherited strategy → reset resources → next gen. Mirror `kernel.py` checkpoint/resume so a 10-gen × 3-model run can survive interruption (CUDA-wedge risk noted in repo history).
- Inheritance: distill survivor strategy via `memory.generate_reflection()`; pass survivor strategy texts into offspring prompt. Extend `decision/llm_policy.py` with an `inherited_strategy` context layer + a donor-game decision prompt template.
- Lineage logging: `parent_id`, `inherited_strategy`, `generation`, `fitness_rank`, per-round action into events JSONL (extend existing event logger schema).
- Tests: `tests/test_evolution_kernel.py` — selection keeps exactly top 50%, offspring receive survivor strategy text, lineage recorded. Smoke-run with `MockPolicy` (no API).

### Phase 2 — Cross-model arm (resolves H7) — **critical path / MVP**
- **New** `decision/anthropic_backend.py` + `decision/gemini_backend.py` implementing `BatchLLMBackendProtocol` (mirror `openai_backend.py`: LRU cache, retry/backoff, batch). GPT-4o uses existing `openai_backend.py`.
- Register all three in `decision/model_config.py` `get_backend()`.
- **New** `scripts/run_evolution.py` (clone `run_cross_cultural_expanded.py` structure): drives `EvolutionRunner` across models × seeds, writes results JSON + per-generation cooperation trajectory CSV.
- Experiment: 3 models × 5 seeds × 10 gen. **Expected**: Claude cooperation rises, GPT-4o/Gemini collapse → replicates paper, restores a defensible H7.
- Cost guard: short donor prompts (~150 tok); validate on a 2-gen × 4-agent × 1-seed dry pass per model before the full sweep.

### Phase 3 — Grounding × evolution (extends H2 / Φ–P_LLM under selection) — novel
- 2× arm: ungrounded vs ESS-grounded initial populations under the evolutionary loop (reuse `ablation_level` switch). Question: does grounding change *which* norms survive selection? Tests whether the Φ/P_LLM dissociation persists or breaks under evolutionary pressure — a stronger test than the static N=100 null.

### Phase 4 — Cross-cultural evolution (resolves pending LLM replication) — novel
- Seed 6 initial populations from the WVS/ESS cluster trust bands (`run_cross_cultural_expanded.py` cluster defs). Evolve each; test whether evolved cooperation rank matches the WVS gradient. This is the LLM-channel cross-cultural replication currently marked pending.

### Phase 5 — Analysis + monograph integration
- Metrics (`metrics/`): per-generation cooperation rate, terminal cooperation, cross-model divergence, cross-cultural Spearman vs WVS, multi-seed 95% CIs, MWU/permutation tests (reuse existing stats utilities).
- Figures: cooperation-vs-generation trajectories per model; cross-cultural evolved-gradient.
- Write-up: new results subsection in `docs/capstone-latex-template/partes/results.tex` + discussion in `discussion.tex`; update `README.md` Key Results and `docs/paper.md`. Frame as: cooperation *can* evolve (escaping the static RLHF-attractor null), and it diverges by model + by culture.

## Suggested order / dependencies
Phase 0 → Phase 1 → **Phase 2 (MVP, prove H7 first)** → Phases 3 & 4 (parallel extensions) → Phase 5. Phases 0–1 are API-free (MockPolicy) — full test coverage before any API spend.

## Verification
- `pytest tests/test_donor_game.py tests/test_evolution_kernel.py` — substrate + loop correctness.
- MockPolicy smoke run: `python scripts/run_evolution.py --dry-run` (no API, validates pipeline end-to-end).
- Per-model cost-validation micro-run (2 gen / 4 agents / 1 seed) before full sweep; inspect parsed donor decisions + lineage in events JSONL.
- Full run: assert per-generation cooperation trajectory written, cross-model divergence reproduced (Claude > GPT-4o/Gemini at final generation), multi-seed CIs non-degenerate.
- Reproducibility: pin seeds (range 42+), record model IDs + API versions in results metadata (mirror `_save_expanded` provenance block).

## Files
**New:** `environment/donor_game.py`, `simulation/evolution_kernel.py`, `decision/anthropic_backend.py`, `decision/gemini_backend.py`, `scripts/run_evolution.py`, `tests/test_donor_game.py`, `tests/test_evolution_kernel.py`, `mastersplan.md`.
**Modify:** `environment/social_dilemmas.py` (add `DonorGame`), `decision/llm_policy.py` (inherited-strategy layer + donor prompt), `decision/model_config.py` (register backends), `metrics/` (evolution metrics), `docs/capstone-latex-template/partes/results.tex` + `discussion.tex`, `README.md`, `docs/paper.md`.
