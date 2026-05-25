# 4. Split RAG into SQL (cohort) + Graph (social) backends

- Status: accepted
- Date: 2026-03-04
- Deciders: BGF core
- Tags: rag, prompt-engineering

## Context

Condition B (grounded LLM) needs to inject **two different shapes** of
empirical context into each agent prompt:

1. *Population-level cohort statistics* — "agents like you (same age
   band, gender, country) have a mean trust of 0.62 and tend to
   cooperate ~46 % of the time." This is a **demographic-stratified
   aggregate** over the ESS parquet.
2. *Local social signals* — "your three most-frequent recent partners
   are X, Y, Z; X cooperated last round, Y stole, Z worked." This is a
   **graph query** over the simulation event log, not over ESS.

A single RAG component conflating the two would either over-fetch (one
big bundle per agent per round) or under-segment (the same prompt
template handling two semantically different sources).

## Decision

Two independent retrievers under one prompt-builder:

- **SQLRAG** (`decision/sql_rag.py`) — DuckDB over `data/ess_clean.parquet`,
  per-agent cohort context cached in an LRU keyed on
  `(age_bin, gender, country, income_decile_bin)`. Returns natural-language
  narrative. Has explicit fallback codes (`no_data_file`,
  `no_peer_cols`, `no_cohort_match`, `query_error`) so silent
  degradation is detectable from the run log
  (`SQLRAG.last_status` + `validation.json`, audit A1.5).

- **GraphRAG** (`decision/graph_rag.py`) — in-process directed graph of
  agent → agent → action edges, updated by
  `simulation/round_processor.py::_update_graph_rag` after each round.
  Returns natural-language narrative of the agent's top-k recent
  partners.

Both retrievers go through `decision/prompt_builder.py` which composes
them with the persona description and recent memory under a single
token budget (`decision/token_budget.py`).

## Consequences

**Positive**

- SQLRAG and GraphRAG can be ablated independently: setting
  `policy.graph_rag = None` yields a "cohort-only" condition; setting
  `policy.sql_rag = None` yields a "social-only" condition. This is the
  basis of the §3.6 prompt-ablation ladder.
- SQLRAG caches per-cohort context — repeated agents in the same
  demographic bin share a single DuckDB query result, large
  speedup for N=500 runs.
- GraphRAG has no parquet dependency, so it works on the HuggingFace
  Space even when the ESS parquet is not deployed (audit A1.5
  surfaces this case explicitly in `validation.json`).

**Negative**

- Two narrative streams in one prompt can exceed the token budget on
  models with smaller context windows. `token_budget.py:24-28` defines
  the trim order (`social_context` first, then `population_context`)
  but this means under-budget runs silently lose the social signal
  first — flagged in §9 Limitation 20 of the paper.
- Two retrievers = two code paths to maintain; the divergence between
  the two `_emit_fallback` / `_warned_*` mechanisms is documented and
  intentional (one is data-source-aware, the other isn't).

## Alternatives rejected

- **Single combined RAG with one fetch.** Rejected because the two
  signal types have different update cadences (SQL = static across
  rounds; Graph = updated every round) and different caching policies.
- **Vector RAG (FAISS / Chroma) over ESS rows directly.** Rejected
  because cohort matching is a hard demographic constraint, not a
  similarity search — DuckDB SQL is the right tool.
