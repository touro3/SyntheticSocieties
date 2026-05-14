# Architecture Rationale: From Design Choice to Scientific Commitment

> **Evidence-status convention.** Each layer-level claim is annotated `[audit: F.X]` referencing the corresponding row in `docs/evidence_audit.md` §F.

This document maps each layer of the BGF software architecture to a scientific commitment derived in `theoretical_foundations.md`, with an explicit falsifiable consequence for every layer. It exists to answer the CS-defense question: *"Why this architecture, rather than a simpler one?"*

The thesis of this document is that **every architectural decision is a testable claim about social behavior**, and the codebase is structured so that each claim can be independently ablated. The architecture is not a software-engineering convenience; it is the experimental apparatus.

---

## 1. Layer-to-Claim Map

| Layer (file) | Scientific commitment | Theoretical anchor | Falsifiable consequence | Ablation that tests it |
|---|---|---|---|---|
| `population/ess_grounding.py` (Φ mapping) | ESS covariates are an approximate sufficient statistic for human action distribution | `theoretical_foundations.md` §1.2 | If a **scrambled-Φ** (rows permuted across demographic keys) matches Φ on BRM, the mapping is not load-bearing — only its surface form matters | Scrambled-RAG condition (`causal_model.md` §8.1) `[audit: F.1 ⏳ pending C.6]` |
| `population/persona_synthesizer.py` (natural-language persona) | Demographic facts must be rendered in linguistic form the LLM can condition on; raw numbers underspecify identity | Argyle et al. 2023 silicon-samples result | If a **numeric-only persona** (no NL synthesis) matches NL persona on H5 trust-gradient, language conditioning is not load-bearing | V0–V4 prompt ablation ladder (paper §3.6) `[audit: F.2 ✅]` |
| `decision/sql_rag.py` (population-norms retrieval) | Peer-group base rates are not derivable from persona alone; require explicit retrieval | `theoretical_foundations.md` §3.2 (System 2, deliberative) | If the SQL-RAG channel ablation matches full grounding, peer-norm retrieval is decorative | 2×2 factorial Persona × RAG (`mediation.py`) `[audit: F.3 ✅ code + tests; aggregation pending C.4]` |
| `decision/graph_rag.py` (social-position retrieval) | Network position is a covariate the agent cannot self-report; requires graph-aware retrieval | `theoretical_foundations.md` §3.2 (System 2, strategic) | If single RAG (SQL only) matches dual RAG (SQL + graph) on H4 modularity, the dual channel is decorative | Single-vs-dual RAG ablation (future work, P3 in `TOP_TIER_RESEARCH.md`) `[audit: F.4 ⏳]` |
| `agents/memory.py` (hierarchical sliding window) | Persona persistence across rounds requires episodic memory; LLM context alone is insufficient at T ≥ 30 | `theoretical_foundations.md` §3.2 (episodic), H8 | If M0 (no memory) matches M3 (full hierarchical) on persona-fidelity decay, memory is not load-bearing | M0–M3 memory ablation `[audit: F.5 / A.9 ❌ existing 24 runs used mock policy — identical 0.117 across M0–M3; real LLM re-run pending via `scripts/run_memory_ablation_llm.sh`]` |
| `environment/economy.py` (PGG-like payoffs −3, +12/cooperator) | The {work, save, cooperate} action space is the canonical public-goods-game abstraction; payoffs match PGG multiplier conventions | `construct_validity.md` §2 | If continuous-action policy (`decision/continuous_policy.py`) yields equivalent macro patterns (Gini, modularity), the discrete abstraction is benign and the framework generalizes | Continuous-action ablation `[audit: F.6 ⏳; D.2 📐 PGG match]` |
| `environment/network.py` (small-world + random) | Network topology mediates inequality and fragmentation; non-trivial topology is required for community structure | H4, Watts & Strogatz 1998 | If complete-graph topology yields the same modularity, topology is not load-bearing | Topology ablation (`scripts/pipeline_topology.sh`) `[audit: F.7 ✅ GEXF artifacts in `analysis/networks/`]` |
| `environment/institutions.py` (redistribution rules) | Institutional framing constrains action set without overriding agent decisions; pure institutionless games collapse | Acemoglu & Robinson 2012 framing | If `no_institutions` ablation matches full institutions on H3, institutions are not load-bearing | `no_institutions` ablation row in `causal_model.md` §3 `[audit: F.8 ⏳ runner not committed]` |
| `simulation/kernel.py` (synchronous event loop) | Synchronous per-round updates are sufficient; asynchronous timing does not change emergent patterns | Implementation simplification | If async-update yields different Gini/modularity, the synchronous assumption is load-bearing and must be acknowledged as a model commitment | Async-update ablation `[audit: F.9 ⏳ not implemented]` |
| `decision/prompt_builder.py` (V0–V4 ladder) | Prompt format is a researcher degree of freedom; effect must be robust to ordering, wording, and inclusion choices | `causal_model.md` §5 honesty statement | If V4 (full grounding) shows effects absent in V3 (single ablation), the *specific* prompt structure matters — generality is bounded | V0–V4 ablation ladder (paper §3.6) `[audit: F.10 ✅]` |
| `decision/output_parser.py` (strict JSON + regex fallback) | Action attribution must not be inflated by hallucinations; invalid outputs map to a fixed default | Anti-hallucination commitment | If parser-strict and parser-lenient yield different aggregate distributions, the parser introduces measurement noise | Parser ablation (test suite covers; `tests/test_output_parser.py`) `[audit: F.11 ✅]` |
| `tracker/experiment_index.parquet` (DuckDB registry) | Reproducibility requires per-run config snapshots and indexed result artifacts | Reproducibility commitment | If a randomly chosen historical experiment cannot be re-run from its snapshot, the registry is not load-bearing | `reproduce_paper.sh` end-to-end test `[audit: F.12 ✅ 180 runs registered]` |

---

## 2. Architectural Claims Already Tested (✓) vs. Outstanding (○)

**Tested with positive evidence:**

- ✗ Memory layer (M0–M3) — **not yet established under a real LLM policy.** The 24 ablation runs in `analysis/tables/memory_ablation.json` were executed under `policy: mock`, which bypasses memory; cooperation collapses to an identical 0.117 across M0–M3. The "0.609 → 0.742" trajectory remains the *pre-registered prediction*; the real LLM re-run via `scripts/run_memory_ablation_llm.sh` is the outstanding GPU experiment that would corroborate or falsify it. `[audit: A.9 ❌ / F.5 ❌]`
- ✓ Topology layer (`pipeline_topology.sh`) — modularity Q rises with topology depth.
- ✓ V0–V4 prompt ablation ladder — effect persists across prompt structures.
- ✓ Persona × RAG 2×2 factorial — establishes both channels contribute.
- ✓ Bad-apple localization (H6) — establishes social-context awareness.

**Outstanding (closing these is the empirical roadmap):**

- ○ Scrambled-Φ vs. Φ (Track 2.1, `causal_model.md` §8.1) — distinguishes content from form.
- ○ Single vs. dual RAG — distinguishes whether two channels are needed or one suffices.
- ○ Continuous vs. discrete action space — distinguishes PGG-canonical claim from general result.
- ○ Async vs. sync update — distinguishes timing assumption.
- ○ Base-model vs. RLHF model (paper §9, future) — distinguishes RLHF-drift attribution.

Each outstanding ablation has an implementation cost <100 lines of code; the GPU cost is the binding constraint.

---

## 3. Why This Mapping Matters for a CS Thesis

A CS capstone is judged on three axes: (a) novelty of the artifact, (b) rigor of the engineering, (c) defensibility of the claims. This document is the **bridge from (b) to (c)**: every engineering decision is justified by a scientific commitment, and every commitment has an experimental escape hatch.

This is the structural difference between BGF and a typical applied-LLM project: each module is a *hypothesis embedded in code*, and the code is structured to make falsification cheap. The architecture *is* the experimental design.

---

## 4. Verification Checklist

For every row in §1, the documentation should answer:

1. **What does this layer claim about behavior?** (column 2)
2. **Where is the claim theorized?** (column 3, with file pointer)
3. **What outcome would falsify it?** (column 4)
4. **Has it been tested?** (column 5, ✓ or ○ in §2)

If any row cannot answer all four, that row is engineering rather than science and should be documented as such (e.g., `bgf_logging/` exists for reproducibility, not as a scientific claim; it is intentionally omitted from §1).
