# BGF Scientific Claims — Evidence Audit

**Last full audit:** 2026-05-13 (extends `docs/figure_status.md`'s figure-only audit to *every* scientific claim made in the paper, theoretical foundations, causal model, construct validity, evaluation protocol, and architecture rationale documents.)

This document is the canonical claim→evidence mapping. Every scientific claim in the thesis must appear here with a path to its supporting artifact and a status flag. New claims added to any document must add a row here in the same commit.

---

## Status legend

| Symbol | Meaning |
|---|---|
| ✅ | **VERIFIED** — claim is supported by an existing artifact (test, simulation output, or analysis JSON) and the artifact's numbers match the claim. |
| ⚠️ | **PARTIAL** — supporting artifact exists but is at pilot scale, uses a proxy policy, or has a known caveat. The claim is reported with the caveat in the paper. |
| ⏳ | **PENDING** — no artifact yet; infrastructure exists; execution is blocked on GPU/budget. A runner script is named for each. |
| ❌ | **UNVERIFIED** — number appears in docs but cannot be reproduced from current artifacts. Must be fixed before submission, either by running the missing experiment or by removing/marking the claim. |
| 📐 | **THEORETICAL** — derivation or argument, not an empirical measurement. The "evidence" column points to the proof/derivation. |

---

## Section A — Headline results (paper.md abstract, §5–§6)

| # | Claim | Evidence path | Status | Notes |
|---|---|---|---|---|
| A.1 | Cross-cultural Spearman ρ = +1.000 (6 ESS clusters, dry-run rule-based) | `analysis/cross_cultural_expanded_results.json` lines 36–42 | ⚠️ | `dry_run: true`, `policy_type: rule_based`. LLM-scale replication pending. Paper §6.6 carries the dry-run caveat. |
| A.2 | Cross-cultural Pearson r = +0.983, p = 0.0004 | `analysis/cross_cultural_expanded_results.json` lines 36–55 | ⚠️ | Same caveat as A.1. |
| A.3 | WVS Wave 7 replication r = +0.977 | `analysis/cross_cultural_expanded_results.json` line 57 | ⚠️ | Same caveat as A.1. |
| A.4 | Cross-model ΔB_RLHF: Mistral −17.6%, Qwen −30.0%, GPT-4o-mini +40.3% | `analysis/cross_model_results.json` + `docs/figure_status.md` Fig 10 cross-check table | ✅ | Numbers verified against canonical table 2026-05-11. N=20, T=10 (acknowledged as smallest tier, paper §4.1). |
| A.5 | Trust gradient Spearman ρ = 0.800, exact p = 0.167 | `analysis/tables/trust_gradient.json` lines 36–48 + Fig 9 cross-check | ✅ | Per-group means verified to 4 decimals. |
| A.6 | Feature importance: trust β=+0.287, risk β=−0.187, social_activity β=+0.146, train accuracy 0.608 | `analysis/tables/feature_importance.json` lines 6–17 + Fig 11/12 cross-check | ✅ | N=9,000 ESS Austrian R11 volunteering observations. |
| A.7 | Policy intervention Δcoop = [−0.0151, +0.0008, +0.016, +0.045] at δ ∈ {0%, 5%, 10%, 20%} | `analysis/tables/policy_intervention.json` lines 6–47 | ✅ | n=5 seeds, N=200, T=30, intervention_round=15. |
| A.8 | Condition D rule-based Gini = 0.325 ± 0.001 (empirical EU range) | `experiments/rule_based_seed_{42,43,44}/summary.json` (3 seeds) | ⚠️ | Aggregate not yet emitted to `analysis/paper_numbers.json` under a clean key; needs `python scripts/compute_paper_numbers.py` consolidation pass. |
| A.9 | Memory ablation: persona fidelity M0=0.609 → M3=0.742 monotone | None | ❌ | **CRITICAL** — `docs/figure_status.md` Fig 15 already flags this: all 24 ablation runs used `policy: mock`, which ignores memory entirely. Real M0–M3 cooperation comes back identical at 0.117 across all four levels. **Action:** `bash scripts/run_memory_ablation_llm.sh` (GPU). Until run, the paper.md §6.9 Table 7 and abstract claim must be marked **unverified** or removed. |
| A.10 | Condition A (ungrounded): cooperation rate 0.962, B_RLHF 0.712, Gini 0.625 | `analysis/paper_numbers.json` `condition_a_ablated` block (phase_c_comparison, N=50, T=30, Mistral-7B, seed=42) | ⚠️ | Verified against canonical Fig 2 numbers; `experiments/phase_c_comparison/*.parquet` are missing on disk (Fig 8 status). Cached PNG matches JSON. |
| A.11 | Condition B (grounded): cooperation rate 0.582, B_RLHF 0.420, Gini 0.260 | Same as A.10 | ⚠️ | Same caveat. |
| A.12 | B_RLHF reduction A→B ≈ 60% (primary headline) | Derived from A.10/A.11: (0.712−0.420)/0.712 ≈ 41%; paper abstract says "≈60%" | ❌ | **Numerical inconsistency** — paper abstract should be checked against the canonical JSON. The figure-audit's verified values yield 41%, not 60%. |

## Section B — Theoretical foundations (`docs/theoretical_foundations.md`)

| # | Claim | Evidence path | Status | Notes |
|---|---|---|---|---|
| B.1 | Information-theoretic bound: `KL(π_human||π_LLM+G) ≤ KL(π_human|g(x)||π_LLM+G) + δ` (data-processing inequality, §1.3) | `docs/theoretical_foundations.md` §1 derivation; standard Cover & Thomas (2006) Ch. 2 result | 📐 | Pure derivation, no measurement required. Empirical test of the bound's tightness: per-subgroup KL contribution should track the trust-gradient pattern (cross-link to A.5). |
| B.2 | ESS is an approximate sufficient statistic for human action distribution | Indirect: A.6 feature-importance AUC=0.640; A.5 trust-gradient ρ=0.800; A.1 ρ=1.0 | ⚠️ | These are *consistent* with sufficiency; do not formally prove it. A direct sufficiency test (compare full-ESS-grounded vs trust-only-grounded conditions) is not yet implemented. **Future ablation, P3 priority.** |
| B.3 | Generative-social-science minimality: ablating grounding destroys macro pattern | A.10 vs A.11 (grounding ablation), A.9 (memory ablation, ⚠️), topology ablation (`scripts/pipeline_topology.sh` outputs) | ⚠️ | Grounding-ablation evidence valid (A.10/A.11). Memory-ablation evidence ❌. Topology-ablation evidence ✅ via `analysis/networks/` GEXF exports. Joint claim downgraded to ⚠️ until A.9 fixed. |
| B.4 | Dual-process: persona × RAG factorial yields positive interaction | `metrics/mediation.py` implemented; no published `analysis/tables/mediation.json` output yet | ⏳ | **CPU-only run pending**: build `analysis/mediation_summary.py` to aggregate the existing factorial cells in `experiments/`. See §F. |
| B.5 | RLHF cooperator-prior derivation predicts ΔB_RLHF < 0 sign | A.4 (Mistral, Qwen confirm; GPT-4o-mini anomaly disclosed) | 📐 | Derivation in §4.1–4.3; A.4 is the empirical test. Two of three models support the prediction; the GPT-4o-mini inversion is treated as an alignment-methodology moderator and is honestly disclosed in paper §6.6 and §9 (Limitation 3). |

## Section C — Causal model and negative controls (`docs/causal_model.md`)

| # | Claim | Evidence path | Status | Notes |
|---|---|---|---|---|
| C.1 | Backdoor criterion satisfied by researcher-assigned treatment | `docs/causal_model.md` §6 + §9 derivation | 📐 | Pearl identification by construction (exogeneity). Pure derivation. |
| C.2 | E-value (cooperation ratio) ≈ 2.04 | `docs/causal_model.md` §7 closed-form from A.10/A.11 | ✅ | Numerically verifiable from headline ratios. |
| C.3 | E-value (Gini ratio) ≈ 3.62 | Same | ✅ | Same. |
| C.4 | 2×2 factorial mediation decomposition (persona, RAG, interaction) | `metrics/mediation.py` implemented + `tests/test_mediation.py` ✓ | ⚠️ | Code + unit tests exist; primary-scale factorial output not yet aggregated to `analysis/tables/mediation.json`. Same blocker as B.4. |
| C.5 | Negative control: Padded (Condition P) | `decision/padded_ablation_policy.py` exists | ⏳ | Policy file present; **no N=500/T=30 run yet**. Runner script `scripts/run_padded_control.py` exists (per the original plan). Pre-registered. |
| C.6 | Negative control: Scrambled-ESS (Condition S) | `decision/scrambled_rag_policy.py` — does not exist | ⏳ | **Code not yet written.** ~80 LOC on top of `sql_rag.py`. Marked in `architecture_rationale.md` §2 as ○ outstanding. |
| C.7 | Negative control: Fabricated demographics (Condition F) | Does not exist | ⏳ | Same as C.6; ~80 LOC. |
| C.8 | Sensitivity table predicted ordering {A, P, S, F, B} | `docs/causal_model.md` §10 (predictions only; data not yet collected) | ⏳ | Requires C.5–C.7 runs. Adjudication rule pre-registered with Spearman ρ≥0.9 threshold. |

## Section D — Construct validity (`docs/construct_validity.md`)

| # | Claim | Evidence path | Status | Notes |
|---|---|---|---|---|
| D.1 | ESS-item ↔ behavioral-paradigm ↔ BGF-action mapping | `docs/construct_validity.md` §1 table with published anchors (Berg 1995, Henrich 2010, Fehr & Gächter 2002, Holt & Laury 2002, Falk 2018) | 📐 | Literature mapping; not an empirical claim about BGF. Anchor citations added to `paper/references.bib`. |
| D.2 | BGF (−3, +12/cooperator) payoff matches canonical PGG multiplier range (m ∈ [2,5], Ledyard 1995) | `environment/economy.py` source + Ledyard 1995 anchor | 📐 | Construction; payoff parameters explicit in code and documented in §2.1. |
| D.3 | H9 — BGF cooperation correlates with Herrmann 2008 / Henrich 2010 PGG country rates | `analysis/tables/h9_cross_cultural_behavioral.json` (ρ = +0.886, exact p = 0.033) | ✅ | Spearman ρ = +0.886 (6 clusters), exact two-tailed permutation p = 0.033. Formally significant at α = 0.05. Addresses Limitation 11 circularity. |
| D.4 | Glaeser et al. (2000) attitude↔trust-game correlation r ≈ 0.20–0.35 | Glaeser 2000 paper (now in bib) | 📐 | Literature claim, not BGF-specific. Used to bound the realistic gap between ESS attitudes and behavioral outcomes. |

## Section E — Evaluation protocol (`docs/evaluation_protocol.md`)

| # | Claim | Evidence path | Status | Notes |
|---|---|---|---|---|
| E.1 | Benjamini–Hochberg FDR procedure implementation | `metrics/statistical_inference.py:benjamini_hochberg` + `tests/test_statistical_inference.py` | ✅ | Unit-tested. Monotonicity enforcement verified. |
| E.2 | Bootstrap percentile CI (2,000 resamples, fixed seed 42) | `metrics/statistical_inference.py:bootstrap_ci` + tests | ✅ | Unit-tested. |
| E.3 | Cohen's d / Hedges' g effect sizes (small-n bias correction) | `metrics/statistical_inference.py` + tests | ✅ | Unit-tested. |
| E.4 | A priori power analysis (MDE ≈ 1.32 at n=10, α=0.05, two-sided MWU) | `docs/evaluation_protocol.md` §6 derivation using `statsmodels.stats.power` | ⏳ | Derivation given in §6; reproducer script `analysis/power_curves.py` not yet committed (CPU-only, ~30 LOC). |
| E.5 | BRM Dirichlet weight-sensitivity ≥ 90% of simplex | `analysis/tables/brm_sensitivity.json` (5,000 Dirichlet samples, 100% pass, min Δ = 0.12) | ✅ | All four vertex deltas strictly positive (jsd: 0.156, gini_gap: 0.285, coop_gap: 0.380, stability: 0.120). Verdict: ROBUST. Analytic certificate emitted. |
| E.6 | Convergent-evidence forest plot (H1–H9) | `analysis/tables/forest_plot.json`, `analysis/figures/forest_plot.png` | ✅ | Forest plot generated with verified effect sizes for H1, H2, H5, H7, H9. Pending rows (H3, H4, H6, H8) are placeholders. |

## Section F — Architecture rationale (`docs/architecture_rationale.md`)

Architectural commitments (every layer must have a falsifiable consequence; see §1 table in that doc).

| # | Layer | Test status |
|---|---|---|
| F.1 | `population/ess_grounding.py` (Φ mapping) | ⏳ Scrambled-Φ ablation pending (C.6) |
| F.2 | `population/persona_synthesizer.py` (NL persona) | ✅ V0–V4 prompt ablation ladder (paper §3.6) |
| F.3 | `decision/sql_rag.py` (population norms) | ✅ Persona × RAG factorial mediation (C.4, ⚠️ aggregation pending) |
| F.4 | `decision/graph_rag.py` (social position) | ⏳ Single-vs-dual RAG ablation pending |
| F.5 | `agents/memory.py` (hierarchical memory) | ❌ Mock-policy data only (A.9); LLM re-run blocking |
| F.6 | `environment/economy.py` (PGG payoffs) | ⏳ Continuous-action ablation pending |
| F.7 | `environment/network.py` (small-world + random) | ✅ Topology ablation (`pipeline_topology.sh`) + GEXF artifacts |
| F.8 | `environment/institutions.py` | ⏳ `no_institutions` row in `causal_model.md` §3 — runner not committed |
| F.9 | `simulation/kernel.py` (sync event loop) | ⏳ Async ablation not implemented |
| F.10 | `decision/prompt_builder.py` (V0–V4 ladder) | ✅ Ablation results in paper §3.6 |
| F.11 | `decision/output_parser.py` (strict JSON + regex fallback) | ✅ `tests/test_output_parser.py` |
| F.12 | `tracker/experiment_index.parquet` | ✅ 192 experiment runs registered |

## Section G — Test suite (per-module coverage)

| Module | Test file | Status |
|---|---|---|
| `metrics/behavioral_realism.py` (BRM, B_RLHF) | `tests/test_behavioral_realism.py` | ✅ |
| `metrics/inequality.py` (Gini) | `tests/test_gini_canonical.py`, `tests/test_gini_range_condition_ab.py` | ✅ |
| `metrics/trust_gradient.py` | `tests/test_trust_gradient.py` | ✅ |
| `metrics/cross_cultural.py` | `tests/test_cross_cultural.py`, `tests/test_cross_cultural_expanded_integrity.py` | ✅ |
| `metrics/cross_model.py` | `tests/test_cross_model_scaled.py` | ✅ |
| `metrics/mediation.py` | `tests/test_mediation.py` | ✅ |
| `metrics/persona_decay.py` | `tests/test_persona_decay.py` | ✅ |
| `metrics/statistical_inference.py` | `tests/test_statistical_inference.py` | ✅ |
| `metrics/complexity.py` (phase transitions) | `tests/test_complexity.py` | ✅ |
| `metrics/calibration.py` (ESS calibration) | `tests/test_calibration.py` and `metrics/calibration.py:__main__` validator | ✅ |
| `decision/sql_rag.py`, `decision/graph_rag.py` | `tests/test_rag.py` (+ subset specifically targeted) | ✅ |
| `decision/output_parser.py` | `tests/test_output_parser.py` | ✅ |
| `agents/memory.py` | `tests/test_memory*.py` | ✅ |
| Total | 1,441 test functions across 122 test files | ✅ |

---

## Section H — Outstanding work (consolidated)

Sorted by severity. Each item maps to a row above so progress is auditable.

### H.1 — CRITICAL (blocks defense)

- **A.9 — Memory ablation real-data run**. Submission blocker per `figure_status.md`. Action: `bash scripts/run_memory_ablation_llm.sh` (GPU, ~6–8 h). Until executed, the paper §6.9 / abstract claim "M0=0.609 → M3=0.742" must be either removed or marked unverified.
- **A.12 — Abstract B_RLHF reduction figure**. The "≈60%" headline does not match the canonical paper_numbers.json values (which yield ≈41%). Action: reconcile by either (a) updating the abstract to the verified 41%, or (b) sourcing the 60% claim from a different canonical experiment and citing it. CPU-only inspection of `analysis/paper_numbers.json`.
- **C.5 / F.5 (also Limitation 8 in paper §9)** — Padded ablation at primary scale. Infrastructure exists; ~6 GPU-hours needed.

### H.2 — HIGH (pre-registration commitments, GPU-blocked)

- C.6 — Scrambled-ESS policy file (~80 LOC) + run (~6 GPU-h).
- C.7 — Fabricated-demographics policy file (~80 LOC) + run (~6 GPU-h).
- C.8 — Adjudication table population (depends on C.5, C.6, C.7).

### H.3 — MEDIUM (CPU-only, no blocker)

- B.4 / C.4 — Build `analysis/mediation_summary.py` aggregating existing factorial-cell experiments → `analysis/tables/mediation.json`. ~5 min runtime.
- ~~D.3 — Build H9 cross-cultural behavioral comparison~~ ✅ **DONE** (2026-05-19): `analysis/tables/h9_cross_cultural_behavioral.json` (ρ = +0.886, p = 0.033).
- E.4 — Build `analysis/power_curves.py` (`statsmodels.stats.power` MDE tables for H1–H9). ~5 min.
- ~~E.5 — Build `analysis/brm_sensitivity.py`~~ ✅ **DONE** (2026-05-19): `analysis/tables/brm_sensitivity.json` (5,000 samples, 100% pass, ROBUST certificate).
- ~~E.6 — Build `analysis/forest_plot.py`~~ ✅ **DONE** (2026-05-19): `analysis/tables/forest_plot.json` + `analysis/figures/forest_plot.png`.
- A.8 — Re-aggregate Condition D Gini=0.325±0.001 into a canonical `paper_numbers.json` key.

### H.4 — LOW (literature / theoretical, no empirical work needed)

- B.2 — A future formal sufficiency test by varying which ESS subset enters Φ. Listed as P3 in `TOP_TIER_RESEARCH.md`.

---

## Section I — Reconciliation with `figure_status.md`

`docs/figure_status.md` is the authoritative figure-level audit. This document extends it to all scientific claims. Where the two overlap, `figure_status.md` is binding for figure-specific numbers. The pre-submission checklist at the bottom of `figure_status.md` is also reproduced here as the canonical work-to-do list before submission (see §H.1 / H.2 above).

---

## Section J — How to extend this audit

When adding a new scientific claim to any document:

1. Add a row in the appropriate section (A–F) above with: claim, evidence path, status flag.
2. If status is ⏳ or ❌, add a corresponding entry under §H with the action that will close it.
3. If the claim references a figure, cross-link to its row in `figure_status.md`.
4. Run `python scripts/compute_paper_numbers.py` after any new experiment lands to refresh `analysis/paper_numbers.json` — that file is the canonical source of headline numbers.

A claim that lacks a row here, or whose row is ❌/⏳ without a §H entry, is not a defensible thesis claim and must be marked as such inline (`[evidence: pending]`) until the audit is closed.
