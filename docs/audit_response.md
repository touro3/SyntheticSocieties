# Research-Audit Response: Construct Validity, OOD, Memory, Alignment

This document records the methodology and rationale for the scaffolding added
to close the four gaps raised in the research audit. It is written for direct
reuse in the thesis methodology chapter.

The central claim under defense: **empirical ESS grounding causally produces
behavioral realism** (not an artifact of prompt entropy, metric circularity,
inert memory, or RLHF alignment).

---

## Placebo controls & semantic isolation

**Threat:** realism gain could be mere prompt heterogeneity, not sociological
coherence.

**Response:** 3-arm contrast — grounded (`population.source: empirical`),
placebo (`placebo`), unconditioned (`synthetic`). The placebo arm
(`population/placebo_demographics.py`) keeps the ESS demographic skeleton but
independently permutes the sociological trait vector: marginals preserved,
joint structure destroyed. Runner: `scripts/run_placebo_ablation.py`. Analysis:
`analysis/placebo_variance.py` (reuses the existing one-way ANOVA, k-condition
generic) separates the *semantic* component (grounded vs placebo) from the
*any-conditioning* component (placebo vs unconditioned).

**Reading:** a large grounded→placebo realism drop with a small
placebo→unconditioned drop ⇒ the gain is semantic and the claim holds.

---

## Out-of-distribution validation

**Threat:** BRM is circular — scored against the same ESS source used to
condition agents.

**Data constraint (documented honestly):** the local parquet is **AT-only**
(866 rows, single country). Microdata country holdout is impossible; cross-
country variation exists only at the **cluster-benchmark** level, exactly as
the existing cross-cultural module already operates.

**Response:** leave-one-cluster-out (LOCO) — `population/ood_split.py` resolves
disjoint train/eval cluster partitions; `scripts/run_ood_validation.py` fits
the trust→behavior calibration on train clusters and predicts the held-out
cluster whose published ESS-11 benchmark **never entered the fit**. Small OOD
error ⇒ realism generalizes rather than echoing the conditioning source.
`population/sampling.py` gained `country_filter`/`exclude_countries` (forward-
compatible; loudly errors if a filter empties the AT-only frame rather than
silently resampling the wrong cohort).

---

## Causal diagnostics & memory activation

**Threat:** the memory architecture may be present but behaviorally inert
("myopic agents").

**Response:**
- `bgf_logging/memory_diagnostics.py` — per agent/round retrieval frequency
  and **citation rate** (fraction of retrieved items that survive into the
  rendered prompt). Wired via a non-invasive pass-through wrapper; the prompt
  builder and kernel are unmodified.
- Memory-deletion ablation — `simulation.intervention_hooks.delete_betrayal_memories`
  + `scripts/run_memory_deletion_ablation.py`: wipe a specific betrayal memory
  from the treatment cohort, keep it in control, measure divergence in
  subsequent cooperation toward the betrayer. Non-zero divergence ⇒ memory is
  load-bearing.
- Intervention hooks — `simulation/intervention_hooks.py` + a `scarcity`
  injection type added to `environment/world.py` (riding the existing
  `pending_injections` queue; no kernel change). `trust_shock`, `scarcity`,
  `wealth_shock` enable measuring directional responses against the
  social-science literature.

---

## Alignment bias & system prompts

**Threat:** RLHF "assistant helpfulness" priors may masquerade as emergent
societal behavior; model-variant swapping was not first-class.

**Audit finding (system prompts):** the existing prompts in
`decision/system_prompts.py` are **already mechanically neutral** — they
describe action mechanics and JSON schema only, with no "you are a helpful
assistant", no strategy advice, and persona blocks are descriptive not
prescriptive. No helpfulness lexicon was found. This is a positive result and
is recorded here as evidence, not a defect to fix.

**Response (hardening):**
- New `PERSONA_LOCKED_SYSTEM_PROMPT` (registry key `persona_locked`):
  mechanics byte-identical to `NEUTRAL`, but explicitly forbids AI/assistant/
  meta framing and forces reliance on the ingested persona — isolates any
  residual RLHF identity leakage as a single manipulable factor.
- `decision/model_config.py` now honors `BGF_MODEL_ID`, `BGF_BACKEND_TYPE`,
  and `BGF_MODEL_VARIANT` (`instruct|base|uncensored`) environment variables,
  so base vs instruct vs uncensored models can be swapped without code edits.
  Unset env ⇒ existing behavior preserved exactly (including the
  `.mistral_7b()` / `.gpt4o_mini()` classmethods). `model_variant` is recorded
  for provenance in results tables.

---

## Scope & reproducibility

All scaffolding runs inside the existing container on mock/rule-based policy
for CI; GPU LLM sweeps are user-triggered. New config keys default to
`None`/off so the 105 existing experiments and the full test suite remain
green. Every runner logs full per-round decision trajectories (not just
endpoint distributions) for thesis methodology audit.
