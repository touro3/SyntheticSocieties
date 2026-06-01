# 2. Rule-based ESS policy as the deterministic baseline (Condition D)

- Status: accepted
- Date: 2026-02-12
- Deciders: BGF core
- Tags: experimental-design, scientific-method

## Context

Reviewers of LLM-agent papers routinely ask: "how would you know the
empirical grounding is doing the work, rather than the LLM filling in
plausible-sounding behaviour from its prior?" Without a non-LLM control,
any positive Condition B result is open to the alternative that
the LLM is the source of realism, not the ESS data.

We needed a **deterministic, non-LLM, ESS-grounded** policy that uses the
same `Φ: D_ESS → Profile` map as the LLM conditions but takes decisions
via a closed-form rule rather than via prompted inference. Three options:

1. **Random baseline.** Trivially controls for "the action space matters,"
   nothing else.
2. **Static template / heuristic** (e.g. always cooperate if income >
   median). Easy but ad hoc; the rules become a moving target during
   review.
3. **Logistic regression on ESS predictors.** Reproducible, defensible,
   directly testable against the empirical cooperation correlates
   estimated in §6.7.

## Decision

Condition D is a `RuleBasedESSPolicy` (`decision/rule_based_ess_policy.py`)
that uses a fixed mapping derived from ESS-11 marginals: trust ×
risk-tolerance × competitiveness → softmax over the {work, save,
cooperate} action space, with `steal` reserved for hard-constrained
adversarial agents only. The mapping coefficients are fixed at module
load and not learned per-run, so the policy is **bitwise reproducible**
across seeds.

## Consequences

**Positive**

- Condition D gives BGF a positive result that does not depend on any
  LLM: Gini = 0.325 ± 0.001 at N=500, T=30, 3 seeds — within the
  Eurostat European empirical range. This is the **strongest** result
  in the paper because it has zero LLM-induced confounds.
- Cross-cultural validation (§8.3) uses the same Condition D layer,
  recovering Spearman ρ = +1.000 across six ESS clusters, which then
  replicates against WVS Wave 7 and Herrmann-Thöni-Gächter PGG.
- Condition D runs on CPU at ~zero inference cost, so seed counts are
  not GPU-budget-limited.

**Negative**

- A reviewer can dismiss Condition D as "just a regression model,
  not a synthetic society." We address this in §3.5 by explicitly
  framing D as the *grounding ceiling* — the realism achievable
  if the LLM perfectly executes the grounding signal — not as a
  competitor to Conditions A/B/C.
- Condition D's success makes the §8.1 N=100 Condition A vs B null
  result more conspicuous; that's a feature, not a bug, and is named
  "the `Φ`/`P_LLM` dissociation" in §1 / §10.

## Alternatives explicitly rejected

- **Hand-coded heuristics** rejected for reproducibility / cherry-picking
  risk.
- **Per-run fitted regression** rejected because the coefficients would
  become a hyperparameter — losing the bitwise reproducibility that is
  Condition D's main scientific value.
