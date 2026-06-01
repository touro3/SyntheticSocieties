# 3. BRM as a paired composite realism metric

- Status: accepted
- Date: 2026-02-19
- Deciders: BGF core
- Tags: metrics, scientific-method, jsd

## Context

To make "more realistic" a falsifiable claim, BGF needs a scalar metric
that (i) compares a simulated population against an empirical reference
distribution along multiple complementary axes, (ii) is bounded so it
can be cited with a fixed interpretation, and (iii) is paired (per-seed
comparable across conditions) rather than producing only a single
population summary.

Candidates we evaluated:

- **Single-axis JSD on the wealth distribution.** Loses the network /
  cooperation / temporal-stability dimensions; reviewers correctly
  point out that two populations can match on wealth and disagree on
  everything else.
- **Wasserstein distance only.** Information-rich but not bounded;
  hard to interpret as a "realism score."
- **Hand-weighted multi-axis composite.** Vulnerable to weight-cherry-
  picking.

## Decision

We define `BRM ∈ [0, 1]` as a composite of four bounded components:

1. **Wealth-JSD** — Jensen–Shannon divergence between simulated and ESS
   wealth distributions, normalised so 1 = identical.
2. **Gini-gap** — `1 - |G_sim - G_ESS|`, capped at 0.
3. **Cooperation accuracy** — `1 - |coop_sim - coop_ESS_proxy|`.
4. **Temporal stability** — JSD of the action distribution between the
   first and last quarter of the run.

The composite weights default to uniform (¼ each). **Proposition 3** in
the paper (`metrics/composite_brm.py` + `analysis/brm_sensitivity.py
--emit-certificate`) proves that the *ordering* of `BRM(A)`,
`BRM(B)`, `BRM(C)`, `BRM(D)` is invariant under any positive convex
weighting — so the headline ordering does not depend on the weight
choice. The certificate is regenerated in CI on every commit
(`.github/workflows/ci.yml`).

JSD uses **base-2 log** (`base=2` to `scipy.stats.entropy`) so the
component is in bits and the metric is genuinely bounded in [0, 1].
This was a silent bug pre-2026-05-20 (audit A2.1 in
`docs/AUDIT_DATA_METRICS_LOGGING.md`); fixing it requires regenerating
all pre-fix BRM numbers.

## Consequences

**Positive**

- One scalar per run, paired across conditions.
- Bounded, with a stated interpretation: 1 = perfect realism, 0 =
  maximally distant from ESS.
- Weight robustness proved analytically and re-emitted as a CI artefact
  so reviewers can audit the proof against current code.

**Negative**

- Composite metrics can hide an axis-specific blow-up (e.g. great
  wealth match but pathological cooperation rate). We mitigate this
  by also reporting the four components separately in every table.
- The construct-validity discussion (§3.3, C1–C4) had to grow to
  accommodate this — readers must explicitly understand what BRM
  does and does not measure.

**Mitigation**

- `metrics/composite_brm.py:brm_components()` is the canonical breakdown
  every results table is required to cite.
- `analysis/brm_sensitivity.py` regenerates the Proposition 3
  certificate on every CI run; the test
  `tests/test_brm_certificate_emission.py` fails the build if the
  on-disk certificate disagrees with the freshly emitted one.
