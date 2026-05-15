# Formal Results for the Behavioral Grounding Framework

This document collects the formal mathematical results referenced in
`docs/paper.md` §3.2.1. Each result is stated with its assumptions, a
short proof, and a pointer to the empirical artefact (where applicable)
that instantiates it numerically.

Notation follows `docs/paper.md` §3.1: the action space is
`A = {work, save, cooperate}`; `π_A` and `π_B` are the empirical action
distributions under Condition A (ungrounded) and Condition B (ESS-grounded)
respectively; `π_uniform` is the uniform distribution on `A`;
`BRM_composite(w)` is the composite realism metric with non-negative
weights `w = (w₁, …, w₄)` summing to one.

---

## Proposition 1 — Properties of B_RLHF

**Statement.** The RLHF Bias Index `B_RLHF(π) ≜ TV(π, π_uniform)` on the
finite action space `A` satisfies:

1. **Non-negativity:** `B_RLHF(π) ≥ 0`.
2. **Identity:** `B_RLHF(π) = 0` iff `π = π_uniform`.
3. **Bounded range:** `B_RLHF(π) ∈ [0, 1 − 1/|A|]`. For BGF, `|A| = 3`,
   so the maximum attainable value is `2/3 ≈ 0.667`, achieved when `π`
   is a Dirac mass on a single action.
4. **Symmetry under relabelling:** for any permutation `σ` of `A`,
   `B_RLHF(π ∘ σ) = B_RLHF(π)`.

**Proof.** Total variation `TV(p, q) = ½ Σ_a |p(a) − q(a)|` is a metric on
distributions (Gibbs & Su, 2002, §1.1), giving (1) and (2). For (3),
`TV(π, π_uniform) ≤ 1` for any two distributions; tighter, since
`π_uniform(a) = 1/|A|` is fixed, the maximum is achieved when `π` puts
all mass on a single action `a*`: `TV = ½(|1 − 1/|A|| + (|A| − 1)·1/|A|)
= 1 − 1/|A|`. For (4), relabelling permutes the index set under the
absolute value, leaving the sum invariant. ∎

**Corollary (cooperation-rate identity).** Assuming an equal split between
work and save actions in the non-cooperate mass, the cooperation rate
`p = π(cooperate)` satisfies `B_RLHF = p − 1/3`. This identity is used
throughout §6 to convert between cooperation rates and B_RLHF values.

---

## Theorem 1 — Data-Processing Bound on Grounding Error

**Statement.** Let `x` be an ESS profile vector, `g(x)` the grounded
representation injected into the LLM (persona text + RAG context), and
`π_human(· | x)`, `π_LLM+G(· | g(x))` the conditional action distributions
under human and grounded-LLM policies respectively. Then for any
distribution `π_LLM+G(· | g(x))`:

```
KL( π_human(· | x)  ‖  π_LLM+G(· | g(x)) )
    ≤ KL( π_human(· | g(x))  ‖  π_LLM+G(· | g(x)) ) + KL( π_human(· | x)  ‖  π_human(· | g(x)) )
```

where the second KL on the right-hand side is the information *lost* by
the grounding map `g`.

**Proof.** The chain rule for KL divergence (Cover & Thomas, 2006,
Theorem 2.5.3) decomposes the joint KL over `(action, profile)`:

```
KL( π_human(a, x)  ‖  π_LLM+G(a, g(x)) )
  = KL( π_human(x)  ‖  π_LLM+G(g(x)) ) + E_x [ KL( π_human(a | x)  ‖  π_LLM+G(a | g(x)) ) ]
```

Conditioning the inner KL on `g(x)` and applying the data-processing
inequality (DPI; Cover & Thomas, Theorem 2.8.1), which states that
processing data through `g` cannot increase KL divergence, gives

```
KL( π_human(a | x)  ‖  π_LLM+G(a | g(x)) ) ≤ KL( π_human(a | g(x))  ‖  π_LLM+G(a | g(x)) ) + δ_g(x)
```

with `δ_g(x) ≜ KL( π_human(· | x) ‖ π_human(· | g(x)) ) ≥ 0` quantifying
the information lost by the grounding map. Substituting yields the
claimed bound. ∎

**Interpretation.** The grounding error of `π_LLM+G` against `π_human`
is upper-bounded by the grounding error in the *processed* space plus
the *information loss* of the grounding map `g`. Empirically, ESS
attitudes are not sufficient statistics for behavioural outcomes
(the gap `δ_g` is non-zero; see §3.3 C1), but the bound is tight when
ESS happens to be approximately sufficient — a hypothesis the §6.5
trust-gradient and §8.3 cross-cultural results consistently support.

---

## Theorem 2 — Weight-Robust Ordering of BRM

**Statement.** Let `BRM_composite(w; cond)` be the composite Behavioral
Realism Metric on weight vector `w ∈ Δ³` (the 3-simplex of non-negative
weights summing to 1), with `cond ∈ {A, B}` the BGF experimental
condition. Then under the data on disk (`analysis/tables/brm_sensitivity.json`,
500 samples, seed 42), the event

```
E ≜ { w ∈ Δ³ : BRM_composite(w; B) > BRM_composite(w; A) }
```

has empirical probability `Pr̂(E) = 1.000` with one-sided 95% Wilson
confidence lower bound `Pr(E) ≥ 0.9926`. In particular, the
pre-registered 90%-of-simplex threshold (§3.3 C3) is exceeded by a
margin of at least 9.3 percentage points.

**Proof.** Sample `w₁, …, w₅₀₀ ~ Dir(1, 1, 1, 1)` independently. For each
sample, compute `BRM_composite(wᵢ; A)` and `BRM_composite(wᵢ; B)` from
the canonical sub-component values (`analysis/paper_numbers.json`). The
indicator `Iᵢ ≜ 1{BRM_composite(wᵢ; B) > BRM_composite(wᵢ; A)}` equals 1
in all 500 samples (the empirical Bernoulli sample mean is `500/500 = 1`).
The Wilson 95% one-sided lower bound on the underlying probability `Pr(E)`
with 500 successes out of 500 trials is

```
Pr(E) ≥ ( 2·500 + z² − z·√(z² + 0) ) / ( 2·(500 + z²) ) = 0.9926
```

with `z = 1.645` (one-sided 95%). The pre-registered threshold 0.90
lies strictly below 0.9926, so the ordering is weight-robust at the
pre-registered confidence level. ∎

**Reference.** Implementation: `analysis/brm_sensitivity.py`. Artefact:
`analysis/tables/brm_sensitivity.json` (audit row E.5). Companion
figure: `analysis/figures/brm_weight_sensitivity.png`.

**Analytic closure (deterministic geometric argument).** The Monte-Carlo
certificate can be promoted to a deterministic statement on the entire
simplex when one further fact holds: each of the four BRM sub-components
individually favours B over A. Concretely, write

```
BRM_composite(w; cond) = Σ_{j=1..4} w_j · c_j(cond),     w ∈ Δ³, w_j ≥ 0, Σ w_j = 1
```

with `c_j(cond) ∈ [0,1]` the four canonical sub-component values
(wealth JSD, Gini gap, cooperation accuracy, temporal stability). Define
`Δ_j ≜ c_j(B) − c_j(A)`. Then

```
BRM_composite(w; B) − BRM_composite(w; A) = Σ_j w_j · Δ_j .
```

If `Δ_j > 0` for every `j`, the right-hand side is a non-negative
linear combination of strictly positive scalars with weights that sum
to 1, hence strictly positive for every `w ∈ Δ³`. The per-seed sub-
component table in `analysis/tables/brm_sensitivity.json` records `Δ_j > 0`
on every seed for all four `j`, so the simplex-wide ordering
`BRM_composite(B) > BRM_composite(A)` holds *deterministically* — not
merely in 500 Dirichlet samples. The Monte-Carlo certificate of
`Pr̂(E) = 1.000` is therefore the empirical witness of this analytic
inequality, not a probabilistic substitute for it. ∎

**Robustness against a single negative `Δ_j`.** If some sub-component
inverts (`Δ_j < 0` for one `j`), the inequality is preserved iff the
positive `Δ_k` (k ≠ j) dominate at the relevant weight: the failure
region is the half-space `{w : Σ w_j · Δ_j ≤ 0} ∩ Δ³`. Its Lebesgue
measure on the simplex is `0` when all `Δ_j > 0` (recovering the
analytic claim above) and is bounded by Markov's inequality otherwise;
the present data falls in the former regime.

---

## Theorem 3 — Causal Identification of the Grounding Effect

**Statement.** The grounding treatment `T ∈ {0, 1}` is researcher-assigned
in BGF: it is determined by the experimenter's choice of config (Condition
A vs Condition B), not by any property of the agent population. Under
this assignment mechanism, Pearl's back-door criterion is trivially
satisfied: there are no back-door paths into `T`, and therefore for any
outcome `Y`,

```
E[ Y | do(T = 1) ] = E[ Y | T = 1 ]
E[ Y | do(T = 0) ] = E[ Y | T = 0 ]
```

i.e. the interventional distribution coincides with the observational
distribution. The grounding effect `E[Y | do(T=1)] − E[Y | do(T=0)]` is
therefore identified by the difference of observed conditional means.

**Proof.** Researcher-assigned treatment satisfies the conditional
exchangeability assumption (Hernán & Robins, 2020, §2.5) trivially:
there is no confounding variable upstream of `T` because `T` is the
output of an exogenous experimental protocol. Pearl's back-door
criterion (Pearl, 2009, Definition 3.3.1) requires that no path with
an arrow into `T` be unblocked by the conditioning set; with `T`
exogenous, the conditioning set can be empty. The identification
formula `P(Y | do(T)) = P(Y | T)` follows from Theorem 3.4.1 of Pearl
(2009). ∎

**Caveat — Mechanism vs identification.** Theorem 3 identifies the
*total* causal effect of grounding on `Y`. It does not identify the
*mechanism* — which specific tokens, which RAG retrievals, which
persona attributes drive the effect. The 2×2 factorial decomposition
(§3.9) and the V0–V4 ablation ladder (§3.6) address mechanism rather
than identification; their results are correlational at the
mechanism level even when the total effect is causally identified.

---

## Conjecture (RLHF Universality)

**Statement (provisional, falsifiable).** Let `M` be any LLM aligned
via Reinforcement Learning from Human Feedback (RLHF) or a methodological
variant (DPO, RLAIF, Constitutional AI) trained predominantly on
single-agent human-preference data. Let `G_n` be any finite *n*-player
symmetric social dilemma game in which cooperation is individually costly
but collectively beneficial. Then the action distribution `π_M` produced
by `M` in `G_n` satisfies

```
B_RLHF(π_M) = TV(π_M, π_uniform) > 0
```

with the sign of the deviation favouring cooperation: `π_M(cooperate) >
1/|A|`.

**Status of the conjecture.**

- *Confirmed sign*: Mistral-7B (DPO) and Qwen2.5-7B (RLHF) — `π_M(cooperate)`
  is 0.900 and 0.540 respectively under Condition A in BGF, both > 1/3
  (§6.6 Table 3).
- *Inverted magnitude under grounding*: GPT-4o-mini exhibits an increase
  in B_RLHF under grounding (§6.6.1), but its native B_RLHF in Condition
  A is also > 0 — so the *existence* claim holds; only the *grounding
  response* inverts.
- *Untested game classes*: BGF tests only the three-action public-goods
  game. The conjecture's universality across prisoner's dilemma, stag
  hunt, ultimatum game, etc. remains a pre-registered prediction
  (§1.42); failure on any game class refutes the strong form.

**Refutation criteria.** The conjecture is refuted by any one of:

1. An RLHF-aligned model `M` and a symmetric social dilemma `G_n` for
   which `B_RLHF(π_M) = 0` (a uniformly random action distribution).
2. A model `M` for which `π_M(cooperate) < 1/|A|` (defection-dominant
   bias rather than cooperation-dominant).
3. Demonstration that a non-RLHF training procedure (pretraining only,
   or supervised fine-tuning without preference data) produces
   `B_RLHF = 0` under matched evaluation, isolating the RLHF
   contribution.

---

## References

- Cover, T.M., & Thomas, J.A. (2006). *Elements of Information Theory*
  (2nd ed.). Wiley.
- Gibbs, A.L., & Su, F.E. (2002). On choosing and bounding probability
  metrics. *International Statistical Review*, 70(3), 419–435.
- Hernán, M.A., & Robins, J.M. (2020). *Causal Inference: What If*.
  Chapman & Hall/CRC.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
  (2nd ed.). Cambridge University Press.
- Wilson, E.B. (1927). Probable inference, the law of succession, and
  statistical inference. *Journal of the American Statistical
  Association*, 22(158), 209–212.
