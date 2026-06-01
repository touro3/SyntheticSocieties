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

**Statement (deterministic form).** Let
`BRM_composite(w; cond) = Σ_{j=1..4} w_j · c_j(cond)` be the composite
Behavioral Realism Metric, where `c_j(cond) ∈ [0,1]` are the four
sub-component scores (wealth JSD, Gini gap, cooperation accuracy,
temporal stability), `cond ∈ {A, B}`, and `w` ranges over the weight
3-simplex `Δ³ = { w ∈ ℝ⁴ : w_j ≥ 0, Σ_j w_j = 1 }`. Define the
component-wise advantage `Δ_j ≜ c_j(B) − c_j(A)`. Then

```
   BRM_composite(w; B) > BRM_composite(w; A)  for every w ∈ Δ³
                         ⟺
                   min_{j ∈ {1..4}} Δ_j > 0 .
```

That is, weight-robustness over the *entire* admissible weight space is
exactly equivalent to a four-number sign check — a strict mathematical
guarantee, not a sampled probability.

**Proof.** The difference functional

```
f(w) ≜ BRM_composite(w; B) − BRM_composite(w; A) = Σ_{j=1..4} w_j · Δ_j
```

is a *linear* function of `w` (the `Δ_j` are fixed constants once the
two conditions' sub-components are evaluated). `Δ³` is a compact convex
polytope whose vertices are the four standard basis vectors
`e_1, …, e_4` (`e_j` places all weight on component `j`). A linear
function on a compact convex polytope attains its minimum at a vertex
(fundamental theorem of linear programming; Boyd & Vandenberghe 2004
§4.2). Therefore

```
   min_{w ∈ Δ³} f(w) = min_{j} f(e_j) = min_{j} Δ_j ,
   max_{w ∈ Δ³} f(w) = max_{j} f(e_j) = max_{j} Δ_j .
```

(⇐) If `min_j Δ_j > 0` then `min_{w ∈ Δ³} f(w) = min_j Δ_j > 0`, so
`f(w) > 0`, i.e. `BRM_composite(w; B) > BRM_composite(w; A)`, for every
`w ∈ Δ³`. (⇒) Conversely, if `min_j Δ_j = Δ_{j*} ≤ 0`, then evaluating
at the vertex `w = e_{j*}` gives `f(e_{j*}) = Δ_{j*} ≤ 0`, a feasible
weight at which the ordering fails. Hence the equivalence. ∎

This supersedes the earlier Monte-Carlo certificate (500 Dirichlet
draws, Wilson lower bound `≥ 0.9926`): sampling can only ever produce a
high-confidence *estimate* of `Pr(f(w) > 0)`, whereas the linear-program
argument settles the value of `min_{w} f(w)` exactly. The Dirichlet
sweep's empirical fraction of `1.000` is now read as the numerical
*witness* of the analytic inequality `min_j Δ_j > 0`, not as the
evidentiary basis for it.

**Corollary (constrained weight polytopes).** Expert priors may
restrict weights to a sub-polytope `W ⊆ Δ³` (e.g. box constraints
`w_j ∈ [ℓ_j, u_j]`, or "cooperation accuracy is at least as important
as temporal stability", `w_3 ≥ w_4`). `W` is again a compact convex
polytope, so by the same argument

```
   min_{w ∈ W} f(w) = min_{v ∈ V(W)} Σ_j v_j Δ_j ,
```

where `V(W)` is the finite vertex set of `W`. Weight-robustness over
any such admissible region is therefore decided by enumerating the
vertices of `W` and checking the sign of `f` at each — still a finite,
exact computation with no sampling.

**Verifiable certificate.** The check reduces to emitting the vector
`(Δ_1, Δ_2, Δ_3, Δ_4)` and the scalar `min_j Δ_j`.
`analysis/brm_sensitivity.py --emit-certificate` writes these four
deltas and the verdict (`min_j Δ_j > 0 ⇒ ROBUST`) alongside the legacy
sweep fields in `analysis/tables/brm_sensitivity.json` (audit row E.5);
on the canonical sub-component values the certificate is satisfied
(equal-weight Δ = 0.235 > 0, and the simplex infimum `min_j Δ_j` is
itself positive). Companion figure:
`analysis/figures/brm_weight_sensitivity.png`.

**Reference.** Boyd, S. & Vandenberghe, L. (2004). *Convex
Optimization*, §4.2 (LP optima at polytope vertices). Implementation:
`analysis/brm_sensitivity.py`. Artefact:
`analysis/tables/brm_sensitivity.json`.

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
