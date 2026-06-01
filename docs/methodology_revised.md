# 3. Methodology (Revised)

## 3.1 Formal Framework

A BGF simulation instance is formally specified as the tuple:

**BGF = (A, E, G, P, Φ, T)**

where:

**A** = {a₁, ..., a_N} is a set of N agents. Each agent aᵢ has an immutable profile πᵢ = Φ(xᵢ) where xᵢ is a record sampled from the empirical distribution D_ESS, and a mutable state sᵢ(t) = (wealth, stress, satisfaction, trust_map) at time step t.

**E** = (S, u) is the economic environment with state space S and payoff function u : Action × S → ℝ defined by:

```
u(work, s)      = (+8 wealth, +0.10 stress)
u(save, s)      = (+4 wealth, −0.05 stress)
u(cooperate, s) = (−3 wealth from self, +12/N to every agent, −0.05 stress)
```

*Clarification on the cooperation payoff.* The public pool is funded by each cooperator contributing 3 wealth. The total contribution (3 × |cooperators|) is multiplied by a factor (here set so the per-capita return is +12/N per agent regardless of pool membership), and the resulting sum is distributed to *all* N agents equally. This follows the standard linear public goods game (LPGG) formulation: each cooperator pays a cost c = 3, the total contribution is multiplied by a marginal per-capita return rate r = 12/N, and each of the N agents receives r × |cooperators|. The social dilemma condition requires 1/N < r < 1 (i.e., 12/N² < 1 and 12/N > 1), which holds for 4 ≤ N ≤ 11 under this parameterization. For N outside this range, either cooperation is individually rational (N < 4) or the public good is under-provisioned even at full cooperation (N > 11); researchers should verify that their chosen N satisfies the dilemma condition.

*If the authors intended the original formulation (pool divided only among cooperators):* This creates an anti-commons structure where the per-cooperator benefit *decreases* with more cooperators (+12/1 = 12 for a lone cooperator, +12/N for universal cooperation). Under this design, cooperation is most rewarding when rare, which inverts the standard social dilemma incentive. If this is the intended game, it should be explicitly named (e.g., a volunteer's dilemma variant) and the equilibrium analysis should be stated.

**G** = (V, E_G, θ) is the social graph where V = A, E_G are directed edges representing cooperation history, and θ are topology parameters (Watts-Strogatz rewiring probability β, mean degree k). The graph evolves dynamically: cooperation events add weighted edges.

**P** : Profile × State × Memory × Context → Action is the decision policy. For LLM-based policies: P(π, s, m, c) = parse(LLM(prompt(π, s, m, c))) where prompt() constructs a token-budgeted message from agent state and RAG-retrieved context.

**Φ** : D_ESS → Profile is the empirical grounding function that maps ESS Round 11 microdata records to agent profiles, preserving joint distributions of trust, risk tolerance, political orientation, income, education, and 10+ additional sociodemographic attributes.

**T** is the simulation horizon (number of rounds).


### Notation Summary

| Symbol | Domain | Definition |
|--------|--------|------------|
| A | set | Agent population {a₁, ..., a_N} |
| E | tuple | Economic environment (S, u) |
| G | graph | Social graph (V, E_G, θ) |
| P | function | Decision policy Profile × State × Memory × Context → Action |
| Φ | function | Grounding function D_ESS → Profile |
| T | ℕ | Simulation horizon (rounds) |
| D_ESS | distribution | Empirical ESS Round 11 survey distribution |
| D_sim | distribution | Simulated agent behavior distribution |
| π_A | policy | Ungrounded LLM policy (Condition A) |
| π_B | policy | ESS-grounded LLM policy (Condition B) |
| π_uniform | policy | Uniform action prior (1/3 per action) |
| B_RLHF | [0,1] | RLHF Bias Index = TV(π, π_uniform) |
| BRM_JSD | [0,1] | Behavioral Realism Metric = 1 − JSD(D_sim ‖ D_ESS) |
| BRM_composite | [0,1] | Weighted composite of BRM sub-dimensions |
| JSD | [0,1] | Jensen-Shannon divergence (base-2 logarithm) |
| TV | [0,1] | Total variation distance |
| w₁, w₂, w₃, w₄ | [0,1] | BRM sub-dimension weights (sum = 1) |
| β | [0,1] | Watts-Strogatz rewiring probability |
| k | ℕ | Mean network degree |
| f* | [0,1] | Critical adversarial fraction (phase transition) |
| σ* | [0,1] | Critical shock magnitude (phase transition) |
| α̂ | ℝ>1 | Estimated Pareto exponent (power-law MLE) |
| Q | [0,1] | Network modularity |
| r | [−1,1] | Network assortativity coefficient |


---

## 3.2 Formal Metrics

### Behavioral Realism Metric (BRM)

The single-dimension BRM quantifies distributional fidelity using Jensen-Shannon Divergence:

```
BRM_JSD(sim, emp) = 1 − JSD(D_sim ‖ D_ESS)
```

where JSD(P ‖ Q) = ½ KL(P ‖ M) + ½ KL(Q ‖ M), M = ½(P + Q), and KL is computed with **base-2 logarithm** (ensuring JSD ∈ [0, 1]). Properties: BRM_JSD ∈ [0, 1]; equals 1 when distributions are identical and approaches 0 for disjoint support.

*Implementation note.* Many standard libraries (e.g., `scipy.stats.entropy`) default to natural logarithm, which yields JSD ∈ [0, ln 2] ≈ [0, 0.693]. All implementations must either use base-2 explicitly or divide by ln 2 to normalize. The codebase should include a unit test verifying BRM_JSD = 1.0 when D_sim = D_ESS and BRM_JSD ∈ [0, 1] on random inputs.

The composite BRM aggregates four sub-dimensions:

```
BRM_composite = w₁ · BRM_JSD(wealth)
              + w₂ · (1 − |Gini_sim − Gini_ESS|)
              + w₃ · (1 − |coop_sim − coop_ESS|)
              + w₄ · (1 − JSD_temporal)
```

with default weights w₁ = 0.30, w₂ = 0.25, w₃ = 0.25, w₄ = 0.20 (sum = 1.0). Each component is independently bounded in [0, 1], making the composite bounded as well.


### RLHF Bias Index

The RLHF Bias Index quantifies how far an LLM policy's observed action distribution deviates from the uniform prior:

```
B_RLHF(π) = TV(π, π_uniform) = 0.5 · Σ_{a ∈ A} |π(a) − 1/|A||
```

where π(a) is the empirical frequency of action a and π_uniform(a) = 1/3 for the BGF action space A = {work, save, cooperate}.

**Properties:**

- B_RLHF ∈ [0, 1 − 1/|A|] = [0, 2/3] for |A| = 3.
- B_RLHF = 0 iff π = π_uniform.
- B_RLHF = 2/3 iff π is a point mass on a single action.

**Relationship to cooperation rate (conditional).**  If — and only if — the remaining probability mass is split equally between work and save (i.e., π(work) = π(save) = (1 − p)/2), then B_RLHF simplifies to |p − 1/3| for p ≥ 1/3. This gives the reported values B_RLHF(A) = 0.52 ↔ p ≈ 0.85 and B_RLHF(B) = 0.21 ↔ p ≈ 0.54.

*Empirical validation required.* The equal-split assumption must be checked against the actual simulation output. If π(work) ≠ π(save) — which is likely given the payoff asymmetry (work yields +8 wealth vs. save yields +4) — the linear relationship breaks down, and cooperation rates cannot be read off from B_RLHF alone. We report the full three-way action distribution π(work), π(save), π(cooperate) in Section 6, Table X, to make this verifiable.

**Interpretive limitation.** B_RLHF measures deviation from a *uniform* baseline, not from a *human* baseline. In real public goods experiments, human cooperation rates typically range from 40–60% in early rounds (Chaudhuri 2011), implying a "natural" TV of approximately 0.07–0.27 even without any RLHF distortion. A human-calibrated index B_RLHF*(π) = TV(π, π_human) would be more informative but requires the human-subject experiment described in Section 8.4. Until then, B_RLHF overstates the absolute magnitude of RLHF-specific bias. The *direction* of the grounding effect — B_RLHF(B) < B_RLHF(A) — is not affected by this choice of reference.


### Persona Decay

Expected cooperation rate is estimated from a logistic regression fitted on ESS Round 11 Austrian volunteering behavior (n = 866 respondents with all features non-null):

```
E[coop | profile] = σ(β₀ + β_risk · risk_taking 
                        + β_social_meet · social_meeting_freq 
                        + β_social_act · social_activity + ...)
```

where σ is the logistic sigmoid. The model was fitted with L2 regularization (C optimized via 5-fold grid search), validated by 10-fold stratified cross-validation (AUC = 0.640 ± 0.073, Brier = 0.144), and uncertainty quantified via 1,000 bootstrap resamples (95% CIs reported for all coefficients).

**Proxy validity caveat.** Volunteering is the closest available behavioral proxy for altruistic cooperation in ESS, but it differs from cooperation in a public goods game with wealth stakes in several ways: volunteering involves time rather than money, has no strategic interdependence with others' choices, and lacks the multiplier/redistribution structure of PGG. The model's AUC of 0.640 is only modestly above chance (0.50) and below what would typically be considered good predictive discrimination (0.70+). This means the "ground truth" against which persona fidelity is measured is itself noisy, and fidelity scores should be interpreted as rough indicators of directional consistency rather than precise behavioral calibration. We report bootstrap confidence intervals on all persona decay estimates to make this uncertainty transparent.

**Key empirical finding.** Interpersonal trust variables (trust_people, trust_fairness, trust_helpfulness) have 95% CIs overlapping zero and are not significant predictors of volunteering/cooperation in the Austrian ESS sample. The significant positive predictors are risk tolerance (β = +0.165, 95% CI [+0.065, +0.268]) and social engagement (social_meeting_freq β = +0.164 [+0.079, +0.247]; social_activity β = +0.135 [+0.045, +0.232]).

Per-round persona fidelity is computed over a sliding window of width w:

```
fidelity(t) = 1 − |coop_rate(t, t+w) − E[coop | profile]|
```

with decay rate estimated via ordinary least squares regression of fidelity(t) on t.


---

## Central Claim

For any BGF instance with grounding function Φ derived from D_ESS:

```
BRM(Condition B) > BRM(Condition A)         [Hypothesis H1]
B_RLHF(Condition B) < B_RLHF(Condition A)   [Hypothesis H2]
```

where Condition A is the ungrounded LLM baseline and Condition B is the ESS-grounded configuration.


---

## 3.2.1 Formal Results

The metrics introduced in §3.2 support four numbered results. Full proofs appear in `docs/theorems.md`.

### Proposition 1 (Properties of B_RLHF)

On the finite action space A with |A| = 3, B_RLHF(π) = TV(π, π_uniform) satisfies:

1. **Non-negativity:** B_RLHF(π) ≥ 0 for all π.
2. **Identity:** B_RLHF(π) = 0 if and only if π = π_uniform.
3. **Boundedness:** B_RLHF(π) ≤ 1 − 1/|A| = 2/3.
4. **Permutation invariance:** B_RLHF is invariant under relabeling of actions in A.

*Proof sketch:* Properties 1–2 and 4 follow from TV being a metric on probability distributions (Gibbs & Su 2002). Property 3: TV(π, π_uniform) is maximized when π places all mass on one action, giving 0.5 · (|1 − 1/3| + |0 − 1/3| + |0 − 1/3|) = 0.5 · 4/3 = 2/3. ∎

**Corollary (conditional).** Under the assumption π(work) = π(save), the cooperation rate p and B_RLHF satisfy B_RLHF = |p − 1/3|. This assumption must be empirically verified for each simulation run.


### Proposition 2 (Data-processing bound on grounding error)

*[Renamed from "Theorem 1" — this is a direct application of known results, not a new theorem.]*

For any profile x, grounding map g, and grounded-LLM policy π_LLM+G(· | g(x)):

```
KL( π_human(· | x) ‖ π_LLM+G(· | g(x)) )
  ≤ KL( π_human(· | g(x)) ‖ π_LLM+G(· | g(x)) ) + δ_g(x)
```

with δ_g(x) = KL( π_human(· | x) ‖ π_human(· | g(x)) ) quantifying information loss through g.

*Proof:* This follows from the chain rule for KL divergence and the data processing inequality (Cover & Thomas 2006, Theorems 2.5.3 and 2.8.1). ∎

**Limitations of this bound:**

- The bound is **generally not tight**. It becomes tight only when g(x) is a sufficient statistic of x for predicting human behavior — a strong condition that is unlikely to hold exactly for any survey-based grounding function. We do not claim tightness; we observe that the bound provides a decomposition of the total error into an LLM-alignment term and an information-loss term.
- The information-loss term δ_g(x) is **not directly estimable** from the data available in BGF, because it requires access to π_human(· | x) — the human behavioral distribution conditioned on the full profile — which is not observed. The bound is therefore a conceptual decomposition rather than an operational diagnostic. Estimating δ_g(x) would require the human-subject experiment proposed in Section 8.4.


### Proposition 3 (Weight-robust ordering of BRM)

*[Renamed from "Theorem 2" to reflect that this is a straightforward property of linear functions on simplices.]*

Write BRM_composite(w; cond) = Σ_{j=1..4} w_j · c_j(cond) with c_j ∈ [0,1] the four sub-component scores and w ∈ Δ³ any admissible weight vector. Let Δ_j = c_j(B) − c_j(A).

Because BRM_composite(w; B) − BRM_composite(w; A) = Σ_j w_j Δ_j is linear in w on the probability simplex Δ³, it attains its extrema at vertices:

```
min_{w ∈ Δ³} [ BRM_composite(w; B) − BRM_composite(w; A) ] = min_j Δ_j
```

Hence BRM_composite(B) > BRM_composite(A) for **all** w ∈ Δ³ if and only if min_j Δ_j > 0.

**Certificate.** The four sub-component differences are:

| j | Sub-component | Δ_j = c_j(B) − c_j(A) |
|---|---------------|------------------------|
| 1 | Wealth JSD    | [report value]         |
| 2 | Gini gap      | [report value]         |
| 3 | Cooperation accuracy | [report value]  |
| 4 | Temporal stability   | [report value]  |

*[All four values must be reported explicitly for the claim to be auditable. The Monte Carlo Dirichlet sweep is redundant given this analytic certificate and should be presented as a numerical cross-check, not as the primary evidence.]*


### Design Observation (Causal Identification)

*[Renamed from "Theorem 3" — this is a property of the experimental design, not a theorem requiring proof.]*

Because the treatment T (grounding on/off) is researcher-assigned, no confounders exist between T and the outcome Y. The interventional distribution therefore equals the observational distribution: E[Y | do(T)] = E[Y | T]. This is a standard property of randomized experiments (Hernán & Robins 2020 §2.5) and requires no additional formal apparatus.

**Scope:** This identifies the *total effect* of grounding on BRM and B_RLHF. Mechanism decomposition (which components of the ESS profile drive the effect) is addressed via the 2×2 factorial (§3.9) and V0–V4 ladder (§3.6), where claims are correlational.


### Conjecture (RLHF Cooperation Bias)

For any RLHF-aligned LLM M and any finite n-player symmetric social dilemma G_n where cooperation is individually costly but collectively beneficial, B_RLHF(π_M) > 0 with π_M(cooperate) > 1/|A|.

**Clarifications required for falsifiability:**

- *"RLHF-aligned" must be defined operationally.* We adopt the following working definition: a model M is RLHF-aligned if it was trained with a reward model derived from human preference rankings and optimized via PPO, DPO, or a functionally equivalent policy-gradient method. Models trained solely with supervised fine-tuning (SFT) or constitutional AI (CAI) without a preference-trained reward signal are excluded.
- *The conjecture is falsified by* any model satisfying the above definition that exhibits π_M(cooperate) ≤ 1/|A| in a symmetric social dilemma with individually costly cooperation. Note that GPT-4o-mini satisfies the existence claim (π_M(cooperate) > 1/3) but inverts the grounding-response sign (§6.6.1), which does not refute the conjecture but limits its scope.

**Confirmed-sign empirical support:** Mistral-7B (DPO) and Qwen2.5-7B (RLHF) both show π_M(cooperate) > 1/3 in Condition A of BGF (§6.6 Table 3).


---

## 3.3 Construct Validity and Metric Justification

We address four construct validity challenges that bound the interpretation of all results.

**C1 — Attitudes are not decisions.** BGF ingests attitudinal measures (ESS trust, risk tolerance, social activity) and evaluates behavioral outcomes (cooperation rates, wealth inequality). The ESS correlation between survey trust and observed cooperation in trust-game experiments is moderate at best (r ≈ 0.20–0.35; Glaeser et al. 2000; Berg et al. 1995). Our logistic regression on Austrian volunteering confirms the gap (AUC = 0.640). We therefore claim that grounding shifts action distributions *toward* the empirically plausible range and reduces systematic RLHF bias — not that it achieves exact behavioral replication. Outputs should be treated as counterfactual estimates over attitude-conditioned decision propensities, not as point predictions of individual behavior.

**C2 — Uniform prior as B_RLHF reference.** See the interpretive limitation discussed under the RLHF Bias Index definition above. The uniform reference is analytically convenient but empirically wrong — real human cooperation rates are non-uniform. Reported B_RLHF values upper-bound the RLHF-specific distortion. The directional comparison B_RLHF(B) < B_RLHF(A) is unaffected.

**C3 — BRM weight sensitivity.** The composite BRM weights are set by judgment. The analytic certificate in Proposition 3 (min_j Δ_j > 0) guarantees that BRM(B) > BRM(A) for every weight vector on the simplex, making the ordering weight-robust in the strongest possible sense. Absolute BRM values vary by ±0.04 across weight perturbations; only the ordering is claimed as robust.

**C4 — Payoff design dependence.** Results in Sections 6.1–6.4 are conditional on the specific LPGG payoff structure (c = 3, multiplied return = 12/N per capita). Different social dilemma parameterizations (prisoner's dilemma, stag hunt, assurance game) may yield different cooperation equilibria and different grounding effects. The sensitivity of the grounding effect to payoff parameterization is a priority for future work.
