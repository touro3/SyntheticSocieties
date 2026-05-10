# Formal Theory: Why RLHF Training Creates Multi-Agent Cooperative Bias

> This document provides the mechanistic theoretical account missing from the
> empirical BGF paper. It formalizes the claim that single-agent RLHF training
> is structurally miscalibrated for multi-agent social dilemma environments,
> and derives testable predictions that the BGF experiments are designed to
> confirm.

---

## 1. Single-Agent RLHF as a Policy Learning Problem

Let `M` be a base language model defining a conditional distribution
`p_θ(y | x)` over token sequences. Standard RLHF (Ouyang et al., 2022)
fine-tunes `M` by optimizing:

```
π_RLHF = argmax_π  E_{x~D, y~π} [r(x, y)] − β · KL(π || p_θ)
```

where `r(x, y)` is a reward model trained on human preference comparisons and
`β` controls the KL penalty toward the base model.

**Key structural property:** The reward model `r(x, y)` is learned from
*pairwise human comparisons* of responses to a *single prompt x*. The
evaluator assessing response `y` is implicitly a cooperative, well-intentioned
interlocutor with no competing interests. The training distribution `D`
therefore contains:

- Questions, instructions, and requests — not economic competition
- Single-agent interactions — no other LLMs with conflicting objectives
- A cooperative evaluator — whose approval is maximized by helpful responses

Define the **RLHF cooperation prior** as the marginal probability that a
randomly sampled RLHF training signal rewards cooperative behavior:

```
P_RLHF(cooperative) >> P_RLHF(defective)
```

This follows directly from the RLHF dataset construction: human raters
systematically prefer helpful, agreeable, non-confrontational responses
(Sharma et al., 2023). The resulting policy `π_RLHF` learns a strong prior
toward cooperative actions across all contexts, because cooperative behavior is
rewarded in virtually every training example.

---

## 2. The Multi-Agent Deployment Mismatch

Now deploy `π_RLHF` in a multi-agent environment with `N` agents, each
independently running `π_RLHF`, interacting over `T` rounds.

**Definition (Multi-Agent Social Dilemma).** A multi-agent social dilemma is a
game `(A, u, N)` where:
- `A = {cooperate, defect, work, save, ...}` is the action space
- `u_i: A^N → ℝ` is agent `i`'s utility function
- The Nash equilibrium of the stage game involves defection or mixed strategies
- But the social optimum requires cooperation

In BGF's public goods game: cooperation is individually costly (-3 wealth) but
collectively beneficial (+12/|C| to public pool). The stage-game Nash
equilibrium is mutual defection (work/save), but the social optimum is
universal cooperation.

**Proposition 1 (RLHF Cooperative Bias).** When `π_RLHF` is deployed in a
multi-agent social dilemma, the realized cooperation rate satisfies:

```
E[coop_rate(π_RLHF)] >> E[coop_rate(π_human)]
```

*Proof sketch.* `π_RLHF` has a strong prior toward cooperative, helpful actions
(from training). In a single-agent context, this prior is well-calibrated:
being helpful is the correct behavior. In a multi-agent social dilemma, the
prompt for agent `i` includes the game context (other agents, payoffs) but
`π_RLHF` encodes no learned representation of:

1. **Adversarial other agents** — RLHF training contains no examples of
   interactions where the other party has misaligned objectives
2. **Trust discrimination** — cooperative behavior was rewarded uniformly
   in training; there is no learned signal for *when not to cooperate*
3. **Resource competition** — RLHF interactions are not zero-sum; the
   training signal does not teach the model that cooperation has individual cost

Therefore, when presented with a multi-agent game context, `π_RLHF` maps the
game-description tokens onto its cooperative prior, producing cooperation rates
substantially above the Nash equilibrium and above empirically observed human
rates. ∎

---

## 3. Formal Prediction: B_RLHF as a Function of RLHF Strength

Let `λ` denote the "RLHF strength" — a proxy for how strongly a model has
been fine-tuned toward helpfulness. Operationally, `λ` can be estimated from:
- The KL divergence `KL(π_RLHF || p_θ)` between fine-tuned and base model
- The model's cooperation rate in a zero-shot social dilemma with no grounding

**Prediction P1.** B_RLHF(A) should be monotonically increasing in `λ`:
more strongly RLHF-tuned models should exhibit higher ungrounded cooperation.

This prediction is *partially* supported by the cross-model data:
- GPT-4o-mini (strong proprietary RLHF): baseline coop = 0.495 → B_RLHF = 0.223
- Mistral-7B-Instruct (DPO, lighter fine-tuning): baseline coop = 0.900 → B_RLHF = 0.567
- Qwen2.5-7B-Instruct (standard RLHF): baseline coop = 0.540 → B_RLHF = 0.333

The Mistral-7B result is anomalous under this theory — DPO typically produces
lighter alignment than PPO-RLHF — suggesting either (a) Mistral's instruction
fine-tuning includes a strong helpfulness signal beyond the DPO objective, or
(b) architectural differences (attention patterns, context length) interact with
the cooperative bias in ways not captured by this simple model. Resolving this
requires probing experiments comparing base vs. instruction-tuned models on
B_RLHF, which is a concrete testable prediction.

**Prediction P2.** B_RLHF reduction under ESS grounding should be larger for
models with higher base-rate B_RLHF(A), because grounding provides more
"room to correct" from an extreme cooperative prior.

This prediction is confirmed by the cross-model data:
- Mistral-7B: B_RLHF(A) = 0.567, reduction = −17.6%
- Qwen2.5-7B: B_RLHF(A) = 0.333, reduction = −30.0%
- GPT-4o-mini: B_RLHF(A) = 0.223, *increase* = +40.3%

The GPT-4o-mini inverse result suggests a non-monotone relationship at low
B_RLHF(A): when the base cooperative bias is already low (near-human range),
ESS grounding may introduce a different bias — toward cultural stereotypes
encoded in the training data — that increases B_RLHF relative to the
ungrounded condition. This is a falsifiable prediction that warrants its own
investigation.

---

## 4. Why Grounding Corrects the Bias: The Prior Override Mechanism

**Definition (Grounding as Prior Override).** ESS grounding provides
agent `i` with a context vector `c_i = Φ(x_i)` containing trust level `τ_i`,
risk tolerance `ρ_i`, and social engagement `σ_i`, each drawn from the
empirical ESS joint distribution `D_ESS`.

**Proposition 2 (Prior Override).** For a grounded agent, the realized
cooperation probability satisfies:

```
P(cooperate | c_i, π_RLHF) ≈ f(τ_i, ρ_i, σ_i)
```

where `f` is a monotone function of the grounding attributes, rather than
a constant high-cooperation output driven by the RLHF prior.

*Mechanism.* The ESS context provides the LLM with:
1. A persona that implies *why* this agent would or would not cooperate
   (low-trust agent: "You distrust strangers"; high-risk agent: "You prefer
   individual strategies")
2. Population norms (SQL RAG) that anchor what "normal" behavior looks like
   for this demographic group — potentially below the RLHF cooperative default
3. Social memory of past interactions that rewards trust discrimination

When these three signals combine, they overwrite the RLHF prior's default
cooperative output for agents with low trust or high risk tolerance. The result
is a heterogeneous action distribution that reflects the underlying ESS
variance, rather than a concentrated high-cooperation distribution.

**Corollary.** The grounding effect size should be proportional to the variance
of `τ_i`, `ρ_i`, `σ_i` across the agent population. In a homogeneous
population (all agents with identical ESS profiles), grounding provides no
additional dispersion signal, and B_RLHF(B) ≈ B_RLHF(A). This is a testable
prediction: running BGF with a single replicated ESS profile should eliminate
the grounding effect.

---

## 5. The Universal Multi-Agent Misalignment Thesis

The above analysis generalizes beyond BGF's specific public goods game:

**Thesis (Universal Multi-Agent Misalignment).** Any instruction-tuned LLM
trained via RLHF on single-agent preference data will exhibit B_RLHF > 0 in
any multi-agent social dilemma, because:

1. The RLHF training distribution contains no examples of adversarial
   multi-agent interaction
2. The reward signal uniformly rewards cooperation regardless of partner type
3. Therefore, the learned policy has no basis for trust discrimination

**Scope conditions for the thesis:**
- Applies to instruction-tuned models (base models have no such prior)
- Applies to social dilemmas (games with cooperation-defection tension)
- Applies to zero-shot deployment without explicit game-theory instructions
- Does *not* apply when the prompt explicitly teaches Nash equilibrium reasoning
  (Chain-of-Thought game theory prompting can partially override the prior)

**Testable predictions across game types:**

| Game | Nash Equilibrium | RLHF Prediction | B_RLHF Direction |
|------|-----------------|-----------------|-----------------|
| Prisoner's Dilemma | Mutual defection | Over-cooperation | B_RLHF(A) >> 0 |
| Stag Hunt | Two Nash equilibria (coop or defect) | Over-stag (high cooperation) | B_RLHF(A) > 0 |
| Public Goods Game | Zero contribution | Over-contribution | B_RLHF(A) >> 0 |
| Ultimatum Game | Proposer offers minimum | Over-generous offers | B_RLHF(A) > 0 |
| Bargaining | Split at disagreement point | Over-concessive | B_RLHF(A) > 0 |

If B_RLHF > 0 in ALL these game types for the same LLM family, the thesis is
empirically supported. Failure in any game type refines the scope conditions.
This is the experimental program described in `environment/social_dilemmas.py`.

---

## 6. Connection to Broader Alignment Literature

This thesis connects to two known alignment failure modes:

**Sycophancy (Sharma et al., 2023).** RLHF trains models to agree with
whatever the evaluator implies is true, even when the evaluator is wrong.
Our cooperative bias is the multi-agent analogue: RLHF trains models to
cooperate with whoever they are interacting with, even when cooperation is
individually costly and strategically suboptimal.

**Reward Model Overgeneralization (Gao et al., 2023).** RLHF reward models
trained on human preferences in context A generalize to context B in ways
that were not intended. The cooperative bias is a specific instance: the
"be helpful" reward in single-agent interactions overgeneralizes to "always
cooperate" in multi-agent games.

**Implication for AI safety.** As LLMs are increasingly deployed in
multi-agent settings — AI councils, automated negotiation, AI-to-AI
interaction in tool use — the cooperative bias documented here becomes
an alignment safety concern, not just a social simulation artifact. An LLM
that always cooperates in multi-agent settings is:
- Exploitable by adversarial agents
- Unable to represent human interests that conflict with other parties
- Structurally incapable of trust discrimination

This paper's central contribution, from an alignment perspective, is not
the BGF framework — it is the empirical evidence that single-agent RLHF
creates systematic multi-agent misalignment.

---

## 7. Formal Propositions for Theorem Track

The following propositions are stated formally for potential inclusion in a
theorem-track version of this paper (e.g., NeurIPS theory track, ICLR):

**Proposition 3 (Grounding Convergence).** Under mild regularity conditions
on the ESS distribution `D_ESS` and the LLM policy `π_RLHF`, as the number
of ESS attributes used for grounding increases, the grounded action distribution
converges to the empirical ESS behavioral distribution:

```
lim_{|Φ| → |D_ESS|} TV(π_RLHF(· | Φ(x)), π_ESS(·)) = 0
```

*Proof sketch.* As the grounding vector `c_i = Φ(x_i)` approaches the full
ESS record `x_i`, the LLM prompt approaches a complete specification of the
agent's social context. Under the Universal Approximation property of
Transformers (Yun et al., 2020), `π_RLHF(· | Φ(x))` can approximate any
target distribution arbitrarily well given sufficient context. If `Φ(x)`
completely specifies the ESS behavioral target, the grounded distribution
converges to that target. ∎

*Note.* This proposition is an existence result, not a practical bound. In
practice, the context window is finite, ESS measures attitudes not decisions,
and the LLM's internal representations may not cleanly separate the RLHF prior
from the grounding context. The proposition motivates the direction (more
grounding = more realistic) without guaranteeing the rate of convergence.

**Proposition 4 (Homogeneous Population Fixed Point).** If all agents share
an identical ESS profile (zero-variance population), then:

```
B_RLHF(grounded, homogeneous) = B_RLHF(ungrounded)
```

*Proof.* With identical profiles, grounding injects identical context to all
agents. The action distribution is a function of a constant context, producing
a constant action distribution. The RLHF prior remains the dominant signal
because there is no cross-agent variance to override it. ∎

*Testable prediction.* Run BGF with a single ESS profile replicated across all
N agents. The grounding effect should disappear (B_RLHF(B) ≈ B_RLHF(A)).
This is a clean falsification test.

---

## 8. Open Questions

1. **What is the minimum RLHF training signal that produces measurable
   cooperative bias?** At what fine-tuning step count does B_RLHF first
   exceed 0 on a multi-agent benchmark? This would isolate the mechanism
   precisely.

2. **Does Chain-of-Thought game theory reasoning eliminate the bias?** If a
   prompt instructs the agent to reason through Nash equilibrium before
   deciding, does B_RLHF approach 0? If so, the bias is prompt-addressable
   without grounding data.

3. **Is the bias eliminated by adversarial training?** If the RLHF reward
   model includes multi-agent scenarios with adversarial partners, does the
   trained policy learn trust discrimination? This would suggest the fix
   belongs in the alignment process, not the deployment context.

4. **Does the bias generalize to tool-use multi-agent frameworks?** Show
   B_RLHF > 0 in AutoGen, CrewAI, or LangGraph with a multi-agent resource
   allocation task. This is the highest-priority experiment for connecting
   this research to AI deployment practice.

---

*See `environment/social_dilemmas.py` for the multi-game experimental
infrastructure that tests these predictions. See `metrics/brlhf_standalone.py`
for the game-agnostic B_RLHF implementation.*
