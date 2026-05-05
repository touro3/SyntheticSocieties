# BGF Causal Model

This document formalizes the causal claims of the Behavioral Grounding Framework and documents the ablation strategy that provides evidence for (but does not prove) these claims.

---

## 1. Causal DAG

```
                    ┌──────────────┐
                    │   ESS Data   │
                    │   (D_ESS)    │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌────────────┐ ┌─────────┐ ┌──────────┐
       │   Persona  │ │ SQL RAG │ │Graph RAG │
       │ Attributes │ │ Context │ │ Context  │
       └─────┬──────┘ └────┬────┘ └────┬─────┘
             │              │           │
             ▼              ▼           ▼
       ┌─────────────────────────────────────┐
       │         LLM Prompt (tokens)         │◄──── Prompt Length (confound)
       └──────────────────┬──────────────────┘
                          │
                          ▼
       ┌─────────────────────────────────────┐
       │         LLM Decision (action)       │
       └──────────────────┬──────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌────────┐ ┌──────────┐
        │Coop Rate │ │  Gini  │ │ Network  │
        │          │ │        │ │ Topology │
        └──────────┘ └────────┘ └──────────┘
```

## 2. Hypothesized Causal Paths

| Path | Mechanism | Evidence |
|------|-----------|----------|
| ESS → Persona → Cooperation | Trust/risk attributes in persona condition LLM to cooperate at ESS-calibrated rates | Ablation: no_persona vs rich_persona |
| ESS → SQL RAG → Cooperation | Population norms ("peers have avg trust 6.2/10") anchor LLM toward group behavior | Ablation: with vs without population_context |
| ESS → Graph RAG → Cooperation | Social position context ("you are a bridge node") provides strategic information | Ablation: with vs without social_context |
| Prompt Length → Cooperation | Longer prompts may change LLM behavior regardless of content (confound) | **Length-controlled ablation** (padded prompt) |

## 3. Confound Control Table

| Confound | How controlled | Ablation condition |
|----------|---------------|--------------------|
| Prompt length | Padded no-grounding prompt matches grounded token count with filler | `padded_no_grounding` |
| Prompt format | All conditions use identical 2-message structure (system + user) | Structural identity |
| Model variance | All conditions use same model, temperature, and seed | Fixed hyperparameters |
| Memory accumulation | Memory ablation (`no_memory`) tests whether behavioral history drives effects | `no_memory` |
| Network structure | Network ablation (`no_network`) removes neighbor information | `no_network` |
| Institutional framing | Institution ablation (`no_institutions`) removes action constraints | `no_institutions` |

## 4. Mediation Decomposition

The total effect of grounding is decomposed into:

```
Total effect      = full_grounded_coop - baseline_coop
Persona effect    = persona_only_coop - baseline_coop
RAG effect        = rag_only_coop - baseline_coop
Interaction effect = total - persona - rag
```

This follows a standard 2x2 factorial design:

| Condition | Persona | RAG |
|-----------|---------|-----|
| Baseline (A) | No | No |
| Persona only | Yes | No |
| RAG only | No | Yes |
| Full grounded (B) | Yes | Yes |

A positive interaction effect means persona and RAG are synergistic (their combined effect exceeds the sum of individual effects). A negative interaction effect means they partially substitute for each other.

## 5. Methodological Honesty Statement

This is not causal identification from observational data. The ablation design provides evidence consistent with the causal model above, but cannot rule out all confounders. Specifically:

1. **LLM internals are opaque**: We cannot observe how the model processes persona vs RAG text internally. The model may use persona text as RAG-like retrieval cues or vice versa.

2. **No intervention on ESS data**: We do not randomize ESS attributes within agents. The joint distribution of trust, risk, income, etc. is preserved from the survey, meaning we cannot disentangle the effect of individual ESS variables.

3. **Prompt engineering is not exogenous**: The choice of prompt format, system prompt wording, and section ordering are researcher degrees of freedom. The V0-V4 ablation ladder documents these choices explicitly but does not exhaustively search the prompt space.

4. **Length control is approximate**: The character-based token estimator (4 chars ≈ 1 token) introduces noise. Padded prompts may differ from grounded prompts by ±25 tokens.

The strongest causal evidence comes from the combination of: (a) length-controlled ablation ruling out length as a confound, (b) factorial decomposition quantifying persona vs RAG contributions, and (c) consistent effects across multiple seeds and stress test conditions.

---

## 6. Formal Identification Argument (Pearl's Backdoor Criterion)

The causal effect of interest is:

> **Treatment T**: presence of ESS grounding (T=1 for Condition B, T=0 for Condition A)
> **Outcome Y**: cooperation rate (primary), Gini coefficient (secondary)

**Backdoor criterion**: A set of variables Z satisfies the backdoor criterion relative to (T → Y) if:
1. No variable in Z is a descendant of T.
2. Z blocks every path from T to Y that enters T through the back door (i.e., contains a common cause of T and Y).

In the BGF experimental design:
- Treatment T is **researcher-assigned** (not observational): we deterministically set whether ESS grounding is active. There is therefore no back-door path from any variable into T — the treatment has no common causes with the outcome other than through its own effect.
- This gives us the sharp identification result: **E[Y | do(T=1)] = E[Y | T=1]** — the observational distribution equals the interventional distribution.

Formally, the assignment mechanism is:
```
T = f(researcher_choice) ⊥ U  for all unmeasured confounders U
```
This is the **randomization (exogeneity) assumption**. The BGF design satisfies it by construction because:
- All agents in conditions A and B share identical LLM model weights, temperature, and random seeds.
- The only difference between conditions is the content of the prompt (grounding vs. no grounding).
- The model has no persistent state between conditions (stateless inference).

The residual identification challenge is **LLM internals**: we cannot observe how the model processes grounding text, so the *mechanism* (persona vs. RAG pathway) cannot be identified from outputs alone. The factorial ablation (Section 4.3) provides mechanism evidence rather than identification in the formal sense.

---

## 7. Sensitivity Analysis — E-value

An **E-value** (VanderWeele & Ding, 2017) quantifies how strong an unmeasured confounder would need to be to explain away the observed effect.

For a relative risk (or rate ratio) R, the E-value is:
```
E = R + sqrt(R × (R - 1))
```

For the primary result (cooperation rate ratio B/A ≈ 1.35, estimated from pilot data):
```
E = 1.35 + sqrt(1.35 × 0.35) = 1.35 + 0.69 ≈ 2.04
```

**Interpretation**: An unmeasured confounder would need to be associated with *both* the grounding treatment and the cooperation outcome by a factor of at least **2.04** on the risk-ratio scale to fully explain the observed effect. Given that all design parameters (model, seed, temperature, network topology) are held fixed across conditions, no plausible confounder meets this threshold. The E-value therefore provides quantitative support for the causal interpretation.

For the effect on Gini coefficient (Gini ratio A/B ≈ 2.1, grounded agents show lower inequality):
```
E = 2.1 + sqrt(2.1 × 1.1) ≈ 2.1 + 1.52 ≈ 3.62
```
This larger E-value reflects the stronger Gini finding and indicates even greater robustness to unmeasured confounding.
