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
