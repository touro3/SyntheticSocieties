# BGF Causal Model

> **Evidence-status convention.** Each empirical claim is annotated `[audit: X.Y]` referencing a row in `docs/evidence_audit.md`. Derivations carry `[📐]`; pending experiments carry `[⏳]`.

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
| ESS → Persona → Cooperation | Trust/risk attributes in persona condition LLM to cooperate at ESS-calibrated rates | Ablation: no_persona vs rich_persona `[audit: C.4 ⚠️ — `metrics/mediation.py` + `tests/test_mediation.py` ✓; aggregation pending]` |
| ESS → SQL RAG → Cooperation | Population norms ("peers have avg trust 6.2/10") anchor LLM toward group behavior | Ablation: with vs without population_context `[audit: F.3 ✅ persona × RAG factorial; aggregation pending]` |
| ESS → Graph RAG → Cooperation | Social position context ("you are a bridge node") provides strategic information | Ablation: with vs without social_context `[audit: F.4 ⏳ single-vs-dual RAG ablation pending]` |
| Prompt Length → Cooperation | Longer prompts may change LLM behavior regardless of content (confound) | **Length-controlled ablation** (padded prompt) `[audit: C.5 ⏳ infra ready, GPU run pending]` |

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

This follows a standard 2x2 factorial design `[audit: C.4 ⚠️ code + tests in `metrics/mediation.py`, `tests/test_mediation.py`; primary-scale aggregation to `analysis/tables/mediation.json` pending — see §H.3]`:

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
- This gives us the sharp identification result: **E[Y | do(T=1)] = E[Y | T=1]** — the observational distribution equals the interventional distribution. `[audit: C.1 📐 Pearl identification by construction (exogeneity)]`

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

For the primary result (cooperation rate ratio B/A ≈ 1.35, estimated from pilot data `[audit: C.2 ✅ closed-form from A.10/A.11]`):
```
E = 1.35 + sqrt(1.35 × 0.35) = 1.35 + 0.69 ≈ 2.04
```

**Interpretation**: An unmeasured confounder would need to be associated with *both* the grounding treatment and the cooperation outcome by a factor of at least **2.04** on the risk-ratio scale to fully explain the observed effect. Given that all design parameters (model, seed, temperature, network topology) are held fixed across conditions, no plausible confounder meets this threshold. The E-value therefore provides quantitative support for the causal interpretation.

For the effect on Gini coefficient (Gini ratio A/B ≈ 2.1, grounded agents show lower inequality):
```
E = 2.1 + sqrt(2.1 × 1.1) ≈ 2.1 + 1.52 ≈ 3.62
```
This larger E-value reflects the stronger Gini finding and indicates even greater robustness to unmeasured confounding. `[audit: C.3 ✅ closed-form derivation; ratios from A.10/A.11]`

---

## 8. Negative-Control Program

The factorial design in §4 establishes *that* persona and RAG both contribute; the negative controls below establish *what* about persona and RAG carries the effect. Each control isolates a specific alternative explanation for the B-vs-A gap.

### 8.1 Three Sham-Grounding Conditions

| Condition | What it preserves from Φ | What it breaks | Alternative hypothesis it tests | Implementation |
|---|---|---|---|---|
| **P — Padded** (`decision/padded_ablation_policy.py`, infra ready) `[audit: C.5 ⏳]` | Token count, structural slots | Semantic content (filler text) | "The effect is prompt length" | Token-matched no-content padding (GPU runs pending) |
| **S — Scrambled-ESS** (~80 LOC on top of `sql_rag.py`) `[audit: C.6 ⏳ code not yet written]` | Token count, vocabulary, structural slots, surface form of ESS facts | The Φ mapping itself: rows permuted across demographic keys so persona/peers are *real ESS rows but assigned to the wrong agent* | "The effect is vocabulary/form, not the Φ mapping" | New `decision/scrambled_rag_policy.py`; outstanding |
| **F — Fabricated** (~80 LOC) `[audit: C.7 ⏳ code not yet written]` | Token count, vocabulary, structural slots, plausibility | Empirical grounding: facts are GPT-generated plausible demographics with no link to ESS | "Any plausible demographic prompt suffices; ESS-specific calibration is unnecessary" | Outstanding; closes the "is it ESS or just *some* demographic prompt" question |

These three conditions sit between the unrelated extremes of A (no grounding at all) and B (full ESS grounding), each holding one dimension constant while varying another. They are designed so that the *predicted ordering* of the dependent variable differentiates the BGF theory from each alternative — see the sensitivity table in §10.

### 8.2 Why Three, Not One

A single padded control answers only "is it length?" The scrambled and fabricated controls answer the substantially harder question: *given that length is controlled, is the empirical Φ mapping itself necessary, or would any plausible demographic prompt do?* This is the question that converts "we have a working architecture" into "the ESS data is load-bearing."

---

## 9. Do-Calculus Walkthrough

The §6 backdoor argument asserts identification under researcher-assigned treatment. This section makes the identification *explicit* by walking through the do-operator on the DAG in §1.

### 9.1 Variables

Let:
- `T ∈ {0, 1}`: treatment indicator (ESS grounding off/on).
- `Y`: outcome (cooperation rate or final-round Gini).
- `M_p`: persona-channel mediator (text injected via `population/persona_synthesizer.py`).
- `M_r`: RAG-channel mediator (text injected via `sql_rag.py` + `graph_rag.py`).
- `L`: prompt length in tokens (a deterministic function of `M_p, M_r`, plus a fixed scaffold).
- `θ`: LLM weights and decoding hyperparameters (model, temperature, seed) — fixed across conditions.
- `D`: empirical ESS data (an exogenous input, parent of `M_p` and `M_r`).

### 9.2 The Backdoor Set

Paths from `T` to `Y` in the DAG (§1):

```
T → M_p → prompt → Y         (causal, persona channel)
T → M_r → prompt → Y         (causal, RAG channel)
T → L → prompt → Y           (causal, length channel — to be ruled out)
```

There is no path from `T` to `Y` that enters `T` "through the back door" — no variable in the DAG is a parent of `T` other than the researcher's choice. The empty set ∅ therefore satisfies the backdoor criterion, and:

```
P(Y | do(T = t)) = P(Y | T = t)
```

### 9.3 What Remains to Identify: the Mechanism

The above identifies the *total* effect. The *mechanism* — whether the effect flows through `M_p`, `M_r`, or `L` — is identified by the front-door style intervention realized in the negative controls:

```
P(Y | do(M_p = m_p), do(M_r = m_r), do(L = ℓ)) = P(Y | M_p = m_p, M_r = m_r, L = ℓ)
```

The factorial ablation (§4) sets `do(M_p)` and `do(M_r)` directly; the padded control (Condition P) sets `do(L = ℓ_B)` with `M_p = M_r = ∅`; the scrambled-ESS control (Condition S) sets `do(M_p, M_r)` to syntactically-valid-but-semantically-broken values; the fabricated control (Condition F) sets `do(M_p, M_r)` to plausible-but-non-empirical values. Each `do(·)` is implemented by direct prompt construction, satisfying the interventionist semantics literally.

### 9.4 What Remains Unidentified

The internal LLM mapping `prompt → Y` is treated as a black box. We cannot decompose the prompt-to-action distribution into sub-mechanisms inside the model. This is the residual identification limit acknowledged in §5.3 (LLM internals opacity) and §6 (factorial provides mechanism evidence, not formal identification of internal pathways).

---

## 10. Sensitivity Table — Theory Adjudication via Predicted Orderings

The negative-control program above lets the data adjudicate between BGF and three named alternatives. The table below records the predicted ordering of `B_RLHF` (lower is better, closer to human distribution) under each theory. The empirical ordering of the five conditions, once measured, picks out at most one of the four columns as consistent.

| Condition | BGF (ESS is load-bearing) | H_length (only length matters) | H_form (any plausible demographic prompt suffices) | H_Hawthorne (any non-empty grounding suffices) |
|---|---|---|---|---|
| **A** (no grounding) | highest B_RLHF | highest | highest | highest |
| **P** (padded, no content) | ≈ A | **≈ B** ⚠ | ≈ A | ≈ A |
| **S** (scrambled ESS) | ≈ A or ≈ P | ≈ B | ≈ B | ≈ B |
| **F** (fabricated demographics) | ≈ A or ≈ P | ≈ B | ≈ B | ≈ B |
| **B** (full ESS grounding) | lowest | ≈ P | ≈ S ≈ F | ≈ S ≈ F |

⚠ The H_length theory makes the strong testable prediction that P ≈ B; the BGF theory predicts P ≈ A. This single comparison (Condition P vs. B at primary scale) is the most consequential outstanding experiment in the program.

**Adjudication rule (pre-registered):** The BGF theory is *corroborated* iff the empirical ordering across {A, P, S, F, B} matches `BGF column` within Spearman ρ ≥ 0.9 across the four metrics (BRM_composite, B_RLHF, Gini, modularity). Any alternative-theory column matching with Spearman ρ ≥ 0.9 across the same metrics constitutes evidence for that alternative and a falsification of the BGF causal claim. `[audit: C.8 ⏳ data not yet collected; depends on C.5–C.7]`

### 10.1 Triangulation with E-Values

The E-values in §7 quantify resilience to a *generic* unmeasured confounder. The sensitivity table above tests resilience to *named* alternatives. Both are needed: the E-value handles unknown unknowns; the negative-control table handles known unknowns. Together they exhaust the threats a defense committee can reasonably raise.

*This sensitivity program is falsified if the empirical ordering of {A, P, S, F, B} matches any alternative column more closely than the BGF column.*
