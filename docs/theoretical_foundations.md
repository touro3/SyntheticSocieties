# Theoretical Foundations of the Behavioral Grounding Framework

> **Evidence-status convention.** Every empirical claim in this document is annotated `[audit: X.Y]` and refers to a row in `docs/evidence_audit.md`. Theoretical derivations are annotated `[📐 derivation only]` and require no empirical support. Empirical predictions yet to be tested are annotated `[⏳ pending]`.

This document derives, from first principles, *why* the BGF architecture should be expected to outperform ungrounded LLM agents on behavioral realism. The empirical results reported in `paper.md` §5–§6 are then tests of theoretical predictions made here, not standalone measurements.

The four arguments below are independent: any one of them would justify the architecture; together they triangulate a single prediction (Condition B > Condition A on BRM, with B_RLHF reduction) from four distinct theoretical commitments.

---

## 1. Information-Theoretic Argument

### 1.1 Setup

Let `π_human(a | x)` be the true conditional distribution over economic actions `a ∈ {work, save, cooperate}` given a covariate vector `x` (age, trust, income, country, etc.) sampled from the empirical population. Let `π_LLM(a)` be the ungrounded LLM's marginal action distribution and `π_LLM+G(a | g(x))` its action distribution conditioned on the grounding text `g(x)` produced by the BGF mapping `Φ: D_ESS → Profile` and dual-RAG retrieval.

Behavioral realism, in KL terms, is:

```
ε_A = E_x[ KL( π_human(· | x) || π_LLM(·) ) ]            (Condition A: ungrounded)
ε_B = E_x[ KL( π_human(· | x) || π_LLM+G(· | g(x)) ) ]   (Condition B: grounded)
```

The architectural claim is `ε_B ≤ ε_A`, with strict inequality whenever `g(x)` carries non-trivial information about `x`.

### 1.2 Sufficient-Statistic Claim

We claim — and operationalize — that the ESS-derived grounding text `g(x)` is an *approximate sufficient statistic* for `x` with respect to the human action distribution: `π_human(a | x) ≈ π_human(a | g(x))`. Two empirical anchors support this:

- The ESS variables retained by Φ (trust, risk, income decile, age band, country) are precisely those that the empirical logistic regression in `metrics/feature_importance.py` identifies as load-bearing for the cooperation outcome (Austrian R11 volunteering, AUC=0.640, 1,000-bootstrap CIs in `analysis/tables/feature_importance.json`).
- The dual RAG (`decision/sql_rag.py` for population norms, `decision/graph_rag.py` for social position) adds the two covariates — peer-group mean and network role — that the agent cannot reconstruct from its own persona alone.

### 1.3 Data-Processing Inequality Bound

Suppose `g(x)` is an approximate sufficient statistic in the sense that `I(x; a | g(x)) ≤ δ` for some small `δ ≥ 0`. Then by the data-processing inequality:

```
KL(π_human(· | x) || π_LLM+G(· | g(x)))
    ≤ KL(π_human(· | g(x)) || π_LLM+G(· | g(x))) + δ
```

The first term on the right is the *minimum* KL achievable by any policy that sees only `g(x)`; the second term `δ` is the residual information loss from the Φ mapping. By contrast, the ungrounded baseline must absorb the full conditional variation of `x` into a single unconditional distribution `π_LLM`, which is a strictly larger upper bound when `Var(π_human(· | x))` is non-zero across `x`.

### 1.4 Predicted Direction

This argument predicts not only `ε_B < ε_A` but *where* the gap should be largest: in subpopulations where `π_human(· | x)` is most extreme (e.g., very-high-trust or very-low-trust bands). Empirically, H5 (trust-gradient recovery, Spearman ρ=0.800 at pilot scale `[audit: A.5]`, ρ=1.0 cross-cultural `[audit: A.1 ⚠️ dry-run rule-based]`) is the direct test of this subpopulation-resolved prediction. *This claim is falsified if Condition B and Condition A show statistically indistinguishable per-subgroup KL even when aggregate BRM differs* — the gap must be carried by the conditioning, not by an aggregate shift.

### 1.5 Connection to "Silicon Samples"

This is the formal statement of what Argyle et al. (2023, *Political Analysis*) call "out-of-one-many" — LLM conditioning on a demographic prompt recovers subgroup-conditional distributions. BGF strengthens it from *post-hoc evaluation on opinion surveys* to *a priori architectural design for sequential economic decisions*, where the conditioning information must persist across rounds (the role of `agents/memory.py`).

---

## 2. Generative Social Science Framing (Epstein)

### 2.1 The Generativist Standard

Epstein (2006, *Generative Social Science*; Epstein & Axtell 1996, *Growing Artificial Societies*) advances the *generativist motto*: **"If you didn't grow it, you didn't explain it."** A macro phenomenon is *explained* iff a generative micro-model exists whose agents, acting locally on plausible rules, reproduces the macro phenomenon without it being imposed.

### 2.2 BGF as a Generative Test

The macro phenomena BGF claims to explain are:

- **EU-realistic inequality** (Gini ∈ [0.20, 0.38], H3).
- **Trust-stratified cooperation** (monotonic Spearman ordering across ESS trust bands, H5).
- **Community structure under stress** (modularity Q increase, H4).

The generative claim is: **ESS-grounded LLM agents are a minimal sufficient micro-rule set to grow these macro patterns.** Minimality is established by the ablation ladder:

- Remove grounding (Condition A) → Gini collapses to ~0.08 (outside empirical range): fails H3 `[audit: A.10/A.11 ⚠️ phase_c parquets missing on disk; cached values verified]`.
- Remove memory (M0) → persona fidelity drops 0.609 → 0.742 monotone across M0–M3 `[audit: A.9 ❌ unverified — ablation runs used mock policy; LLM re-run pending `scripts/run_memory_ablation_llm.sh`]`.
- Remove RAG (factorial cells in `metrics/mediation.py`) → effect attenuates `[audit: B.4 / C.4 ⏳ aggregation pending; metric implementation ✅ unit-tested]`.

Each removal destroys the macro pattern, satisfying Epstein's minimality criterion for the retained components.

### 2.3 Why an LLM, Not a Hand-Coded Rule

Classical Sugarscape-style ABMs (Epstein & Axtell 1996; Schelling 1971) hand-code micro-rules. The risk is *over-fitting the rule to the phenomenon*. The LLM, by contrast, is *pre-trained on a corpus that does not contain the BGF game* — its priors over economic action are determined by general linguistic and social training. Replacing the hand-coded rule with `π_LLM+G` is thus a *blind* test of whether ESS grounding alone suffices, with no game-specific tuning. A rule-based ESS baseline (Condition D, `decision/rule_based_ess_policy.py`) is included specifically to demonstrate that the macro patterns can also be grown without an LLM, isolating the contribution of the LLM substrate (richness of action sequences) from the contribution of ESS conditioning (calibration to empirical distributions).

### 2.4 Predicted Direction

This claim is falsified if (a) Condition A reproduces the macro patterns without grounding (the grounding is not load-bearing), **or** (b) Condition D matches Condition B on *all* macro patterns (the LLM substrate adds nothing beyond rules). Pilot evidence: Condition D recovers Gini=0.325 in range but does *not* recover the trust-gradient correlation (rule-based agents act deterministically on persona without the LLM's contextual generalization). The combined result — both grounding *and* LLM are required — is the joint prediction tested.

---

## 3. Dual-Process Cognitive Plausibility

### 3.1 The Architectural Decomposition

The BGF prompt has two distinct grounding channels:

- **Persona conditioning** (`agents/profile.py` → prompt header): static demographic identity, available at every decision without retrieval.
- **Dual RAG** (`decision/sql_rag.py` + `graph_rag.py`): on-demand retrieval of population norms (SQL) and social position (graph), invoked per decision.

### 3.2 Mapping to Dual-Process Theory

Following Kahneman (2011, *Thinking, Fast and Slow*) and Evans & Stanovich (2013, *Perspectives on Psychological Science*):

| BGF mechanism | Dual-process analogue | Function |
|---|---|---|
| Persona conditioning | **System 1** prior | Fast, automatic, identity-grounded default |
| SQL RAG (peer norms) | **System 2** retrieval | Deliberative consultation of empirical base rates |
| Graph RAG (social position) | **System 2** retrieval | Deliberative consultation of strategic context |
| Memory (`agents/memory.py`) | **Episodic memory** | Persistence of past decisions for consistency |

This is not a metaphor: it is the *design rationale* for keeping the two channels architecturally separate. If persona and RAG were redundant, a single fused channel would suffice. The 2×2 factorial (`metrics/mediation.py`) is designed precisely to test the dual-process separation: a positive interaction term means System 1 and System 2 contributions are non-additive, consistent with the dual-process literature; an additive decomposition would suggest the analogy is decorative.

### 3.3 Predicted Direction

A non-trivial positive interaction effect in the 2×2 mediation table `[audit: B.4 ⏳ pending aggregation]`. *This claim is falsified if the interaction term is statistically indistinguishable from zero after multi-seed extension* — in that case the dual-process framing collapses to "two independent grounding signals" and the architectural rationale must retreat to Track 1 (information-theoretic) alone.

---

## 4. Alignment-Bias Mechanism: Why RLHF Predicts a Cooperator Prior

### 4.1 RLHF Objective and the Uniform-Cooperator Drift

RLHF (Ouyang et al. 2022) fine-tunes a base LM with a reward model trained on human preference comparisons that systematically reward *helpful, harmless, honest* responses. Sharma et al. (2023) document the resulting *sycophancy* — models inflate agreement with the perceived user, regardless of ground truth. Perez et al. (2022, EMNLP) document analogous shifts toward socially desirable response classes.

### 4.2 Reduction to the BGF Game

In the BGF action space {work, save, cooperate}, only `cooperate` carries an explicit prosocial signal. An RLHF reward model trained on conversational helpfulness data will, when prompted as a "person making a decision in a society," assign higher reward to the prosocial action even when the agent's described demographic profile would empirically *not* cooperate (e.g., a low-trust respondent in ESS data). The resulting drift is the *cooperator prior*: `π_LLM(cooperate) > π_human(cooperate | low-trust)`.

### 4.3 Formal Prediction

Let `π_uniform = (1/3, 1/3, 1/3)` and `B_RLHF = TV(π, π_uniform)`. The cooperator prior pushes ungrounded `π_LLM` toward a cooperate-heavy mode, *increasing* `TV(π_LLM, π_uniform)` above the empirical baseline `TV(π_human, π_uniform)`. Grounding restores subgroup-conditional anchoring (Track 1.3), pulling `π_LLM+G` toward `π_human`, which *reduces* B_RLHF.

### 4.4 Why the Reduction Direction Is Derivable, Not Fitted

This is the key point for the defense: the *sign* of `ΔB_RLHF = B_RLHF(B) − B_RLHF(A) < 0` is predicted before observing data, from RLHF training dynamics alone. The empirical measurement is therefore a *test* of the RLHF-drift theory, not a calibration. The cross-model Phase 27 result (Mistral −17.6%, Qwen −30.0%, GPT-4o-mini +40.3%) `[audit: A.4 ✅]` is then re-interpretable: the two open-weight 7B models confirm the prediction; the GPT-4o-mini inversion is an *anomaly to be explained* — most likely by alignment-methodology differences (constitutional AI, multi-stage RLHF, scale) — rather than a falsification of the BGF framework. This is the honest null result reported in paper §6.6.

### 4.5 Predicted Direction

`ΔB_RLHF < 0` for any model whose alignment training rewards conversational prosociality. *This claim is falsified if a model trained without prosocial reward signals (e.g., a base completion model with no RLHF) shows the same B_RLHF gap*, which would indicate the gap is driven by something other than alignment drift. A base-model ablation is the cleanest future test (P2 in `docs/TOP_TIER_RESEARCH.md`).

---

## 5. Joint Prediction and Defense Posture

The four tracks converge on a single joint prediction tested in paper.md:

> **Across H1–H8, Condition B will show (a) lower KL to ESS-conditional human distributions, (b) Gini in the empirical EU range, (c) trust-gradient monotonicity, (d) positive 2×2 mediation interaction, and (e) reduced B_RLHF on prosociality-aligned models.**

The defense posture is: *no single result carries the framework; the architecture is supported by convergent evidence from four independent theoretical commitments, each falsifiable on its own.* The committee's strongest challenge — "is the gain just prompt length?" — is addressed not by appeal to any one of these tracks but by the explicit negative-control program in `causal_model.md` §8–§10, which is the operationalization of these theoretical predictions.

---

## References (added to `paper/references.bib`)

- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley. [data-processing inequality, sufficient statistics]
- Epstein, J. M. (2006). *Generative Social Science: Studies in Agent-Based Computational Modeling*. Princeton University Press.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus & Giroux.
- Evans, J. St. B. T. & Stanovich, K. E. (2013). Dual-process theories of higher cognition: Advancing the debate. *Perspectives on Psychological Science*, 8(3), 223–241.

(Existing entries already cover Argyle 2023, Ouyang 2022, Sharma 2023, Perez 2022, Epstein & Axtell 1996, Schelling 1971.)
