# BGF Experimental Hypotheses

This document formalizes the experimental hypotheses tested by the BGF simulation framework.

---

## H1: Empirical Grounding Improves Realism

**Claim**: LLM agents with ESS-derived personas produce more realistic wealth distributions than random or rule-based baselines.

**Metric**: Jensen–Shannon divergence between simulated and empirical wealth distributions.

**Comparison**: LLM policy vs. random, rule-based, and template baselines.

**Expected outcome**: JSD(LLM, ESS) < JSD(baseline, ESS)

---

## H2: Persona Conditioning Affects Cooperation

**Claim**: ESS persona conditioning significantly affects agent cooperation rates compared to no-persona ablation.

**Metric**: Cooperation rate (cooperate actions / total actions).

**Comparison**: `ablation=rich_persona` vs. `ablation=no_persona` vs. `ablation=minimal_persona`.

**Expected outcome**: Rich persona agents exhibit cooperation rates correlated with their trust_people attribute; no-persona agents show uniform cooperation.

---

## H3: Memory Improves Temporal Stability

**Claim**: Access to interaction history stabilizes agent behavior over time.

**Metric**: Mean round-to-round JSD (temporal stability metric).

**Comparison**: `ablation=no_memory` vs. full LLM with memory.

**Expected outcome**: mean_jsd(no_memory) > mean_jsd(with_memory)

---

## H4: Network Topology Modulates Inequality

**Claim**: Social network topology affects the rate and magnitude of wealth inequality emergence.

**Metric**: Gini coefficient trajectory over simulation rounds.

**Comparison**: Fully connected vs. random vs. small-world networks.

**Expected outcome**: Small-world networks produce higher Gini than fully-connected due to clustering effects.

---

## H5: LLM Decisions Are Robust to Seed Variation

**Claim**: LLM-based simulations produce stable aggregate outcomes across different random seeds.

**Metric**: Coefficient of variation (CV) of wealth mean across seeds.

**Comparison**: 5-seed sweep for LLM vs. template baselines.

**Expected outcome**: CV(LLM) < 0.15 (low variation indicates robustness).

---

## H6: Temperature Controls Decision Diversity

**Claim**: LLM sampling temperature controls the diversity of agent decisions without changing mean outcomes.

**Metric**: Shannon entropy of action distributions, mean wealth.

**Comparison**: Temperature = {0.1, 0.5, 0.7, 1.0}.

**Expected outcome**: Higher temperature → higher entropy, similar mean wealth.

---

## H7: Cross-Model Generalizability of RLHF Cooperative Bias

**Claim**: The RLHF cooperative bias (B_RLHF > 0 in Condition A) is a general property of instruction-tuned LLMs, not specific to Mistral-7B-Instruct-v0.3.

**Metric**: B_RLHF = TV(π, π_uniform) for each model × condition pair.

**Comparison**: Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, GPT-4o-mini — each tested in Condition A (ungrounded) and Condition B (ESS-grounded).

**Expected outcome**: B_RLHF(A) > 0 for all models (bias present universally); B_RLHF(B) < B_RLHF(A) for the majority of models (grounding reduces bias in most families).

**Result**: Partially confirmed. All three models exhibit B_RLHF > 0 in Condition A (bias is universal in kind). Grounding reduces B_RLHF for Mistral-7B (−17.6%) and Qwen2.5-7B (−30.0%) but increases it for GPT-4o-mini (+40.3%), identifying alignment methodology as a moderating variable. See Section 5.6, Table 3.

---

## H8: Trust-Gradient Recovery via Grounding Function

**Claim**: The grounding function Φ preserves ESS trust-to-cooperation gradients — sub-populations with higher ESS interpersonal trust produce higher simulated cooperation rates.

**Metric**: Spearman rank correlation between ESS trust group mean (μ_trust) and simulated cooperation rate (mean_coop_rate) across four trust bands.

**Comparison**: Low-Trust [0.2, 0.4), Moderate-Trust [0.4, 0.6), High-Trust [0.6, 0.8), Very-High-Trust [0.8, 1.0) sub-populations.

**Expected outcome**: Spearman r > 0 and statistically significant (p < 0.10); rank order preserved: coop_rate(VH) > coop_rate(H) > coop_rate(M) > coop_rate(L).

**Result**: Confirmed. Spearman r ≥ 0.80 (p < 0.10, n = 4 groups) across 3 seeds. Rank order preserved in all seed conditions. Reproducible without GPU via `make trust-gradient`. See Section 5.5, Table 2.
