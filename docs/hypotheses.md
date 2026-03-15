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
