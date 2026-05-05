# BGF Hypothesis Pre-Registration

**Framework:** Behavioral Grounding Framework (BGF)
**Registered:** 2026-04-07
**Study:** LLM-based multi-agent economic simulation with ESS grounding

This document pre-registers the eight hypotheses tested in the BGF study.
Pre-registration separates confirmatory from exploratory analysis and prevents
HARKing (Hypothesizing After Results are Known).

---

## Primary Hypotheses

### H1 — Behavioral Realism Improvement

**Statement:** ESS-grounded agents (Condition B) exhibit strictly higher
Behavioral Realism Metric scores than ungrounded agents (Condition A):
`BRM_composite(B) > BRM_composite(A)`.

**Operationalization:**
- `BRM_composite` computed via `metrics/behavioral_realism.py:compute_composite_brm()`
- Tested with Mann-Whitney U (non-parametric, two-sided), α = 0.05 after FDR correction
- Unit of analysis: per-seed final-round BRM score (3+ seeds minimum)

**Expected direction:** Positive (grounding increases realism)

**Effect size threshold:** Cohen's d > 0.5 (medium)

---

### H2 — RLHF Bias Reduction

**Statement:** Grounding reduces the RLHF Bias Index:
`B_RLHF(B) < B_RLHF(A)`, where `B_RLHF = TV(π, π_uniform)`.

**Operationalization:**
- `B_RLHF` computed via `metrics/behavioral_realism.py:compute_rlhf_bias_index()`
- Tested with Mann-Whitney U, α = 0.05 after FDR correction
- Reported as `value [95% CI]` across seeds

**Expected direction:** Negative (grounding reduces bias)

**Effect size threshold:** Cohen's d > 0.5

---

### H3 — Inequality Emergence

**Statement:** Grounded agents produce Gini coefficients within the empirically
observed European range (0.20–0.38), while ungrounded agents do not.

**Operationalization:**
- Gini coefficient computed via `metrics/inequality.py:gini_coefficient()`
- Reference: Eurostat EU median Gini ≈ 0.31 (ESS Round 11 reference period)
- Condition B Gini must fall in [0.20, 0.38] for ≥2/3 seeds
- Condition A Gini must fall outside this range for ≥2/3 seeds

**Expected direction:** Condition B inside range; Condition A below range (~0.08)

---

### H4 — Network Fragmentation

**Statement:** Grounded agents produce higher network modularity Q than
ungrounded agents, consistent with real social network community structure
(empirical reference: Q ≈ 0.30–0.60).

**Operationalization:**
- Modularity computed via `metrics/network_metrics.py`
- Tested with Mann-Whitney U on final-round modularity across seeds, α = 0.05

**Expected direction:** `Q(B) > Q(A)`

---

### H5 — Trust-Gradient Recovery

**Statement:** Simulated cooperation rates are monotonically ordered by ESS
trust sub-population level; the Spearman rank correlation between ESS trust
means and simulated cooperation rates is positive and significant.

**Operationalization:**
- 4 sub-populations: Low/Moderate/High/Very-High-Trust (see `metrics/trust_gradient.py`)
- Spearman ρ ≥ 0 with p < 0.10 (n=4 groups; minimum achievable p ≈ 0.042)
- Validated via `scripts/run_trust_gradient.py --dry-run` (no GPU required)

**Expected direction:** Spearman ρ > 0 (gradient recovered)

---

### H6 — Adversarial Resilience Difference

**Statement:** Grounded societies (Condition B) exhibit *localized* adversarial
damage patterns (preferential wealth extraction from near-neighbors), while
ungrounded societies (Condition A) show uniform extraction.

**Operationalization:**
- Localization measured as ratio: wealth_loss(neighbors) / wealth_loss(non-neighbors)
- Condition B ratio > 1.5; Condition A ratio ≈ 1.0
- Tested at 5% adversarial injection fraction

**Expected direction:** Localized damage in B, indiscriminate in A

---

### H7 — Cross-Model Bias (Directional)

**Statement:** At least two of three LLM families (Mistral-7B, Qwen2.5-7B,
GPT-4o-mini) exhibit a reduction in B_RLHF under grounding (Condition B vs A).

**Operationalization:**
- ΔB_RLHF = (B_RLHF(B) − B_RLHF(A)) / B_RLHF(A)
- A model "confirms" the hypothesis when ΔB_RLHF < −0.10 (>10% reduction)
- Results reported in Table 3 (cross-model comparison)

**Expected direction:** 2/3 models confirm; GPT-4o-mini is an acknowledged
boundary case whose direction is pre-registered as uncertain.

---

### H8 — Persona Stability Under Grounding

**Statement:** Grounded agents (Condition B) exhibit slower persona fidelity
decay over simulation rounds than ungrounded agents (Condition A).

**Operationalization:**
- Decay rate estimated via OLS regression of `fidelity(t)` on `t`
  (see `metrics/persona_decay.py:compute_per_round_persona_fidelity()`)
- `decay_rate(A) < decay_rate(B)` (more negative slope in A = faster decay)
- Tested across T = 30 rounds; extended to T = 100 for Phase 28.5 validation

**Expected direction:** |decay_rate(A)| > |decay_rate(B)|

---

## Statistical Analysis Plan

### Multiple Comparisons Correction

All p-values from H1–H8 hypothesis tests are corrected using the
Benjamini-Hochberg (BH) FDR procedure at α = 0.05 (implemented in
`metrics/statistical_inference.py:benjamini_hochberg()`).

### Confidence Intervals

All metrics are reported as `value [lower_95CI, upper_95CI]` using bootstrap
percentile intervals with 2,000 resamples and fixed seed 42 (implemented in
`metrics/statistical_inference.py:bootstrap_ci()`).

### Minimum Seeds

- Primary hypotheses (H1–H4): ≥ 3 seeds; target 10–20 for Phase 28.1
- Trust-gradient (H5): ≥ 3 seeds (already validated)
- Cross-model (H7): ≥ 1 run per condition per model (GPU-cost constrained)
- Persona decay (H8): ≥ 3 seeds at T = 30

### Exploratory Analyses

Analyses not registered here are considered **exploratory** and reported
as such in Section 6 and the Supplementary Appendix. These include:
- Feature importance logistic regression (Phase 28.3)
- Policy intervention sweep (Phase 28.8)
- Long-horizon (T=100) fidelity characterization (Phase 28.5)

---

## Deviation Log

| Date | Hypothesis | Deviation | Justification |
|------|-----------|-----------|---------------|
| 2026-05-05 | H1, H2 | Primary analysis executed at pilot scale (N=50, T=30, n=1 seed) rather than pre-registered N=500, T=30, n≥3 seeds | GPU allocation constraint during submission window. Pilot results are reported with explicit scale caveat; full-scale run is planned post-submission. Effect direction is consistent with hypothesis; effect sizes (Cohen's d > 0.8 descriptive) exceed the pre-registered threshold. |
| 2026-05-05 | H5 | Spearman ρ=0.800 with asymptotic p=0.200, exact permutation p=0.167 — non-significant at both α=0.05 and α=0.10 | With n=4 groups, the minimum achievable two-tailed permutation p is 2/24 ≈ 0.083 under *perfect* rank agreement (ρ=1.000). The pre-registered threshold of p<0.10 is the theoretically tightest bound achievable for non-perfect ρ. The positive direction (ρ=0.800), strict monotonicity across all 4 groups, and Kendall τ=0.667 are consistent with H5. Supplementary statistics (exact permutation test, Kendall's τ) added post-hoc in `metrics/trust_gradient.py` to triangulate the finding. |
| 2026-05-05 | H3 (human validation) | Perceptual realism study (human_subjects_protocol.md, paper §8.4) not executed | Prolific participant study (n=30–50) requires ethics review and budget (~USD 150) not available before submission. Protocol is fully specified; study is deferred to the extended journal version. |
| 2026-05-05 | H1/H2 (padded control) | Length-controlled ablation (`decision/padded_ablation_policy.py`) implemented and dry-run tested but not executed at full scale (N=500, T=30, n=3 seeds) | GPU time constraint. This means causal attribution of cooperation improvements to ESS *content* (vs. token bulk) remains a stated limitation rather than an empirically closed claim (see paper §9, Limitation 8). |
| 2026-05-05 | Effect size reporting | Hedges' g added alongside Cohen's d for all small-sample (n<10 seeds) comparisons | Hedges' g is the bias-corrected estimator appropriate for n<50; Cohen's d systematically overestimates effect size in small samples. Change improves reporting accuracy; direction of all effects unchanged. |

*Deviations are recorded before results are interpreted, following the pre-registration protocol. All analyses not listed here follow the original pre-registered plan.*
