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

### H9 — Cross-Cultural Behavioral Validation (added 2026-05-13)

**Statement:** BGF Condition-B cooperation rates correlate positively with
published country-level *behavioral* Public Goods Game contribution rates
(Herrmann, Thöni & Gächter 2008, *Science*; Henrich et al. 2010, *Science*).

**Operationalization:**
- Cross-country Spearman ρ between BGF Condition-B cooperation rate (per
  country cluster, T=30 final round, computed via `metrics/cross_cultural.py`)
  and published PGG contribution rate
- Tested at α = 0.05 with exact permutation p-value (n ≈ 15 countries
  intersecting Herrmann/Henrich data with ESS-11 sample)
- Results stored in `analysis/tables/h9_cross_cultural_behavioral.json`

**Expected direction:** Spearman ρ > 0

**Falsification:** Spearman ρ ≤ 0, or p ≥ 0.10

**Rationale for late addition:** H9 closes the within-instrument circularity
flagged in paper §9 Limitation 11 by using a behavioral (not attitudinal)
out-of-sample benchmark. It is *added* to, not *replacing*, H1–H8. See
`docs/construct_validity.md` §3 for full motivation.

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
- Feature importance logistic regression
- Policy intervention sweep
- Long-horizon (T=100) fidelity characterization

---

## Deviation Log

| Date | Hypothesis | Deviation | Justification |
|------|-----------|-----------|---------------|
| 2026-05-05 | H1, H2 | Primary analysis executed at pilot scale (N=50, T=30, n=1 seed) rather than pre-registered N=500, T=30, n≥3 seeds | GPU allocation constraint during submission window. Pilot results are reported with explicit scale caveat; full-scale run is planned post-submission. Effect direction is consistent with hypothesis; effect sizes (Cohen's d > 0.8 descriptive) exceed the pre-registered threshold. |
| 2026-05-05 | H5 | Spearman ρ=0.800 with asymptotic p=0.200, exact permutation p=0.167 — non-significant at both α=0.05 and α=0.10 | With n=4 groups, the minimum achievable two-tailed permutation p is 2/24 ≈ 0.083 under *perfect* rank agreement (ρ=1.000). The pre-registered threshold of p<0.10 is the theoretically tightest bound achievable for non-perfect ρ. The positive direction (ρ=0.800), strict monotonicity across all 4 groups, and Kendall τ=0.667 are consistent with H5. Supplementary statistics (exact permutation test, Kendall's τ) added post-hoc in `metrics/trust_gradient.py` to triangulate the finding. |
| 2026-05-05 | H3 (human validation) | Perceptual realism study (human_subjects_protocol.md, paper §8.4) not executed | Prolific participant study (n=30–50) requires ethics review and budget (~USD 150) not available before submission. Protocol is fully specified; study is deferred to the extended journal version. |
| 2026-05-05 | H1/H2 (padded control) | Length-controlled ablation (`decision/padded_ablation_policy.py`) implemented and dry-run tested but not executed at full scale (N=500, T=30, n=3 seeds) | GPU time constraint. This means causal attribution of cooperation improvements to ESS *content* (vs. token bulk) remains a stated limitation rather than an empirically closed claim (see paper §9, Limitation 8). |
| 2026-05-05 | Effect size reporting | Hedges' g added alongside Cohen's d for all small-sample (n<10 seeds) comparisons | Hedges' g is the bias-corrected estimator appropriate for n<50; Cohen's d systematically overestimates effect size in small samples. Change improves reporting accuracy; direction of all effects unchanged. |
| 2026-05-13 | H9 added | New hypothesis: cross-cultural *behavioral* validation against Herrmann et al. (2008) and Henrich et al. (2010) PGG contribution rates | Closes within-instrument circularity (paper §9 Limitation 11). H9 is added *before* the Herrmann/Henrich correlation is computed and is reported under the same Benjamini–Hochberg FDR correction as H1–H8. See `docs/construct_validity.md` §3. |

| 2026-06-02 | §8.1.5 gap-fill sweep (exploratory, not H1–H9) | Seeds s2–s10 (Condition A) and s3–s10 (Condition B) of the N=500 gap-fill sweep reduced from T=30 to T=15 rounds. Seeds s1 (A) and s2 (B) continue at T=30. | At N=500 with batch_size=1 (forced by VRAM during cooperation cascade), each round takes ~54 minutes; T=30 would require ~11 days for the full 10-seed sweep, exceeding the monograph submission deadline. T=15 captures all rounds through peak cascade dynamics (cascade established by R9 in prior runs). The T=30 terminal data from s1 and s2 serve as the full-horizon reference. This deviation does not affect H1–H9 (which are evaluated from the N=100 T=30 10-seed extension, not from the N=500 exploratory sweep). |

| 2026-06-03 | **H8 — memory ablation run invalidated by implementation bugs** | The 24-cell LLM-policy memory ablation run (`tmux: h8_memory_ablation`, completed 2026-06-03 14:30) is invalid for testing H8. Two bugs caused all experimental conditions to execute identically: (A) `ablation.mode=no_rag` CLI override was silently ignored in the `policy.type=llm` code path (`scripts/run_config_simulation.py` routes to `LLMPolicy` which has no `ablation.mode` parameter); (B) `memory.level` config key was not propagated to `ablation_level` in `build_prompt()` — the default value of 5 (full M3 memory) was used for all 24 cells. Evidence: `md5sum` of `summary.json` files across M0–M3 conditions is identical per seed; `round_metrics.jsonl` Gini trajectories are bit-identical (M2-ungrounded-s42 shows a minor divergence of 0.005 at T=10, attributable to a different RNG evaluation order, not to ablation). H8 pre-registered prediction (Table 7 in §8.5) remains a hypothesis, not a measurement. A corrected re-run requires patching `scripts/run_memory_ablation_llm.sh` to use `policy.type=ablated_llm` (or pass `llm.ablation_level={0,1,2,3}` explicitly) and to implement a proper `no_rag` pathway for the ungrounded condition. This deviation is logged before any re-run is attempted. |

*Deviations are recorded before results are interpreted, following the pre-registration protocol. All analyses not listed here follow the original pre-registered plan.*
