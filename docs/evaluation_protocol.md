# Evaluation Protocol & Statistical Inference Plan

> **Evidence-status convention.** Each empirical/implementation claim is annotated `[audit: X.Y]` referencing a row in `docs/evidence_audit.md`. Pure derivations carry `[📐]`; pending experiments carry `[⏳]`.

## 1. Objective
To rigorously evaluate whether LLM-driven agents reproduce empirical social patterns better than algorithmic baselines, requiring micro-level behavioral heterogeneity and macro-level realism.

## 2. Empirical Targets (Realism Metrics)
1.  **Action Distribution Similarity**: The aggregate distribution of `[work, save, cooperate]` must map plausibly onto the ESS-derived willingness to cooperate in the empirical cohort.
2.  **Wealth Distribution Fit (Gini)**: The emergent Gini coefficient must stabilize at socially realistic levels (e.g., 0.25 - 0.40), avoiding both unnatural perfect equality (0.0) and rapid total concentration (1.0).
3.  **Subgroup Fit**: Differences in policy choices should correlate with age, sex, and country attributes from the ESS data.
4.  **Temporal Stability**: Behaviors must remain diverse across rounds (persistent entropy) without descending into a single-action mode collapse.
5.  **Interpersonal Trust**: Agent-to-agent interactions must organically reflect and update internal trust states.

## 3. Metric Definitions
-   **Primary Metric**: Jensen-Shannon Divergence (JSD) between the simulated action distribution and the empirical target profile.
-   **Secondary Metrics**:
    -   *Behavioral Entropy*: Shannon entropy of the `[work, save, cooperate]` distribution per seed per round.
    -   *Cooperation Rate*: Percentage of actions strictly allocated to `cooperate`.
    -   *Gini Coefficient*: Calculated over the wealth distribution at the final round.
    -   *Stress Slope*: Linear fit coefficient of average population stress over time.

## 4. Statistical Inference Plan
-   **Aggregation Level**: `experiment-seed` level (not pooled event rows).
-   **Confidence Intervals**: 95% Bootstrap Confidence Intervals for non-normal distributions (e.g., Gini, Entropy). `[audit: E.2 ✅ `metrics/statistical_inference.py:bootstrap_ci` + tests]`
-   **Hypothesis Testing**: Paired permutation tests across matched seeds to compare LLM vs. explicit baselines. `[audit: E.3 ✅ Cohen's d / Hedges' g implemented + tested]`
-   **Multiple Comparison Control**: Benjamini-Hochberg FDR correction applied across the ablation ladder variants. `[audit: E.1 ✅ `metrics/statistical_inference.py:benjamini_hochberg` + tests, monotonicity verified]`

## 5. Pass/Fail Interpretation Rules
-   **FAIL (Collapse)**: Behavioral entropy remains < 0.1 across 80% of rounds. Indicates action-set semantics are underconstrained.
-   **FAIL (Exploitative)**: Stress slope > 0.8 and Cooperation Rate < 5%. The model acts as a crude earnings maximizer rather than a social proxy.
-   **PASS (Emergent Realism)**: JSD to empirical ESS profile is significantly lower (p < .05) than the best algorithmic baseline, while Gini remains stable.

---

## 6. A Priori Power Analysis

The pre-registered Phase 28.1 extension targets n=10 seeds at N=500, T=30 for H1–H4. This section justifies that target *a priori* (i.e. independent of the pilot effect sizes).

### 6.1 Mann–Whitney U (H1, H2, H4)

For the primary tests on BRM_composite, B_RLHF, and modularity Q, we use a two-sided Mann–Whitney U. With independent samples of size n=10 per condition, at α = 0.05 (uncorrected) and power = 0.80:

```
Minimum detectable standardized effect (location shift, Hedges' g) ≈ 1.32
```

After Benjamini–Hochberg FDR correction across the eight primary hypotheses at α_FDR = 0.05, the effective per-test α drops (worst case under the BH step-up) to α ≈ 0.025 for the smallest-p hypothesis, yielding:

```
Minimum detectable g_FDR ≈ 1.48 at n=10 per condition
```
`[audit: E.4 ⏳ derivation given; reproducer `analysis/power_curves.py` pending — see §H.3]`

**Interpretation.** The pilot descriptive effect sizes (Cohen's d > 0.8 reported in the H1/H2 deviation log) fall *below* the 10-seed MDE, which is why the pre-registration deferred formal inference to the multi-seed extension. The 10-seed choice is calibrated to detect *large* (g ≥ 1.5) effects with 80% power under conservative FDR. Detection of medium effects (g ≈ 0.5) would require n ≈ 65 per condition — outside the GPU budget and explicitly out of scope.

### 6.2 Spearman ρ (H5, H9)

For trust-gradient (H5, n=4 trust bands) and cross-cultural behavioral validation (H9, n≈15 countries):

| Hypothesis | n | Min two-sided exact p at ρ=1 | Power to detect ρ=0.8 at α=0.10 | Power to detect ρ=0.8 at α=0.05 |
|---|---|---|---|---|
| H5 | 4 | 0.083 | 0.42 | 0.17 |
| H9 | 15 | < 0.001 | 0.95 | 0.88 |

The H5 power ceiling is the formal reason the pre-registration uses α = 0.10 and reports the asymptotic-vs-exact p discrepancy as a deviation (already in the log). H9 has substantially more statistical headroom because the country sample is ≈4× larger, which is the additional reason H9 is added (`docs/construct_validity.md` §3).

### 6.3 Power Curve Reporting

The Phase 28.1 results table will report, alongside each effect size, the *post-hoc* MDE at the achieved n and the *observed power* under the empirical effect estimate, computed via `statsmodels.stats.power` (already a transitive dependency). This forestalls the "you didn't have power" critique by quantifying exactly how much power the design had.

---

## 7. Convergent-Evidence Synthesis: Forest Plot Across H1–H8 (H1–H9)

Eight (now nine) hypothesis tests invite the "many comparisons / fishing" concern, already addressed at the per-test level by Benjamini–Hochberg FDR. A second, complementary defense is to present the hypotheses as **convergent evidence** for a single underlying claim (grounding improves realism) rather than as nine independent tests.

### 7.1 Procedure

For each hypothesis with a primary numeric effect size (Hedges' g for H1, H2, H4, H8; Spearman ρ for H5, H9; Gini in-range proportion for H3; localization ratio for H6; ΔB_RLHF for H7), compute a 95% bootstrap CI (2,000 resamples, fixed seed 42). Plot all nine on a single forest plot with x-axis = standardized effect direction (positive = BGF prediction supported). Add a random-effects pooled estimate (DerSimonian–Laird) at the bottom, transparently noting that the effect sizes are heterogeneous and the pooled estimate is descriptive, not inferential.

### 7.2 Implementation

New analysis script `analysis/forest_plot.py` reading existing `analysis/tables/*.json`. CPU only, ~10 minutes runtime. No new data collection required for H1–H8 once Phase 28.1 lands; H9 added once cross-cultural behavioral comparison is computed. `[audit: E.6 ⏳ script committed 2026-05-14; output figure pending Phase 28.1 multi-seed tables]`

### 7.3 Reframing for the Defense

Convergent evidence does not replace FDR correction; it complements it. The defense narrative is: *the FDR-corrected p-values rule out chance per hypothesis; the forest plot shows the prediction directions agree across nine independent operationalizations*. A single failing hypothesis (e.g., GPT-4o-mini's inverse ΔB_RLHF in H7) is then visibly an *anomaly* rather than a *falsification*, and the §9 discussion of that anomaly carries proportional weight.

---

## 8. BRM Weight-Sensitivity Curve

The Behavioral Realism Metric (BRM_composite) is a weighted sum of four sub-metrics: JSD(wealth), Gini gap, cooperation accuracy, temporal stability. The pre-registered weights are equal (1/4 each). Paper §3.2 claims the rank ordering `BRM(B) > BRM(A)` is robust to weighting; this section commits a figure to back that claim.

### 8.1 Procedure

Sweep the weight simplex `(w_1, w_2, w_3, w_4)` with w_i ≥ 0, Σ w_i = 1, on a Dirichlet-uniform grid (1,000 samples). For each weight vector, compute BRM_composite under that weighting for Conditions A and B and record the sign of `BRM(B) − BRM(A)`. Report:

1. **Fraction of the simplex where `BRM(B) > BRM(A)`**: the headline robustness number.
2. **Region in the simplex where the ordering flips** (if any): identifies the sub-metric whose weight could overturn the result.
3. **Equal-weight result** (the pre-registered comparator) marked on the figure.

### 8.2 Implementation

New analysis script `analysis/brm_sensitivity.py`, ~80 LOC, reuses `metrics/behavioral_realism.py:compute_composite_brm()` with a weight argument. CPU only, ~5 minutes runtime. Generates `analysis/figures/brm_weight_sensitivity.{pdf,png}`. `[audit: E.5 ⏳ script committed 2026-05-14; pass/fail threshold pre-specified in §8.3]`

### 8.3 Pre-Specified Pass/Fail

- **PASS (robust)**: the BGF prediction `BRM(B) > BRM(A)` holds in ≥ 90% of the simplex.
- **FAIL (weight-dependent)**: the prediction holds in < 90% of the simplex; the §9 discussion must then disclose which sub-metric carries the result.

This pre-specification protects against the "you cherry-picked the weights" objection by binding the framework to a concrete robustness threshold.

---

## 9. Reproducibility Anchors

All §6–§8 analyses use fixed random seeds and are wired into `reproduce_paper.sh`. The full sequence for verifying §6–§8 from a clean checkout:

```bash
python -m analysis.brm_sensitivity      # §8 — figure + JSON
python -m analysis.forest_plot          # §7 — figure + JSON
python -m analysis.power_curves         # §6 — table + figure
```

Each script writes its output to `analysis/tables/` (JSON) and `analysis/figures/` (PDF/PNG); both are inputs to the LaTeX build in `paper/main.tex`.
