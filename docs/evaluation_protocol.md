# Evaluation Protocol & Statistical Inference Plan

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
-   **Confidence Intervals**: 95% Bootstrap Confidence Intervals for non-normal distributions (e.g., Gini, Entropy).
-   **Hypothesis Testing**: Paired permutation tests across matched seeds to compare LLM vs. explicit baselines.
-   **Multiple Comparison Control**: Benjamini-Hochberg FDR correction applied across the ablation ladder variants.

## 5. Pass/Fail Interpretation Rules
-   **FAIL (Collapse)**: Behavioral entropy remains < 0.1 across 80% of rounds. Indicates action-set semantics are underconstrained.
-   **FAIL (Exploitative)**: Stress slope > 0.8 and Cooperation Rate < 5%. The model acts as a crude earnings maximizer rather than a social proxy.
-   **PASS (Emergent Realism)**: JSD to empirical ESS profile is significantly lower (p < .05) than the best algorithmic baseline, while Gini remains stable.
