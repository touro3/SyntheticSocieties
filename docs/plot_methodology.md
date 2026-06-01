# Plot Methodology — What Creates Each Figure

This document explains the data sources, metrics, and computations used
to generate each publication figure in `analysis/figures/`.

---

## 1. `llm_grounding_comparison.png` — LLM-Alone vs ESS-Grounded

> **Most important figure**: demonstrates the value of empirical grounding.

| Panel | Data Source | Metric | Computation |
|-------|-----------|--------|-------------|
| A. Wealth Distribution | `cmp_llm_s{42,123,7}` vs `ablation_no_persona_s{42,123,7}` | Histogram of `summary.json → wealth.values` | Direct histogram comparison |
| B. Action Distribution | Same experiments | `event_action_counts` | Proportion of {work, save, cooperate} |
| C. Lorenz Curves | Same + `ablation_minimal_persona_*`, `ablation_rich_persona_*` | `metrics/inequality.py → lorenz_curve()` | Cumulative wealth share vs population share |
| D. Key Metrics | All ablation conditions | Gini via `gini_coefficient()`, cooperation rate | Bar comparison across grounding levels |

**LLM Model**: Mistral-7B-Instruct-v0.3 (float16, 4×P100)
**ESS Data**: Round 10 (2020–2022), 15 attributes per agent

---

## 2. `policy_heatmap.png` — Policy Comparison Heatmap

| Metric | Source | Formula |
|--------|--------|---------|
| Mean Wealth | `summary.json → wealth.values` | `np.mean(values)` |
| Gini | Same | `metrics/inequality.py → gini_coefficient()` |
| Work/Save/Coop % | `summary.json → event_action_counts` | `count / total_actions` |
| Stress | `summary.json → stress.mean` | Averaged across seeds |

**Policies**: LLM (ESS-Grounded), Template (ESS Archetypes), Rule-Based, Random
**Seeds**: 42, 123, 7 (3 runs per policy)

---

## 3. `ablation_effect.png` — Ablation Study

**Ablation conditions** (each modifies the LLM prompt):
- `rich_persona`: Full ESS persona (15 attributes)
- `minimal_persona`: Only age + gender
- `no_persona`: No persona information at all

**Metrics**: Wealth distribution, action proportions, Gini coefficient
**Data**: `ablation_{mode}_s{42,123,7}/summary.json`

---

## 4. `distribution_divergences.png` — Statistical Distances

| Metric | Module | Formula |
|--------|--------|---------|
| Jensen-Shannon Divergence | `metrics/distribution.py` | `JSD(P,Q) = H((P+Q)/2) - (H(P)+H(Q))/2` |
| KL Divergence | Same | `D_KL(P‖Q) = Σ P(x) log(P(x)/Q(x))` |
| Wasserstein Distance | Same | `scipy.stats.wasserstein_distance()` |

**Reference**: ESS-grounded LLM wealth distribution
**Compared against**: All other policies + ablation conditions

---

## 5. `lorenz_curves_all.png` — Lorenz Curves

**Module**: `metrics/inequality.py → lorenz_curve()`
**Formula**: Cumulative wealth share = `cumsum(sorted_wealth) / sum(wealth)`
**Gini**: `G = (2 × Σ(i × x_sorted[i])) / (n × Σ(x)) - (n+1)/n`

---

## 6. `calibration_gap.png` — Calibration vs Evaluation

**Module**: `metrics/calibration.py → calibration_evaluation_split()`
**Split**: Seeds 42+123 → calibration, Seed 7 → evaluation
**Gap**: `|eval_metric - cal_metric| / cal_metric × 100%`
**Risk levels**: <10% LOW, 10-25% MEDIUM, >25% HIGH

---

## 7. `perturbation_robustness.png` — Prompt Sensitivity

**Module**: `decision/prompt_perturbation.py`
**Modes**:
- `rephrase`: Paraphrase persona (same semantics, different words)
- `shuffle`: Randomize attribute order in persona block
- `noise`: Inject 2-3 distractor sentences

**Data**: `pert_{mode}_s{42,123,7}/summary.json`

---

## 8. `ladder_ablation.png` — V0-V5 Prompt Ablation Ladder

| Metric | Source | Formula |
|--------|--------|---------|
| Action Proportion | `summary.json → event_action_counts` | `count / total_actions` for {work, save, cooperate} |
| Gini | `summary.json → wealth.values` | `gini_coefficient(wealth)` |

**Data**: `abl_v{0..5}_llm_s{42,123,7}/summary.json`
**Description**: Tracks the effectiveness of a cumulative prompt engineering ladder (adding stress warnings, cooperation incentives, trust memory, neutral phrasing, and higher temperature) to recover behavioral diversity.

---

## 9. `results_dashboard.png` — Comprehensive Dashboard

Combines panels A-F from all above into single 2×3 figure.

---

## Existing Figures (from previous scripts)

| Figure | Script | Description |
|--------|--------|-------------|
| `policy_wealth_comparison.png` | `plot_policy_comparison_full.py` | Wealth histograms per policy |
| `policy_behavior_comparison.png` | Same | Action bars + cooperation rates |
| `policy_dynamics_comparison.png` | Same | Gini/wealth/cooperation over rounds |
| `policy_summary_radar.png` | Same | 5-metric radar chart |
| `empirical_vs_synthetic.png` | `plot_empirical_analysis.py` | ESS data vs synthetic population |
| `ess_demographics.png` | Same | Age/gender distributions |
| `ess_trust_politics.png` | Same | Trust & politics distributions |
| `ess_behavioral_profiles.png` | Same | Behavioral proxies |
| `ess_population_heatmap.png` | Same | ESS variable correlations |

---

## Hardware & Software

| Component | Specification |
|-----------|-------------|
| CPUs | 2× Xeon 22-Core (44 cores total) |
| GPUs | 4× NVIDIA Tesla P100 (16GB each) |
| RAM | 512 GB |
| CUDA | 12.8 (nvcc), 12.4 runtime |
| Driver | NVIDIA 550.144.03 |
| LLM | Mistral-7B-Instruct-v0.3 (float16) |
| Framework | PyTorch + HuggingFace Transformers |
| Data | European Social Survey Round 10 |
