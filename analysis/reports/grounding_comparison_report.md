# Grounding Comparison Report

Main figures focus on `Auditable Random`, `Pure LLM`, and `Grounded LLM`.
Template and rule-based baselines are omitted from the main figures because earlier comparisons showed weak separation and limited explanatory value.

## Primary Summary

| Condition | Seeds | Wealth Mean | Gini | Stress Mean | Work Rate | Save Rate | Cooperation Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Auditable Random (ESS Persona) | 3 | 85.717 | 0.118 | 0.825 | 0.277 | 0.393 | 0.330 |
| Pure LLM (ESS Persona) | 3 | 89.417 | 0.080 | 0.927 | 0.320 | 0.667 | 0.013 |
| Grounded LLM (ESS Persona) | 3 | 87.083 | 0.147 | 1.053 | 0.300 | 0.193 | 0.507 |

## Interpretation

The main behavioral shift is not in work propensity. Work remains in a relatively narrow band across the primary conditions: `Auditable Random=0.277`, `Pure LLM=0.320`, and `Grounded LLM=0.300`.

The strongest difference appears in how non-work behavior is allocated between saving and cooperation. `Pure LLM` strongly favors saving (`save_rate=0.667`) and nearly eliminates cooperation (`cooperation_rate=0.013`). By contrast, `Grounded LLM` sharply reduces saving (`save_rate=0.193`) and increases cooperation (`cooperation_rate=0.507`). `Auditable Random` remains behaviorally mixed, with `save_rate=0.393` and `cooperation_rate=0.330`.

In short-horizon economic terms, `Pure LLM` currently performs best on mean wealth (`89.417`) and lowest inequality (`Gini=0.080`). `Grounded LLM` is more cooperative, but that increase in cooperation does not yet convert into lower inequality or lower stress over 5 rounds (`wealth_mean=87.083`, `Gini=0.147`, `stress_mean=1.053`). `Auditable Random` remains the lowest-stress baseline (`stress_mean=0.825`) while preserving a balanced action mix.

These results suggest that, in the current environment, cooperation is behaviorally meaningful but not yet rewarded quickly enough to outperform self-preserving strategies on short-horizon wealth and stress metrics.

## Caveats

- These comparisons are based on 3 seeds, so uncertainty estimates are still coarse.
- `Pure LLM` here means no population grounding, no social context, no memory context, and no balancing hint.
- `Grounded LLM` includes ESS-derived population grounding plus social context and memory, so it should not be interpreted as an ESS-only effect.
- Lorenz curves are generated as supplementary material only.