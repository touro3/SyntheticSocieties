# Grounding Comparison Report

Main figures intentionally focus on `Auditable Random`, `Pure LLM`, and `LLM + ESS`.
Template and rule-based baselines are omitted from the main figures because prior comparisons showed weak separation and limited explanatory value.

## Primary Summary

| Condition | Seeds | Wealth Mean | Gini | Stress Mean | Work Rate | Save Rate | Cooperation Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Auditable Random
(ESS Persona) | 3 | 85.717 | 0.118 | 0.825 | 0.277 | 0.393 | 0.330 |
| LLM + ESS
(ESS Persona) | 3 | 87.083 | 0.147 | 1.053 | 0.300 | 0.193 | 0.507 |
| Pure LLM
(ESS Persona) | 3 | 89.417 | 0.080 | 0.927 | 0.320 | 0.667 | 0.013 |

## Notes

- `Pure LLM` here means no population grounding, no social context, no memory context, and no balancing hint.
- `LLM + ESS` includes ESS-derived population grounding plus social context and memory.
- `Auditable Random` is a weighted stochastic baseline with deterministic seeds and per-step audit logs.
- Lorenz curves are generated as supplementary material only.