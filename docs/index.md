# Behavioral Grounding Framework

The Behavioral Grounding Framework (BGF) is an agent-based simulation
platform that tests whether large language models grounded in empirical
socio-economic microdata (European Social Survey Round 11) produce more
realistic behaviour than off-the-shelf LLMs.

This documentation site is the navigable index of the project's
written artefacts. The source of truth for everything is the GitHub
repository — [`lucastourinho/SyntheticSocieties`](https://github.com/lucastourinho/SyntheticSocieties).

## Where to start

| If you want to …                                         | Read                                                                |
| -------------------------------------------------------- | ------------------------------------------------------------------- |
| Understand the scientific contribution                   | [the paper](paper.md)                                               |
| Read the TCC monograph (IDP structure)                   | [the monograph](monograph.md)                                       |
| Understand the architecture in 10 minutes                | the [ADR index](adr/README.md)                                      |
| Reproduce a result locally                               | the [reproduction commands in CLAUDE.md](https://github.com/lucastourinho/SyntheticSocieties/blob/main/CLAUDE.md) |
| Hit the API from a client                                | the [OpenAPI spec](api/openapi.yaml)                                |
| Audit which numbers in the paper are still trustworthy   | the [audit log](AUDIT_DATA_METRICS_LOGGING.md)                      |
| Cite the dataset                                         | the [datasheet](datasheet_ess_synthetic.md)                         |
| Cite the model                                           | the [model card](model_card_bgf.md)                                 |
| Propose a feature, file an audit bug, or contribute      | the GitHub issue templates                                          |

## Headline findings

- **Rule-based grounding (Condition D)** recovers Eurostat-range inequality
  (Gini = 0.325 ± 0.001 at N=500, T=30, 3 seeds) and the cross-cultural
  cooperation gradient (Spearman ρ = +1.000 across six ESS clusters,
  independently replicated against WVS Wave 7 and Herrmann–Thöni–Gächter
  per-city public-goods-game contributions).
- **The Φ/P_LLM dissociation.** Pre-registered confirmatory tests at N=100
  do not yet show the same effect when the LLM does the deciding. The
  N=500 LLM sweep currently in flight is the next data point.

See the [paper](paper.md) for the full story and the [audit log](AUDIT_DATA_METRICS_LOGGING.md)
for the bugs that have shifted reported numbers between revisions.
