# Architecture Decision Records (ADRs)

This directory captures the **non-obvious design choices** that shaped the
Behavioral Grounding Framework. Each record explains one decision: the
context that forced it, the alternatives considered, what was picked,
and what the consequences are.

We use the lightweight [MADR 3.0](https://adr.github.io/madr/) template.

## Index

| #   | Title | Status | Date |
| --- | ----- | ------ | ---- |
| [0001](0001-policy-pluggability-via-protocol.md) | Policy pluggability via `PolicyProtocol` (PEP 544) | accepted | 2026-02-12 |
| [0002](0002-rule-based-grounding-as-baseline.md) | Rule-based ESS policy as the deterministic baseline (Condition D) | accepted | 2026-02-12 |
| [0003](0003-brm-as-paired-realism-metric.md) | BRM as a paired composite realism metric | accepted | 2026-02-19 |
| [0004](0004-sql-rag-and-graph-rag-split.md) | Split RAG into SQL (cohort) + Graph (social) backends | accepted | 2026-03-04 |
| [0005](0005-duckdb-for-experiment-tracking.md) | DuckDB + Parquet for the experiment registry | accepted | 2026-03-15 |

## How to add a new ADR

```bash
make adr-new TITLE="Use Wasserstein instead of JSD for calibration"
```

This drops a numbered template into `docs/adr/`. Fill in **Context**,
**Decision**, **Consequences** at minimum. Add it to the index above.

## Status workflow

- `proposed` — under discussion, not yet binding
- `accepted` — in effect, code reflects it
- `superseded by NNNN` — replaced by a newer ADR (link both ways)
- `deprecated` — no longer applies but kept for history

ADRs are append-only — never edit an accepted record. To change course,
write a new ADR that supersedes it.
