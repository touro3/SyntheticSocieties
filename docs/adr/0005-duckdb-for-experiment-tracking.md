# 5. DuckDB + Parquet for the experiment registry

- Status: accepted
- Date: 2026-03-15
- Deciders: BGF core
- Tags: persistence, devops, reproducibility

## Context

With 258+ completed experiment directories on disk and ~10 new runs
per session, browsing `experiments/*/summary.json` by `ls` and `jq` is
no longer viable. Analysis scripts need to query across runs ("all
Condition B cells at N=500 with seed in {1..10}") without re-parsing
hundreds of JSON files. Two design constraints:

1. **No server.** The framework runs on a single workstation, a
   HuggingFace Space, and a shared GPU server. Bringing up Postgres
   or MongoDB on each is operational overhead with zero scientific
   value.
2. **The store must be reproducible from disk.** If someone hands us a
   tarball of `experiments/`, we must be able to rebuild the index
   from scratch with one command.

Options:

- **SQLite.** Works, single file, no server. Row-oriented; analytical
  queries ("for every condition × seed, what's the mean coop rate?") are
  ~5–10× slower than a columnar engine.
- **Postgres.** Server-required; overkill for a single-user analytical
  workload.
- **DuckDB + Parquet.** Columnar, single-process, zero-config, native
  Parquet I/O so the index is just a view over disk artefacts. Same
  query language as Postgres for the parts we use.

## Decision

The registry is `tracker/experiment_index.parquet`, rebuildable from
disk by `tracker/build_index.py`. All cross-run queries go through
`tracker/analytics.py`, which opens an in-process DuckDB connection,
registers the parquet as a view, and exposes typed query helpers
(`top_seeds`, `condition_by_metric`, `regression_detection`).

The same `tracker/analytics.py` is reused by:

- `scripts/decide_n500_pivot.py` (§8.1.3 pivot decision)
- `scripts/analyze_cross_model_scaled.py`
- The `/experiments` API endpoint (`api/app.py`)

## Consequences

**Positive**

- One-line cross-run queries: `analytics.query("SELECT condition,
  AVG(brm) FROM index WHERE n_agents = 500 GROUP BY condition")`.
- The index is bytewise reproducible from `experiments/` — `git`
  ignores `*.parquet` so it never enters version control, but rebuilds
  in seconds.
- DuckDB also powers `decision/sql_rag.py` (ADR 0004), so we ship one
  DB engine across two distinct features.

**Negative**

- DuckDB is younger than Postgres / SQLite; long-tail bugs (especially
  around concurrent writers) are still being fixed upstream. We treat
  the registry as **read-mostly** and write via single-process rebuilds
  rather than incremental inserts.
- One more native dependency in `requirements-api.txt` (a wheel for
  every supported OS).

## Alternatives rejected

- **Plain SQLite** — analytical queries too slow at the column scale we
  need.
- **Stay on `glob('experiments/*/summary.json')`** — already painful at
  ~250 runs; will not survive the planned N=500 sweep.
