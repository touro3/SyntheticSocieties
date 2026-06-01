<!-- Thanks for the PR. Keep the body short — the diff and the test
suite tell most of the story. -->

## Summary

<!-- One paragraph: what this changes and why. Reference an issue
or an ADR if applicable. -->

## Scope

<!-- Tick at least one. -->
- [ ] Bug fix (no behavioural change beyond the fix)
- [ ] Correctness / audit fix (changes published numbers — list affected runs below)
- [ ] New feature
- [ ] Refactor (no behavioural change)
- [ ] Documentation / ADR
- [ ] DevOps / CI / build
- [ ] Test-only

## Checklist

- [ ] `pytest tests/ --no-cov -p no:cacheprovider --ignore=tests/test_llm_policy.py --ignore=tests/test_cross_model.py` passes locally
- [ ] `make lint` clean
- [ ] `make type-check` clean (or new mypy errors explicitly accepted in PR description)
- [ ] `make bandit` clean for `agents`/`bgf_logging`/`decision`/`environment`/`metrics`/`population`/`simulation`/`tracker`/`utils`
- [ ] If touching the Flask API: `tests/test_openapi_spec.py` still passes and `docs/api/openapi.yaml` updated
- [ ] If introducing a new metric / condition / experimental sweep: pre-registration text in `docs/hypothesis_preregistration.md` updated
- [ ] If changing published numbers: corresponding `analysis/tables/*.csv` regenerated and `docs/AUDIT_DATA_METRICS_LOGGING.md` updated
- [ ] If shipping a non-obvious design choice: ADR added under `docs/adr/` (`make adr-new TITLE="…"`)
- [ ] CHANGELOG.md updated under `## [Unreleased]`

## Affected runs / artefacts

<!-- If this PR changes any published number, list the experiment_ids
and the metric(s) that move. If it doesn't, write "none". -->

## Risk and rollback

<!-- One line: blast radius of a bad merge + how to roll back. -->
