# Contributing to SyntheticSocieties

## Dev Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd SyntheticSocieties

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install in editable mode with dev extras
make install
# or: pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

## Running Tests

```bash
# Full suite (no GPU required)
pytest tests/ -v

# Fast mode — stop on first failure
make test-fast

# Exclude GPU-dependent tests explicitly
pytest tests/ -k "not llm_backend and not huggingface and not gpu"

# Coverage report (must be ≥ 70%)
make coverage
```

Tests live in `tests/`. Shared fixtures are in `tests/conftest.py`. All new code requires tests — follow the TDD approach already used throughout the project.

## Adding a New Policy

1. Create `decision/my_policy.py` implementing `propose_action(profile, state, memory, context, round_id) -> ProposedAction`
2. Inherit from `LLMPolicyBase` if the policy uses an LLM backend (gives you `_generate_with_retries`, `_fallback_action`, RAG accessors)
3. Add the policy type to `configs/schema.py` → `PolicyConfig.type` literal
4. Wire it up in `simulation/kernel.py` where the policy is instantiated from config
5. Add `tests/test_my_policy.py` with at least: action output type, fallback on backend error, valid action_type values

## Adding a New Metric

1. Create `metrics/my_metric.py` with a pure function `compute_my_metric(agents, rounds) -> float | dict`
2. Call `gini_coefficient` from `metrics/inequality.py` if you need Gini — do not reimplement it
3. Wire the metric into `metrics/__init__.py` if it should be part of the standard report
4. Add `tests/test_my_metric.py`

## Code Quality

```bash
# Lint (ruff — replaces flake8)
make lint

# Auto-format
make format

# Type checking
make type-check
```

The project uses `ruff.toml` for linting configuration (line-length 120, isort, pyflakes, pyupgrade). `Optional[X]` is preferred over `X | None` for annotation clarity — UP007/UP045 are intentionally ignored.

## Pre-commit

The `.pre-commit-config.yaml` runs `ruff` (lint + format) and `mypy` on every commit. To run it manually:

```bash
pre-commit run --all-files
```

## Pull Request Checklist

- [ ] New tests added for all new logic
- [ ] `make lint` passes with zero errors on touched files
- [ ] `pytest tests/ -k "not llm"` passes locally
- [ ] No new `warnings.warn()` calls — use `logging.getLogger(__name__).warning(...)` instead
- [ ] No new Gini implementations — import from `metrics/inequality.py`
- [ ] Config changes reflected in `configs/schema.py` (Pydantic validation)

## Project Structure Quick Reference

| Layer | Purpose |
|-------|---------|
| `agents/` | Agent data structures (profile, state, memory) |
| `decision/` | Policies, prompt builders, output parser, RAG backends |
| `environment/` | Economy engine, network topology, world state |
| `simulation/` | Simulation kernel (batched event loop) |
| `metrics/` | 20+ evaluation metrics, calibration, Gini |
| `population/` | ESS-grounded agent population synthesis |
| `tracker/` | DuckDB experiment registry |
| `configs/` | Hydra YAML configs + Pydantic schema |
| `scripts/` | Experiment pipelines (run_*.py) |
| `tests/` | 1050+ unit and integration tests |
