.PHONY: all test reproduce reproduce-fast reproduce-figures reproduce-tables reproduce-docker \
        plots install format lint type-check coverage coverage-fast clean help \
        security bandit pip-audit openapi-lint adr-new pre-commit-install verify-deps lock \
        docs docs-serve docs-build

PYTHON := python
PYTEST := pytest
SEEDS  := 42,123,7

# ── Default ───────────────────────────────────────────────────────────────────
all: test

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v

test-fast:
	$(PYTEST) tests/ -v -x --tb=short

test-cov:
	$(PYTEST) tests/ -v --cov=. --cov-report=term-missing

# ── Reproduction (CPU-only, no GPU required) ──────────────────────────────────
reproduce:
	bash reproduce_paper.sh

reproduce-fast:
	$(PYTHON) scripts/run_full_pipeline.py \
		--rounds 5 \
		--agents 10 \
		--seeds $(SEEDS)

reproduce-full:
	$(PYTHON) scripts/run_full_pipeline.py \
		--rounds 30 \
		--agents 100 \
		--seeds $(SEEDS)

# Figure-only regen from on-disk experiments — no simulations are run.
reproduce-figures:
	bash scripts/reproduce_paper.sh --figures

# Re-aggregate every analysis table/JSON from on-disk experiment dirs without
# re-running any simulation. Idempotent — safe to invoke on every commit.
# Covers: §8.1 modularity, §8.1 experiment-matrix CSV, §8.2 Condition-D N=500
# 10-seed JSON, the BRM Proposition 3 certificate, and the §6.0 forest plot.
reproduce-tables:
	@echo "[tables] Re-aggregating analysis tables from on-disk experiments..."
	@if compgen -G "experiments/mx_A_s*" > /dev/null 2>&1; then \
		$(PYTHON) scripts/extract_modularity_from_cells.py \
			--out analysis/tables/network_modularity_8_1.csv; \
		$(PYTHON) scripts/run_experiment_matrix.py --conditions A B \
			--seeds 1,2,3,4,5,6,7,8,9,10 --rounds 30 --agents 100 --skip-existing \
			--output-csv analysis/tables/experiment_matrix_results.csv | tail -5; \
	else \
		echo "        (no mx_A_s* cells on disk — skipping §8.1 re-aggregation)"; \
	fi
	@if compgen -G "experiments/mx_D_n500_s*" > /dev/null 2>&1; then \
		$(PYTHON) scripts/build_condition_d_table.py \
			--seeds 1,2,3,4,5,6,7,8,9,10 --prefix mx_D_n500_s \
			--out-json analysis/condition_d_results_n500_10seed.json | tail -3; \
	else \
		echo "        (no mx_D_n500_s* cells on disk — skipping §8.2 N=500 D)"; \
	fi
	$(PYTHON) analysis/brm_sensitivity.py --emit-certificate --n-samples 500 \
		--out-json analysis/tables/brm_sensitivity.json
	@if [ -f analysis/tables/forest_plot.json ]; then \
		$(PYTHON) analysis/forest_plot.py | tail -3 || true; \
	fi
	@echo "[tables] Done."

# Build + run the CPU-only reproducibility container. Asserts pytest passes
# inside the container at build time, then regenerates figures on run.
reproduce-docker:
	docker build -f Dockerfile.repro -t bgf-repro:1.0.0 .
	docker run --rm -v "$$(pwd)/analysis:/app/analysis" bgf-repro:1.0.0

# Pin the full dependency closure from the validated venv. Run only when
# requirements*.txt has been intentionally changed and the env reinstalled.
lock:
	@test -d venv || (echo "venv/ not present — create it first"; exit 1)
	. venv/bin/activate && pip freeze --exclude-editable > requirements.lock.txt
	@echo "Wrote requirements.lock.txt ($$(wc -l < requirements.lock.txt) lines)"

# ── Plots only ────────────────────────────────────────────────────────────────
plots:
	$(PYTHON) scripts/run_full_pipeline.py --plots-only

# ── Phase-specific analyses ───────────────────────────────────────────────────
trust-gradient:
	$(PYTHON) scripts/run_trust_gradient.py

phase-transitions:
	$(PYTHON) scripts/run_phase_transition_sweeps.py --analyze-only

# ── Code quality ──────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

format:
	ruff format . && ruff check --fix .

lint:
	ruff check .

type-check:
	mypy . --ignore-missing-imports --exclude tests/

coverage:
	$(PYTEST) tests/ -q -k "not llm" --cov=. --cov-report=xml --cov-report=term-missing --cov-fail-under=70

# Fast coverage subset for capstone defense — excludes slow tests, short summary.
coverage-fast:
	$(PYTEST) tests/ -q -k "not llm and not slow" --cov=agents --cov=decision \
		--cov=environment --cov=metrics --cov=population --cov=simulation \
		--cov=tracker --cov=utils \
		--cov-report=term --no-header --tb=no -p no:cacheprovider

# ── Security ──────────────────────────────────────────────────────────────────
pip-audit:
	pip install --quiet pip-audit
	pip-audit --requirement requirements-ci.txt --disable-pip --strict
	@echo "(API requirements scanned in CI weekly; run pip-audit -r requirements-api.txt manually if needed.)"

bandit:
	pip install --quiet "bandit[toml]>=1.7"
	bandit -r agents bgf_logging decision environment metrics population simulation tracker utils -ll -ii

security: pip-audit bandit
	@echo "✓ Security scans clean (pip-audit + bandit)."

# ── OpenAPI ───────────────────────────────────────────────────────────────────
openapi-lint:
	@command -v npx >/dev/null 2>&1 || { echo "npx not found — install Node.js."; exit 1; }
	npx --yes @redocly/cli@latest lint docs/api/openapi.yaml

# ── Architecture Decision Records ─────────────────────────────────────────────
# Usage: make adr-new TITLE="Use DuckDB for tracker"
adr-new:
	@test -n "$(TITLE)" || (echo "Usage: make adr-new TITLE=\"...\""; exit 1)
	@bash docs/adr/new.sh "$(TITLE)"

# ── Pre-commit ────────────────────────────────────────────────────────────────
pre-commit-install:
	pip install --quiet pre-commit
	pre-commit install --install-hooks

# ── Reproducibility ───────────────────────────────────────────────────────────
verify-deps:
	@echo "Pinning sanity check:"
	@grep -c '==' requirements-api.txt || true
	@grep -c '==' requirements-ci.txt || true

# ── Docs site (MkDocs Material) ──────────────────────────────────────────────
docs: docs-serve
docs-serve:
	pip install --quiet mkdocs mkdocs-material mkdocs-autorefs "mkdocstrings[python]" pymdown-extensions
	mkdocs serve

docs-build:
	pip install --quiet mkdocs mkdocs-material mkdocs-autorefs "mkdocstrings[python]" pymdown-extensions
	mkdocs build --strict --site-dir site

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache htmlcov .coverage

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "Available targets:"
	@echo "  test              Run full test suite (481+ tests)"
	@echo "  test-fast         Run tests, stop on first failure"
	@echo "  reproduce         Full reproduction via reproduce_paper.sh"
	@echo "  reproduce-fast    Quick 5-round/10-agent baseline run"
	@echo "  reproduce-full    Full 30-round/100-agent run (CPU baselines only)"
	@echo "  reproduce-figures Regenerate every figure from on-disk experiments (no sim)"
	@echo "  reproduce-tables  Re-aggregate analysis tables/JSON from on-disk experiments"
	@echo "  reproduce-docker  Build + run CPU-only repro container (asserts tests pass)"
	@echo "  lock              Re-freeze requirements.lock.txt from the active venv"
	@echo "  coverage-fast     Coverage on non-LLM core modules (capstone defense subset)"
	@echo "  plots             Regenerate all figures from existing experiments"
	@echo "  trust-gradient    Run trust-gradient sub-population analysis"
	@echo "  phase-transitions Analyze phase transitions from existing runs"
	@echo "  install           Install project in editable mode with dev extras"
	@echo "  format            Auto-format with ruff"
	@echo "  lint              Run ruff linter"
	@echo "  type-check        Run mypy type checker"
	@echo "  coverage          Run tests with coverage report (fail under 70%%)"
	@echo "  clean             Remove Python cache files"
	@echo "  pip-audit         Scan CI requirements for known CVEs"
	@echo "  bandit            Static security scan of production code"
	@echo "  security          pip-audit + bandit"
	@echo "  openapi-lint      Lint docs/api/openapi.yaml via Redocly CLI (needs Node)"
	@echo "  adr-new TITLE=... Create a new ADR in docs/adr/"
	@echo "  pre-commit-install Install pre-commit hooks locally"
	@echo "  verify-deps       Quick pinning sanity check"
