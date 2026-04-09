.PHONY: all test reproduce reproduce-fast plots install format lint type-check coverage clean help

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
	@echo "  plots             Regenerate all figures from existing experiments"
	@echo "  trust-gradient    Run trust-gradient sub-population analysis"
	@echo "  phase-transitions Analyze phase transitions from existing runs"
	@echo "  install           Install project in editable mode with dev extras"
	@echo "  format            Auto-format with ruff"
	@echo "  lint              Run ruff linter"
	@echo "  type-check        Run mypy type checker"
	@echo "  coverage          Run tests with coverage report (fail under 70%%)"
	@echo "  clean             Remove Python cache files"
