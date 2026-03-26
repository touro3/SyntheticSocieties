#!/usr/bin/env bash
# reproduce_paper.sh
# One-command reproduction of the BGF paper results (CPU-only, no GPU required).
# GPU-dependent LLM experiments are excluded; baselines and all metrics are included.
#
# Usage:
#   bash reproduce_paper.sh
#   bash reproduce_paper.sh --full   # 30 rounds, 100 agents (slower)

set -euo pipefail

FULL=0
if [[ "${1:-}" == "--full" ]]; then
    FULL=1
fi

echo "======================================================="
echo "  Behavioral Grounding Framework — Paper Reproduction"
echo "======================================================="
echo ""

# ── 1. Python version check ────────────────────────────────────────────────
REQUIRED_PYTHON="3.10"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[1/5] Python version: $PYTHON_VERSION (required >= $REQUIRED_PYTHON)"

python3 -c "
import sys
major, minor = sys.version_info.major, sys.version_info.minor
if (major, minor) < (3, 10):
    print('ERROR: Python >= 3.10 is required')
    sys.exit(1)
"

# ── 2. Check / create virtualenv ──────────────────────────────────────────
echo ""
echo "[2/5] Checking virtual environment..."
if [[ -d venv ]]; then
    echo "      Found existing venv/ — activating."
    source venv/bin/activate
else
    echo "      No venv found. Creating venv/..."
    python3 -m venv venv
    source venv/bin/activate
    echo "      Installing dependencies from requirements.txt..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
fi

# ── 3. Run test suite ─────────────────────────────────────────────────────
echo ""
echo "[3/5] Running test suite..."
pytest tests/ -v --tb=short -q 2>&1 | tail -20
echo "      Test suite complete."

# ── 4. Run CPU-only simulation pipeline ──────────────────────────────────
echo ""
if [[ $FULL -eq 1 ]]; then
    echo "[4/5] Running FULL baseline pipeline (30 rounds, 100 agents, seeds 42,123,7)..."
    python scripts/run_full_pipeline.py \
        --rounds 30 \
        --agents 100 \
        --seeds 42,123,7
else
    echo "[4/5] Running FAST baseline pipeline (5 rounds, 20 agents, seeds 42,123,7)..."
    echo "      (Use --full for the complete paper reproduction run)"
    python scripts/run_full_pipeline.py \
        --rounds 5 \
        --agents 20 \
        --seeds 42,123,7
fi

# ── 5. Trust gradient analysis ───────────────────────────────────────────
echo ""
echo "[5/5] Running trust-gradient sub-population validation..."
python scripts/run_trust_gradient.py || echo "      (Skipping — run after full experiments)"

echo ""
echo "======================================================="
echo "  Reproduction complete."
echo "  Figures: analysis/figures/"
echo "  Results: analysis/tables/"
echo "  Experiments: experiments/"
echo "======================================================="
