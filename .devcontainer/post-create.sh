#!/usr/bin/env bash
# Post-create bootstrap for the BGF VSCode dev container.
# Runs once after the container is built. Idempotent.
set -euo pipefail

cd /workspaces/SyntheticSocieties || cd "$(dirname "$0")/.."

echo "── BGF dev container post-create ──"
python --version
pip --version

# CI-only requirement set is the fastest install path (CPU, no torch).
# Use this for editor / lint / test work; switch to requirements-api.txt
# only when you actually need to run the Flask API locally.
pip install --upgrade pip
pip install -r requirements-ci.txt
pip install -e . --no-deps || true

# Pre-commit hooks
if [ -f .pre-commit-config.yaml ]; then
  pip install pre-commit
  pre-commit install --install-hooks || true
fi

# Smoke test — non-fatal, just confirms wiring
python - <<'PY' || true
import importlib
for m in ("agents.agent", "decision.llm_policy", "simulation.kernel", "metrics.behavioral_realism"):
    importlib.import_module(m)
    print(f"  ok: {m}")
PY

echo "── Ready. Try: make help ──"
