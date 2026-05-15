#!/usr/bin/env bash
# Fast CI pre-flight contract check for the ruflo-inspired additions.
# Exercises: new unit tests, a tiny mock simulation, witness write+verify,
# and the structured output formatter.  Designed to finish in well under a
# minute with no GPU.
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate || true

echo "==> 1/4 unit tests (new modules)"
python -m pytest -q --no-cov \
  tests/test_persistent_memory.py \
  tests/test_witness.py \
  tests/test_trajectory_bank.py \
  tests/test_regression_detection.py \
  tests/test_sweep_state.py

echo "==> 2/4 tiny mock simulation"
SMOKE_EXP="smoke_$(date +%s)"
python scripts/run_config_simulation.py configs/base_config.yaml \
  "project.experiment_id=${SMOKE_EXP}" \
  "policy.type=mock" \
  "simulation.rounds=2" \
  "simulation.population_size=4" >/dev/null
test -f "experiments/${SMOKE_EXP}/summary.json"

echo "==> 3/4 reproducibility witness verify"
python scripts/verify_witness.py "experiments/${SMOKE_EXP}"

echo "==> 4/4 structured output formatter"
python - <<'PY'
from utils.output import OutputFormatter
import io, json
buf = io.StringIO()
OutputFormatter("json", buf).table([{"a": 1, "b": 2}])
json.loads(buf.getvalue())  # raises if not valid JSON
print("formatter OK")
PY

# Best-effort: regression query only if a tracker index exists.
if [ -f tracker/experiment_index.parquet ]; then
  echo "==> (bonus) regression query"
  python -c "from tracker.analytics import detect_regression; print(len(detect_regression()))"
fi

rm -rf "experiments/${SMOKE_EXP}"
echo "SMOKE OK"
