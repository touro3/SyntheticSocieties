#!/usr/bin/env bash
# =============================================================================
# Paper reproduction entrypoint.
#
# Pins PYTHONHASHSEED so that any dict/set iteration order or hash()-derived
# value is byte-identical across machines and Python processes — a hard
# requirement for the "identical seed → identical result" reproducibility
# guarantee. Delegates the actual pipeline to scripts/run_all_experiments.sh.
# =============================================================================

set -euo pipefail

export PYTHONHASHSEED=0

cd "$(dirname "$0")"

exec bash scripts/run_all_experiments.sh "$@"
