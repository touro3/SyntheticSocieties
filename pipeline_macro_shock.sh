#!/bin/bash
set -e
RUN_DIR=$(ls -td artifacts/persona_fidelity/persona_fidelity_* 2>/dev/null | head -n 1)
python scripts/run_macro_shock.py --artifact-dir "$RUN_DIR" --pop-size 500 --rounds 30
python scripts/plot_macro_shock.py
echo "✅ MACRO SHOCK PIPELINE COMPLETED!"
