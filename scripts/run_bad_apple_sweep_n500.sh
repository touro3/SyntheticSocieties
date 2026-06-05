#!/usr/bin/env bash
# =============================================================================
# run_bad_apple_sweep_n500.sh — Bad-apple phase transition at primary scale
#
# Re-runs the bad-apple sweep at N=500, T=30 (rule-based policy, no GPU needed)
# to determine the phase transition inflection f* at primary scale and verify
# whether f* shifts from ≈0.02 (N=20 pilot) toward the 10–20% range predicted
# by evolutionary game theory (Nowak & May, 1992).
#
# Paper claim (pending): f* at N=500 may differ substantially from N=20.
# Existing data: analysis/bad_apple_sweep.json is from N=20, T=20 pilot.
#
# Requirements: CPU only (rule_based policy), ~30–60 min
#
# Usage:
#   source venv/bin/activate
#   bash scripts/run_bad_apple_sweep_n500.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

FRACTIONS=(0.00 0.02 0.05 0.08 0.10 0.12 0.15 0.18 0.20 0.25 0.30 0.35 0.40)
SEEDS=(42 123 7)
N_AGENTS=500
N_ROUNDS=30
RESULTS_FILE="analysis/bad_apple_sweep_n500.json"

echo "========================================================"
echo "  BGF Bad-Apple Phase Transition Sweep — N=500, T=30"
echo "  Fractions: ${FRACTIONS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  $(date)"
echo "========================================================"

RESULTS="[]"

for frac in "${FRACTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        exp_id="bad_apple_n500_f${frac/./}_s${seed}"
        echo "── f=${frac} | seed=${seed} → ${exp_id}"

        venv/bin/python scripts/run_full_pipeline.py \
            --agents "$N_AGENTS" \
            --rounds "$N_ROUNDS" \
            --seeds "$seed" \
            --bad-apple-frac "$frac" \
            --policy rule_based \
            --population.source empirical \
            --exp-id "$exp_id" \
            2>&1 | tail -3
    done
done

echo ""
echo "── Fitting sigmoid and computing f*..."
venv/bin/python - <<'EOF'
import json, glob, numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

fracs  = [0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]
seeds  = [42, 123, 7]
per_frac = []

for frac in fracs:
    gini_vals = []
    for seed in seeds:
        tag = f"bad_apple_n500_f{str(frac).replace('.','')}_s{seed}"
        p = Path(f"experiments/{tag}/metrics.json")
        if p.exists():
            m = json.loads(p.read_text())
            gini = m.get("gini", m.get("gini_final", None))
            if gini is not None:
                gini_vals.append(float(gini))
    if gini_vals:
        per_frac.append({
            "bad_fraction": frac,
            "gini_mean": float(np.mean(gini_vals)),
            "gini_std":  float(np.std(gini_vals)),
            "n_seeds":   len(gini_vals),
        })

xs = [d["bad_fraction"] for d in per_frac]
ys = [d["gini_mean"]    for d in per_frac]

def sigmoid(x, L, k, x0, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

fit_result = {}
try:
    popt, _ = curve_fit(sigmoid, xs, ys, p0=[0.15, 15, 0.15, ys[0]], maxfev=20000,
                        bounds=([0, 0, 0, 0], [1, 200, 1, 1]))
    L, k, x0, b = popt
    yhat  = sigmoid(np.array(xs), *popt)
    ss_res = sum((np.array(ys) - yhat)**2)
    ss_tot = sum((np.array(ys) - np.mean(ys))**2)
    r2 = 1 - ss_res/ss_tot
    fit_result = {"f_star": round(x0, 4), "k": round(k, 2), "L": round(L, 4), "b": round(b, 4), "r2": round(r2, 4)}
    print(f"  Sigmoid fit: f*={x0:.4f}, k={k:.2f}, R²={r2:.4f}")
except Exception as e:
    print(f"  Sigmoid fit failed: {e}")

out = {"n_agents": 500, "n_rounds": 30, "per_fraction": per_frac, "sigmoid_fit": fit_result}
Path("analysis/bad_apple_sweep_n500.json").write_text(json.dumps(out, indent=2))
print(f"  Saved → analysis/bad_apple_sweep_n500.json")
EOF

echo ""
echo "========================================================"
echo "  Sweep complete. Update paper Section 6.4 with f* from"
echo "  analysis/bad_apple_sweep_n500.json"
echo "========================================================"
