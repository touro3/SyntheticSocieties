"""
Cooperation Formula Validation (Phase 29, Section 6).

Tests the BGF hand-crafted formula:
    E[coop] = 0.2 + 0.6 * trust * (1 - risk)

against empirical data (Prolific responses or synthetic fallback).

Fits a logistic regression: coop_binary ~ trust + risk + trust:risk
and compares fitted coefficients against the hand-crafted formula.

Usage:
    # Dry-run with synthetic data (no Prolific CSV needed):
    python scripts/validate_cooperation_formula.py --synthetic

    # Real Prolific data:
    python scripts/validate_cooperation_formula.py \
        --input-csv data/human/responses.csv

    # Custom output:
    python scripts/validate_cooperation_formula.py --synthetic \
        --out-json analysis/tables/formula_validation.json \
        --out-plot analysis/figures/formula_validation.png
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

# ── Hand-crafted formula coefficients ────────────────────────────────────────

FORMULA_INTERCEPT = 0.20
FORMULA_TRUST_COEF = 0.60
FORMULA_RISK_COEF = -0.60  # E[coop] decreases as risk increases


# ── Synthetic data ────────────────────────────────────────────────────────────


def generate_synthetic(n: int = 50, n_rounds: int = 10, seed: int = 42) -> tuple[list[float], list[float], list[float]]:
    """Generate (trust, risk, coop_rate) for N synthetic participants.

    Ground truth follows the BGF formula exactly (with Gaussian noise).
    Returns lists: trusts, risks, coop_rates (one entry per participant).
    """
    rng = random.Random(seed)
    trusts, risks, coops = [], [], []
    for _ in range(n):
        t = rng.uniform(0.05, 0.95)
        r = rng.uniform(0.05, 0.95)
        p_coop = max(0.0, min(1.0, FORMULA_INTERCEPT + FORMULA_TRUST_COEF * t * (1 - r)))
        # Simulate binomial draws (n_rounds rounds per participant)
        c_count = sum(1 for _ in range(n_rounds) if rng.random() < p_coop)
        trusts.append(t)
        risks.append(r)
        coops.append(c_count / n_rounds)
    return trusts, risks, coops


def load_from_csv(path: Path) -> tuple[list[float], list[float], list[float]]:
    """Load (pre_trust, pre_risk, coop_rate) from the Prolific responses CSV.

    Requires columns: participant_id, round_id, action, pre_trust, pre_risk.
    """
    import csv

    rows: dict[str, dict] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["participant_id"]
            if pid not in rows:
                rows[pid] = {
                    "pre_trust": float(row.get("pre_trust", 0.5)),
                    "pre_risk": float(row.get("pre_risk", 0.5)),
                    "n_coop": 0,
                    "n_total": 0,
                }
            rows[pid]["n_total"] += 1
            if row.get("action", "").lower() == "cooperate":
                rows[pid]["n_coop"] += 1

    trusts, risks, coops = [], [], []
    for p in rows.values():
        if p["n_total"] == 0:
            continue
        trusts.append(p["pre_trust"])
        risks.append(p["pre_risk"])
        coops.append(p["n_coop"] / p["n_total"])
    return trusts, risks, coops


# ── OLS / logistic helpers (pure Python, no sklearn required) ─────────────────


def _design_matrix(trusts: list[float], risks: list[float]) -> list[list[float]]:
    """Build [1, trust, risk, trust*risk] design matrix rows."""
    return [[1.0, t, r, t * r] for t, r in zip(trusts, risks)]


def _ols(X: list[list[float]], y: list[float]) -> list[float]:
    """OLS via normal equations: β = (X'X)^{-1} X'y."""
    n, p = len(X), len(X[0])
    # X'X
    XtX = [[sum(X[i][j] * X[i][k] for i in range(n)) for k in range(p)] for j in range(p)]
    # X'y
    Xty = [sum(X[i][j] * y[i] for i in range(n)) for j in range(p)]
    # Gaussian elimination with partial pivoting
    aug = [XtX[j] + [Xty[j]] for j in range(p)]
    for col in range(p):
        # Pivot
        max_row = max(range(col, p), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            continue
        for row in range(p):
            if row != col:
                factor = aug[row][col] / pivot
                aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(p + 1)]
        aug[col] = [v / pivot for v in aug[col]]
    return [aug[j][p] for j in range(p)]


def _r_squared(y: list[float], y_hat: list[float]) -> float:
    y_bar = sum(y) / len(y)
    ss_res = sum((yi - yhi) ** 2 for yi, yhi in zip(y, y_hat))
    ss_tot = sum((yi - y_bar) ** 2 for yi in y)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _bootstrap_ci(X: list[list[float]], y: list[float], n_boot: int = 500, seed: int = 0) -> list[tuple[float, float]]:
    """Bootstrap 95% CIs for OLS coefficients."""
    rng = random.Random(seed)
    n = len(y)
    p = len(X[0])
    coef_samples: list[list[float]] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        Xs = [X[i] for i in idx]
        ys = [y[i] for i in idx]
        try:
            coef_samples.append(_ols(Xs, ys))
        except Exception:
            pass

    cis = []
    for j in range(p):
        vals = sorted(c[j] for c in coef_samples)
        lo = vals[int(0.025 * len(vals))]
        hi = vals[int(0.975 * len(vals))]
        cis.append((round(lo, 4), round(hi, 4)))
    return cis


# ── Plotting ──────────────────────────────────────────────────────────────────


def _plot(trusts: list[float], coops: list[float], y_hat: list[float], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    formula_pred = [
        FORMULA_INTERCEPT + FORMULA_TRUST_COEF * t * 0.5  # fix risk=0.5 for 1-D slice
        for t in trusts
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: predicted vs observed scatter
    ax = axes[0]
    ax.scatter(y_hat, coops, alpha=0.6, s=40, color="#4299e1", label="OLS fit")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect fit")
    ax.set_xlabel("Predicted cooperation rate (OLS)")
    ax.set_ylabel("Observed cooperation rate")
    ax.set_title("Predicted vs Observed")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: trust vs coop scatter with formula overlay (at risk=0.5)
    ax = axes[1]
    ax.scatter(trusts, coops, alpha=0.5, s=30, color="#68d391", label="Observed")
    t_grid = [i / 100 for i in range(101)]
    f_line = [FORMULA_INTERCEPT + FORMULA_TRUST_COEF * t * 0.5 for t in t_grid]
    ax.plot(t_grid, f_line, "r-", lw=2, label="Formula (risk=0.5)")
    ax.set_xlabel("Pre-survey trust (normalised)")
    ax.set_ylabel("Cooperation rate")
    ax.set_title("Trust → Cooperation (at risk=0.5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate E[coop|profile] formula.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (no CSV required).")
    parser.add_argument(
        "--input-csv",
        default="data/human/responses.csv",
        help="Prolific responses CSV (requires pre_trust, pre_risk columns).",
    )
    parser.add_argument("--n-synthetic", type=int, default=50, help="Number of synthetic participants (default 50).")
    parser.add_argument("--n-bootstrap", type=int, default=500, help="Bootstrap samples for CIs (default 500).")
    parser.add_argument("--out-json", default="analysis/tables/formula_validation.json")
    parser.add_argument("--out-plot", default="analysis/figures/formula_validation.png")
    args = parser.parse_args()

    if args.synthetic:
        print(f"Generating {args.n_synthetic} synthetic participants...")
        trusts, risks, coops = generate_synthetic(n=args.n_synthetic)
        data_source = "synthetic"
    else:
        csv_path = Path(args.input_csv)
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}. Use --synthetic for offline testing.")
            return
        trusts, risks, coops = load_from_csv(csv_path)
        data_source = str(csv_path)

    n = len(trusts)
    print(f"Fitting model on {n} participants (source: {data_source})")

    X = _design_matrix(trusts, risks)
    beta = _ols(X, coops)
    y_hat = [sum(beta[j] * X[i][j] for j in range(4)) for i in range(n)]
    r2 = _r_squared(coops, y_hat)

    cis = _bootstrap_ci(X, coops, n_boot=args.n_bootstrap)

    # AIC (approximate, OLS): n * log(RSS/n) + 2*k
    rss = sum((yi - yhi) ** 2 for yi, yhi in zip(coops, y_hat))
    k = 4  # intercept + 3 predictors
    aic = n * math.log(rss / n + 1e-12) + 2 * k if rss > 0 else float("nan")

    labels = ["intercept", "trust", "risk", "trust:risk"]
    formula_vals = [FORMULA_INTERCEPT, FORMULA_TRUST_COEF, FORMULA_RISK_COEF, 0.0]

    print("\n=== Cooperation Formula Validation ===")
    print(f"{'Coef':<12} {'Fitted':>8} {'95% CI':<18} {'Formula':>8} {'Match?':>6}")
    print("-" * 58)
    matches = []
    coef_results = []
    for label, fitted, ci, formula in zip(labels, beta, cis, formula_vals):
        in_ci = ci[0] <= formula <= ci[1]
        matches.append(in_ci)
        flag = "✓" if in_ci else "✗"
        print(f"{label:<12} {fitted:>8.4f} [{ci[0]:>7.4f}, {ci[1]:>7.4f}]  {formula:>7.4f}  {flag:>6}")
        coef_results.append(
            {
                "name": label,
                "fitted": round(fitted, 4),
                "ci_lo": ci[0],
                "ci_hi": ci[1],
                "formula": formula,
                "formula_within_ci": in_ci,
            }
        )

    print(f"\nR² = {r2:.4f}   AIC ≈ {aic:.2f}   Formula-within-CI: {sum(matches)}/{len(matches)}")

    result = {
        "data_source": data_source,
        "n_participants": n,
        "r_squared": round(r2, 4),
        "aic": round(aic, 2),
        "coefficients": coef_results,
        "formula": "E[coop] = 0.20 + 0.60 * trust * (1 - risk)",
        "formula_validated": all(matches),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out_json}")

    _plot(trusts, coops, y_hat, Path(args.out_plot))


if __name__ == "__main__":
    main()
