#!/usr/bin/env python3
"""Confirmatory analytics for the pre-registered 10-seed N=500 A/B scale-up.

Press-play pipeline: run *after* ``scripts/launch_gpu_ab.sh`` finishes.
It reads the experiment registry (no run replay), computes the
confirmatory statistics the paper §8.1 promises, and writes both a
machine-readable JSON and the reserved figures.

What it produces
----------------
1. Per-condition point estimates + **BCa bootstrap 95% CIs** for
   cooperation rate, wealth Gini, and B_RLHF (10 000 resamples).
2. Per-metric **Mann-Whitney U** (primary, distribution-free) and Welch
   t as a secondary check, A vs B.
3. **Benjamini-Hochberg FDR** correction across the metric family
   (paper pre-registers BH-FDR over H1-H8; here applied to the metric
   panel actually computable from the registry).
4. Figures:
   - ``analysis/figures/ten_seed_ab_ci.png`` — grouped means + 95% CI.
   - ``analysis/figures/ten_seed_seed_strip.png`` — per-seed strip/forest.
5. ``analysis/tables/ten_seed_confirmatory.json`` — full results, with an
   explicit ``status`` field and an honest ``brm_note`` (BRM_composite
   robustness is the deterministic Theorem 2 certificate, not a sampled
   CI; see analysis/brm_sensitivity.py --emit-certificate).

Design choices
--------------
* Distribution-free primary test: at 10 seeds/arm normality is not
  assured; Mann-Whitney U is the pre-registered primary, Welch t is
  reported as a sensitivity check, never as the headline.
* BCa (bias-corrected accelerated) bootstrap rather than percentile:
  small-n CIs need the bias/skew correction.
* Runs gracefully on an empty registry (no runs yet) — it reports
  ``status="awaiting_runs"`` and exits 0, so it is safe to wire into CI
  now and it simply activates when the seeds land.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "tracker" / "experiment_index.parquet"
SQL_FILE = REPO / "analysis" / "ten_seed_aggregate.sql"
FIG_DIR = REPO / "analysis" / "figures"
TBL_DIR = REPO / "analysis" / "tables"
OUT_JSON = TBL_DIR / "ten_seed_confirmatory.json"

POP_SIZE = 500
HORIZON = 30
N_BOOT = 10_000
RNG = np.random.default_rng(42)  # deterministic; matches paper seed convention

METRICS = {
    "cooperation_rate": "Cooperation rate",
    "wealth_gini": "Wealth Gini",
    "b_rlhf": "B_RLHF (TV vs uniform)",
}


def load_per_run() -> dict[str, dict[str, np.ndarray]]:
    """Return {condition: {metric: array-over-seeds}} from the registry.

    Uses the shared ten_seed_aggregate.sql so SQL stays single-sourced.
    """
    import duckdb

    if not REGISTRY.exists():
        return {}

    con = duckdb.connect()
    con.execute(f"SET variable registry_path = '{REGISTRY.as_posix()}';")
    con.execute(f"SET variable pop_size = {POP_SIZE};")
    con.execute(f"SET variable horizon = {HORIZON};")
    # The first statement in the .sql is the per_run SELECT; read it back.
    sql = SQL_FILE.read_text()
    per_run_select = sql.split(";")[0]  # 'SET variable ...' lines are 3 stmts
    # Re-extract just the final SELECT * FROM per_run block robustly:
    marker = "SELECT * FROM per_run"
    body = sql[sql.index("WITH runs AS") : sql.index(marker) + len(marker)]
    body += " ORDER BY condition, seed"
    rows = con.execute(body).fetchall()
    cols = [d[0] for d in con.description]
    con.close()

    idx = {c: i for i, c in enumerate(cols)}
    out: dict[str, dict[str, list]] = {}
    for r in rows:
        cond = r[idx["condition"]]
        d = out.setdefault(cond, {m: [] for m in METRICS})
        for m in METRICS:
            v = r[idx[m]]
            if v is not None:
                d[m].append(float(v))
    return {c: {m: np.asarray(v, float) for m, v in md.items()} for c, md in out.items()}


def bca_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Bias-corrected accelerated bootstrap CI for the mean."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    boot = RNG.choice(x, size=(N_BOOT, n), replace=True).mean(axis=1)
    theta = x.mean()
    # bias-correction z0
    prop = np.mean(boot < theta)
    prop = min(max(prop, 1.0 / N_BOOT), 1 - 1.0 / N_BOOT)
    z0 = _norm_ppf(prop)
    # acceleration via jackknife
    jack = np.array([np.delete(x, i).mean() for i in range(n)])
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    zl, zu = _norm_ppf(alpha / 2), _norm_ppf(1 - alpha / 2)
    a1 = _norm_cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    a2 = _norm_cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
    lo = np.quantile(boot, max(0.0, min(1.0, a1)))
    hi = np.quantile(boot, max(0.0, min(1.0, a2)))
    return float(lo), float(hi)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation; adequate for CI endpoints.
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Mann-Whitney U p-value (SciPy if available, else normal approx)."""
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        a, b = np.asarray(a), np.asarray(b)
        n1, n2 = len(a), len(b)
        allv = np.concatenate([a, b])
        order = allv.argsort()
        ranks = np.empty(len(allv))
        ranks[order] = np.arange(1, len(allv) + 1)
        r1 = ranks[:n1].sum()
        u1 = r1 - n1 * (n1 + 1) / 2
        u = min(u1, n1 * n2 - u1)
        mu = n1 * n2 / 2
        sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        if sigma == 0:
            return 1.0
        z = (u - mu) / sigma
        return float(2 * (1 - _norm_cdf(abs(z))))


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import ttest_ind
        return float(ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        a, b = np.asarray(a), np.asarray(b)
        va, vb = a.var(ddof=1), b.var(ddof=1)
        na, nb = len(a), len(b)
        se = math.sqrt(va / na + vb / nb)
        if se == 0:
            return 1.0
        t = (a.mean() - b.mean()) / se
        df = se**4 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
        # survival of |t| via normal approx (df>=~9 here)
        return float(2 * (1 - _norm_cdf(abs(t))))


def benjamini_hochberg(pvals: dict[str, float], q: float = 0.05) -> dict[str, dict]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, max_sig = {}, -1
    for i, (k, p) in enumerate(items, 1):
        thresh = i / m * q
        if p <= thresh:
            max_sig = i
    for i, (k, p) in enumerate(items, 1):
        out[k] = {"p": p, "bh_threshold": i / m * q, "reject_h0": i <= max_sig}
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return float((b.mean() - a.mean()) / sp) if sp else float("nan")


def make_figures(data: dict, results: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    conds = sorted(data)  # ['A_ungrounded', 'B_grounded']
    metrics = list(METRICS)

    # --- Figure 1: grouped means with 95% BCa CI -------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    xs = np.arange(len(metrics))
    for ci_i, cond in enumerate(conds):
        means = [results["per_condition"][cond][m]["mean"] for m in metrics]
        los = [results["per_condition"][cond][m]["ci95"][0] for m in metrics]
        his = [results["per_condition"][cond][m]["ci95"][1] for m in metrics]
        err = [np.array(means) - np.array(los), np.array(his) - np.array(means)]
        ax.bar(xs + (ci_i - 0.5) * width, means, width, label=cond,
               yerr=err, capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([METRICS[m] for m in metrics])
    ax.set_ylabel("Value")
    ax.set_title(f"10-seed N={POP_SIZE} T={HORIZON} confirmatory A/B\n"
                 "(error bars = BCa bootstrap 95% CI, 10k resamples)")
    ax.legend()
    fig.tight_layout()
    p1 = FIG_DIR / "ten_seed_ab_ci.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    written.append(str(p1))

    # --- Figure 2: per-seed strip --------------------------------------
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4))
    for ax, m in zip(axes, metrics):
        for ci_i, cond in enumerate(conds):
            y = data[cond][m]
            x = np.full_like(y, ci_i, dtype=float) + RNG.normal(0, 0.04, len(y))
            ax.scatter(x, y, alpha=0.8, label=cond if ax is axes[0] else None)
            ax.hlines(y.mean(), ci_i - 0.2, ci_i + 0.2, color="k")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=15, ha="right")
        ax.set_title(METRICS[m])
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Per-seed values (n=10/arm) with condition means")
    fig.tight_layout()
    p2 = FIG_DIR / "ten_seed_seed_strip.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    written.append(str(p2))
    return written


def main() -> int:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    data = load_per_run()

    have = {c: len(next(iter(v.values()), [])) for c, v in data.items()}
    if len(data) < 2 or min(have.values(), default=0) == 0:
        payload = {
            "status": "awaiting_runs",
            "message": ("No mx_A_s* / mx_B_s* runs at N=500 T=30 in the "
                        "registry yet. Launch scripts/launch_gpu_ab.sh, "
                        "then re-run this script — it is press-play."),
            "seeds_found": have,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return 0

    results: dict = {"status": "complete", "n_per_arm": have,
                     "per_condition": {}, "contrasts": {}}
    a_key = next(k for k in data if k.startswith("A"))
    b_key = next(k for k in data if k.startswith("B"))

    for cond, md in data.items():
        results["per_condition"][cond] = {}
        for m, arr in md.items():
            lo, hi = bca_ci(arr)
            results["per_condition"][cond][m] = {
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "n": int(len(arr)),
                "ci95": [lo, hi],
            }

    pvals = {}
    for m in METRICS:
        a, b = data[a_key][m], data[b_key][m]
        p_mw = mann_whitney_u(a, b)
        pvals[m] = p_mw
        results["contrasts"][m] = {
            "delta_B_minus_A": float(b.mean() - a.mean()),
            "mannwhitney_p": p_mw,
            "welch_p_sensitivity": welch_t(a, b),
            "cohens_d": cohens_d(a, b),
        }

    bh = benjamini_hochberg(pvals, q=0.05)
    for m, info in bh.items():
        results["contrasts"][m]["bh_fdr"] = info

    results["brm_note"] = (
        "BRM_composite weight-robustness is the DETERMINISTIC Theorem 2 "
        "certificate (min_j Δ_j > 0), not a sampled CI. Emit it via "
        "`python analysis/brm_sensitivity.py --emit-certificate`. This "
        "script intentionally reports only registry-derivable metrics "
        "(coop rate, Gini, B_RLHF) so the press-play path needs no run "
        "replay; absolute BRM per run, if needed, is joined separately "
        "from metrics/ artifacts."
    )
    results["primary_test"] = "Mann-Whitney U (two-sided), BH-FDR q=0.05"

    figs = make_figures(data, results)
    results["figures"] = figs
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in
                       ("status", "n_per_arm", "contrasts")}, indent=2))
    print(f"\nWrote {OUT_JSON}")
    for f in figs:
        print(f"Wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
