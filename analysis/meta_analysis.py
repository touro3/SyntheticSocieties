"""Random-effects meta-analysis pooling A/B grounding contrasts across all
available pilot data (CPU-only).

Sources combined
----------------
- analysis/tables/grounding_comparison_seed_metrics.csv  — primary LLM pilot
  (per-seed cooperation_rate and wealth_gini for `pure_llm_ess_persona` (A)
  vs `grounded_llm_ess_persona` (B)).
- analysis/cross_model_results.json                      — cross-model panel
  (Mistral-7B, Qwen2.5-7B, GPT-4o-mini at N=20, T=10).

Statistical model
-----------------
For each study (= one model family, one outcome variable), compute Hedges'
g_i and its variance v_i for the A vs B contrast. Pool across studies via
the DerSimonian-Laird (DL) random-effects estimator:
    Q     = Σ w_i (g_i − g_mean_fe)²       (heterogeneity)
    τ²    = max(0, (Q − k+1) / (Σ w_i − Σ w_i² / Σ w_i))
    w*_i  = 1 / (v_i + τ²)
    g_RE  = Σ w*_i g_i / Σ w*_i
    SE_RE = sqrt(1 / Σ w*_i)
    I²    = max(0, (Q − k+1) / Q) × 100%

Output
------
- analysis/tables/meta_analysis.json (per-study + pooled estimates)
- analysis/figures/meta_analysis_forest.png (one row per study)

Reads-only; no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
SEED_CSV = REPO / "analysis" / "tables" / "grounding_comparison_seed_metrics.csv"
CROSS_MODEL_JSON = REPO / "analysis" / "cross_model_results.json"
OUT_JSON = REPO / "analysis" / "tables" / "meta_analysis.json"
OUT_FIG = REPO / "analysis" / "figures" / "meta_analysis_forest.png"


def hedges_g(x_a: np.ndarray, x_b: np.ndarray) -> tuple[float, float]:
    """Hedges' g and its variance for a two-sample comparison.

    Returns (g, var_g). g > 0 means A > B (use --signed if you want B > A).
    """
    n_a, n_b = len(x_a), len(x_b)
    if n_a < 2 or n_b < 2:
        return float("nan"), float("nan")
    s_a, s_b = x_a.std(ddof=1), x_b.std(ddof=1)
    s_pooled = np.sqrt(((n_a - 1) * s_a**2 + (n_b - 1) * s_b**2) / (n_a + n_b - 2))
    if s_pooled == 0:
        return float("nan"), float("nan")
    d = (x_a.mean() - x_b.mean()) / s_pooled
    # Hedges' bias correction
    J = 1 - 3 / (4 * (n_a + n_b) - 9)
    g = J * d
    var_g = (n_a + n_b) / (n_a * n_b) + g**2 / (2 * (n_a + n_b))
    return float(g), float(var_g)


def dl_random_effects(gs: np.ndarray, vs: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects pooled estimate."""
    k = len(gs)
    if k < 2:
        g_p = float(gs[0]) if k == 1 else float("nan")
        se_p = float(np.sqrt(vs[0])) if k == 1 else float("nan")
        return {
            "k": k,
            "g_pooled": g_p,
            "se_pooled": se_p,
            "ci95_lo": g_p - 1.96 * se_p if k == 1 else float("nan"),
            "ci95_hi": g_p + 1.96 * se_p if k == 1 else float("nan"),
            "tau2": 0.0,
            "I2_pct": 0.0,
            "Q": 0.0,
            "df": 0,
            "z": g_p / se_p if k == 1 and se_p > 0 else float("nan"),
            "p_two_tailed": float("nan"),
            "g_fixed_effect": g_p,
        }
    w_fe = 1.0 / vs
    g_fe = float(np.sum(w_fe * gs) / np.sum(w_fe))
    Q = float(np.sum(w_fe * (gs - g_fe) ** 2))
    df = k - 1
    c = np.sum(w_fe) - np.sum(w_fe**2) / np.sum(w_fe)
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    w_re = 1.0 / (vs + tau2)
    g_re = float(np.sum(w_re * gs) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))
    I2 = float(max(0.0, (Q - df) / Q) * 100) if Q > 0 else 0.0
    z = g_re / se_re if se_re > 0 else float("nan")
    p_two = float(2 * (1 - stats.norm.cdf(abs(z)))) if not np.isnan(z) else float("nan")
    return {
        "k": k,
        "g_pooled": g_re,
        "se_pooled": se_re,
        "ci95_lo": g_re - 1.96 * se_re,
        "ci95_hi": g_re + 1.96 * se_re,
        "tau2": tau2,
        "I2_pct": I2,
        "Q": Q,
        "df": df,
        "z": z,
        "p_two_tailed": p_two,
        "g_fixed_effect": g_fe,
    }


def build_studies() -> list[dict]:
    """Construct one study per (data source, outcome). Each study has
    paired per-seed/per-run cooperation rates under A and B."""
    studies: list[dict] = []

    # ── Source 1: seed-level CSV ─────────────────────────────────────────
    df = pd.read_csv(SEED_CSV)
    # The primary contrast: pure_llm_ess_persona (A) vs grounded_llm_ess_persona (B)
    a = df[df["condition_key"] == "pure_llm_ess_persona"].sort_values("seed")
    b = df[df["condition_key"] == "grounded_llm_ess_persona"].sort_values("seed")
    if len(a) >= 2 and len(b) >= 2:
        for metric, label in [("cooperation_rate", "Coop rate"), ("wealth_gini", "Gini")]:
            x_a = a[metric].values.astype(float)
            x_b = b[metric].values.astype(float)
            g, v = hedges_g(x_a, x_b)
            studies.append(
                {
                    "study": f"Primary pilot — {label} (A − B)",
                    "source": "grounding_comparison_seed_metrics.csv",
                    "n_a": int(len(x_a)),
                    "n_b": int(len(x_b)),
                    "mean_a": float(x_a.mean()),
                    "mean_b": float(x_b.mean()),
                    "hedges_g": g,
                    "var_g": v,
                }
            )

    # ── Source 2: cross-model JSON ──────────────────────────────────────
    cm = json.loads(CROSS_MODEL_JSON.read_text())
    cm_df = pd.DataFrame(cm)
    for model in sorted(cm_df["model_id"].unique()):
        sub = cm_df[cm_df["model_id"] == model]
        for metric, label in [
            ("cooperation_rate", "Coop rate"),
            ("rlhf_bias_index", "B_RLHF"),
            ("gini", "Gini"),
        ]:
            ra = sub[sub["condition"] == "A"][metric].values.astype(float)
            rb = sub[sub["condition"] == "B"][metric].values.astype(float)
            if len(ra) < 2 or len(rb) < 2:
                continue
            g, v = hedges_g(ra, rb)
            studies.append(
                {
                    "study": f"Cross-model {model} — {label} (A − B)",
                    "source": "cross_model_results.json",
                    "n_a": int(len(ra)),
                    "n_b": int(len(rb)),
                    "mean_a": float(ra.mean()),
                    "mean_b": float(rb.mean()),
                    "hedges_g": g,
                    "var_g": v,
                }
            )
    return studies


def main() -> None:
    studies = build_studies()
    valid = [s for s in studies if not (np.isnan(s["hedges_g"]) or np.isnan(s["var_g"]))]
    print(f"Built {len(studies)} study rows ({len(valid)} usable for pooling).\n")

    # Split by outcome (coop vs Gini vs B_RLHF) — pool each separately.
    by_outcome: dict[str, list[dict]] = {}
    for s in valid:
        outcome = s["study"].split("—")[1].split("(")[0].strip()
        by_outcome.setdefault(outcome, []).append(s)

    pooled: dict[str, dict] = {}
    for outcome, rows in by_outcome.items():
        gs = np.array([r["hedges_g"] for r in rows])
        vs = np.array([r["var_g"] for r in rows])
        pooled[outcome] = dl_random_effects(gs, vs)
        pooled[outcome]["studies"] = [r["study"] for r in rows]
        p = pooled[outcome]
        print(
            f"  {outcome:<12}  k={p['k']}  g={p['g_pooled']:+.3f}  "
            f"95% CI=[{p['ci95_lo']:+.3f}, {p['ci95_hi']:+.3f}]  "
            f"τ²={p['tau2']:.3f}  I²={p['I2_pct']:.1f}%  p={p['p_two_tailed']:.4f}"
        )

    out = {
        "method": "Hedges' g + DerSimonian-Laird random-effects pooling",
        "sign_convention": "Positive g means Condition A > Condition B on the named metric.",
        "studies": studies,
        "pooled_by_outcome": pooled,
        "audit_row": "B.meta (new)",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n✓ {OUT_JSON}")

    # ── Forest figure ────────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9.5, max(4, 0.4 * len(valid) + 2)))
        labels = [s["study"] for s in valid]
        gs = np.array([s["hedges_g"] for s in valid])
        ses = np.sqrt([s["var_g"] for s in valid])
        ypos = np.arange(len(labels))[::-1]
        ax.errorbar(gs, ypos, xerr=1.96 * ses, fmt="o", color="#1e6091", capsize=3)
        ax.axvline(0, ls="--", color="#777", lw=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Hedges' g  (positive = Condition A > Condition B)")
        ax.set_title("Random-Effects Meta-Analysis of A vs B Grounding Contrasts")
        # Pooled diamonds at the bottom
        bottom = -1
        for outcome, p in pooled.items():
            if np.isnan(p["g_pooled"]):
                continue
            ax.errorbar([p["g_pooled"]], [bottom], xerr=1.96 * p["se_pooled"], fmt="D", color="#c1121f", capsize=4)
            ax.text(
                p["g_pooled"],
                bottom - 0.15,
                f"Pooled {outcome}: g={p['g_pooled']:+.3f} "
                f"[{p['ci95_lo']:+.2f}, {p['ci95_hi']:+.2f}], I²={p['I2_pct']:.0f}%",
                ha="center",
                va="top",
                fontsize=7,
                color="#c1121f",
            )
            bottom -= 1.2
        ax.set_ylim(bottom - 1, max(ypos) + 1)
        fig.tight_layout()
        OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_FIG, dpi=150)
        print(f"✓ {OUT_FIG}")
    except Exception as exc:
        print(f"figure skipped: {exc}")


if __name__ == "__main__":
    main()
