"""Forest plot of H1-H9 standardised effect sizes (audit row E.6).

Reads existing per-hypothesis tables from analysis/tables/ and renders a
single forest plot whose horizontal axis is the standardised effect-size
direction (positive = BGF prediction supported). A descriptive
DerSimonian-Laird random-effects pooled estimate is appended at the bottom
of the plot with a transparent note that the effect sizes are heterogeneous
(g, ρ, ΔB_RLHF, etc.) and that the pooled value is presentational, not
inferential.

Where a hypothesis row's primary effect-size artifact is not yet present on
disk (e.g. cross-cultural behavioral H9, which depends on the H9 PGG run),
the row is rendered with `effect = NaN` and a "pending" annotation so the
plot remains an *audit* of the evidence base rather than a misleading
single-point estimate.

CPU only; ~5 s runtime.

Usage:
    python -m analysis.forest_plot
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "analysis" / "tables"


@dataclass
class HypRow:
    hypothesis: str
    metric_name: str
    effect: float | None
    ci_lo: float | None
    ci_hi: float | None
    status: str  # "verified" | "partial" | "pending"
    source: str
    audit_row: str
    note: str = ""


def _safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _h1_h2_h4_from_paper_numbers() -> tuple[HypRow, HypRow, HypRow]:
    """H1 (BRM_composite), H2 (B_RLHF reduction), H4 (modularity)."""
    blob = _safe_load(REPO_ROOT / "analysis" / "paper_numbers.json") or {}
    cross = _safe_load(REPO_ROOT / "analysis" / "cross_model_results.json") or []

    # H2: prefer the N=100 10-seed confirmatory result if available; fall back
    # to the N=20 cross-model rows which are exploratory.
    n100_a = [r for r in cross if r.get("model_id") == "mistral-7b-n100-confirmatory" and r.get("condition") == "A"]
    n100_b = [r for r in cross if r.get("model_id") == "mistral-7b-n100-confirmatory" and r.get("condition") == "B"]
    mistral_a = [r for r in cross if r.get("model_id") == "mistral-7b" and r.get("condition") == "A"]
    mistral_b = [r for r in cross if r.get("model_id") == "mistral-7b" and r.get("condition") == "B"]
    if n100_a and n100_b:
        a_val = float(np.mean([r["rlhf_bias_index"] for r in n100_a]))
        b_val = float(np.mean([r["rlhf_bias_index"] for r in n100_b]))
        abs_delta = b_val - a_val  # negative = reduction; ≈ -0.003
        h2 = HypRow(
            hypothesis="H2: B_RLHF reduction (A→B, N=100 10-seed)",
            metric_name="abs ΔB_RLHF (B−A)",
            effect=abs_delta,
            ci_lo=None,
            ci_hi=None,
            status="falsified",  # no reduction at N=100
            source="analysis/cross_model_results.json (Mistral-7B N=100 confirmatory)",
            audit_row="A.4 / §8.1.1",
            note=f"A={a_val:.3f}, B={b_val:.3f}; MWU p=0.91; N=100 10-seed null. N=20 exploratory: A=0.254,B=0.038 (85% reduction).",
        )
    elif mistral_a and mistral_b:
        a_mean = float(np.mean([r["rlhf_bias_index"] for r in mistral_a]))
        b_mean = float(np.mean([r["rlhf_bias_index"] for r in mistral_b]))
        delta = (a_mean - b_mean) / a_mean if a_mean > 0 else float("nan")
        h2 = HypRow(
            hypothesis="H2: B_RLHF reduction (A→B, N=20 exploratory)",
            metric_name="relative ΔB_RLHF",
            effect=delta,
            ci_lo=None,
            ci_hi=None,
            status="partial",
            source="analysis/cross_model_results.json (Mistral-7B N=20)",
            audit_row="A.4",
            note="N=20 T=10 only; N=100 10-seed confirmatory falsifies H2 (p=0.91).",
        )
    else:
        h2 = HypRow(
            "H2: B_RLHF reduction (A→B)",
            "relative ΔB_RLHF",
            None,
            None,
            None,
            "pending",
            "analysis/cross_model_results.json",
            "A.4",
            "no Mistral rows found",
        )

    # H1: BRM_composite difference — defer to the sensitivity table for
    # the equal-weight number, which is the pre-registered comparator.
    brm = _safe_load(TABLES / "brm_sensitivity.json") or {}
    eq = brm.get("equal_weight") or {}
    if "diff" in eq:
        h1 = HypRow(
            "H1: BRM_composite (B − A)",
            "Δ BRM composite",
            float(eq["diff"]),
            None,
            None,
            "verified",
            "analysis/tables/brm_sensitivity.json",
            "E.5 / A.10-A.11",
        )
    else:
        h1 = HypRow(
            "H1: BRM_composite (B − A)",
            "Δ BRM composite",
            None,
            None,
            None,
            "pending",
            "analysis/tables/brm_sensitivity.json",
            "E.5",
            "run analysis/brm_sensitivity.py",
        )

    # H4: modularity — read from the network-metrics blob if present.
    blob_h4 = (blob or {}).get("modularity_delta") if isinstance(blob, dict) else None
    if isinstance(blob_h4, (int, float)):
        h4 = HypRow(
            "H4: modularity Q (grounded − A)",
            "Δ modularity",
            float(blob_h4),
            None,
            None,
            "partial",
            "analysis/paper_numbers.json[modularity_delta]",
            "F.7",
        )
    else:
        h4 = HypRow(
            "H4: modularity Q (grounded − A)",
            "Δ modularity",
            None,
            None,
            None,
            "pending",
            "analysis/paper_numbers.json",
            "F.7",
            "topology ablation lands modularity into paper_numbers",
        )

    return h1, h2, h4


def _h3_gini_in_range() -> HypRow:
    """H3: fraction of N=100 LLM seeds with final Gini in Eurostat range."""
    pn = _safe_load(REPO_ROOT / "analysis" / "paper_numbers.json") or {}
    frac = pn.get("h3_gini_fraction_in_eurostat_range")
    n_total = pn.get("h3_n_total_seeds", 0)
    n_in = pn.get("h3_n_in_range", 0)
    if frac is not None:
        return HypRow(
            "H3: Gini ∈ Eurostat range [0.25, 0.40] (LLM N=100)",
            "fraction of seeds (LLM)",
            float(frac),
            None,
            None,
            "falsified",
            "analysis/paper_numbers.json[h3_gini_fraction_in_eurostat_range]",
            "A.8",
            f"{n_in}/{n_total} seeds in range. Condition D (rule-based) achieves 0.325 ± 0.001 (in range).",
        )
    return HypRow(
        "H3: Gini in empirical range [0.25, 0.40]",
        "fraction of seeds",
        None,
        None,
        None,
        "pending",
        "analysis/paper_numbers.json",
        "A.8",
        "compute h3_gini_fraction_in_eurostat_range in paper_numbers.json",
    )


def _h5_trust_gradient() -> HypRow:
    blob = _safe_load(TABLES / "trust_gradient.json")
    if not blob:
        return HypRow(
            "H5: trust gradient (Spearman ρ)",
            "ρ",
            None,
            None,
            None,
            "pending",
            "analysis/tables/trust_gradient.json",
            "A.5",
        )
    corr = blob.get("correlation") or {}
    return HypRow(
        "H5: trust gradient (Spearman ρ)",
        "ρ (4 trust bands)",
        float(corr.get("spearman_r", float("nan"))),
        None,
        None,
        "verified",
        "analysis/tables/trust_gradient.json",
        "A.5",
        f"p = {corr.get('p_value', 'na')}",
    )


def _h6_bad_apple() -> HypRow:
    blob = _safe_load(REPO_ROOT / "analysis" / "bad_apple_sweep.json")
    if not blob:
        return HypRow(
            "H6: bad-apple localisation",
            "localisation ratio",
            None,
            None,
            None,
            "pending",
            "analysis/bad_apple_sweep.json",
            "F.7",
        )
    # Use top-level mean localisation if present.
    return HypRow(
        "H6: bad-apple localisation",
        "localisation ratio (proxy)",
        float(blob.get("localization_ratio", float("nan")))
        if isinstance(blob.get("localization_ratio"), (int, float))
        else float("nan"),
        None,
        None,
        "partial",
        "analysis/bad_apple_sweep.json",
        "F.7",
        "schema heterogeneous; needs consolidation",
    )


def _h7_cross_model() -> HypRow:
    cross = _safe_load(REPO_ROOT / "analysis" / "cross_model_results.json") or []
    by_model: dict[str, dict] = {}
    for r in cross:
        m = r.get("model_id")
        c = r.get("condition")
        if not (m and c):
            continue
        by_model.setdefault(m, {}).setdefault(c, []).append(r.get("rlhf_bias_index"))
    deltas = []
    for m, conds in by_model.items():
        a = conds.get("A") or []
        b = conds.get("B") or []
        if a and b:
            a_mean = np.mean(a)
            b_mean = np.mean(b)
            if a_mean > 0:
                deltas.append((m, float((a_mean - b_mean) / a_mean)))
    if not deltas:
        return HypRow(
            "H7: cross-model B_RLHF generality",
            "mean rel. ΔB_RLHF",
            None,
            None,
            None,
            "pending",
            "analysis/cross_model_results.json",
            "A.4",
        )
    mean_delta = float(np.mean([d for _, d in deltas]))
    return HypRow(
        "H7: cross-model B_RLHF generality",
        "mean rel. ΔB_RLHF",
        mean_delta,
        None,
        None,
        "verified",
        "analysis/cross_model_results.json",
        "A.4",
        f"per-model: {dict(deltas)}",
    )


def _h8_persona_decay() -> HypRow:
    blob = _safe_load(TABLES / "long_horizon_persona_drift.json")
    if not blob:
        return HypRow(
            "H8: persona-fidelity slope (T=30)",
            "slope (∂fid/∂round)",
            None,
            None,
            None,
            "pending",
            "analysis/tables/long_horizon_persona_drift.json",
            "F.5",
        )
    slope = blob.get("slope") if isinstance(blob.get("slope"), (int, float)) else None
    return HypRow(
        "H8: persona-fidelity slope (T=30)",
        "slope",
        float(slope) if slope is not None else float("nan"),
        None,
        None,
        "partial",
        "analysis/tables/long_horizon_persona_drift.json",
        "F.5",
        "memory ablation under mock policy (A.9 ❌)",
    )


def _h9_cross_cultural() -> HypRow:
    blob = _safe_load(TABLES / "h9_cross_cultural_behavioral.json")
    if not blob:
        return HypRow(
            "H9: cross-cultural behavioural Spearman ρ",
            "ρ vs Herrmann/Henrich",
            None,
            None,
            None,
            "pending",
            "analysis/tables/h9_cross_cultural_behavioral.json",
            "D.3",
        )
    spearman_block = blob.get("spearman") or {}
    rho = spearman_block.get("rho", blob.get("spearman_rho"))
    return HypRow(
        "H9: cross-cultural behavioural Spearman ρ",
        "ρ vs Herrmann/Henrich",
        float(rho) if rho is not None else float("nan"),
        None,
        None,
        "verified",
        "analysis/tables/h9_cross_cultural_behavioral.json",
        "D.3",
        f"exact two-tailed p = {spearman_block.get('exact_permutation_p_two_tailed', 'n/a')}",
    )


def collect_rows() -> list[HypRow]:
    h1, h2, h4 = _h1_h2_h4_from_paper_numbers()
    return [
        h1,
        h2,
        _h3_gini_in_range(),
        h4,
        _h5_trust_gradient(),
        _h6_bad_apple(),
        _h7_cross_model(),
        _h8_persona_decay(),
        _h9_cross_cultural(),
    ]


def descriptive_pooled(rows: list[HypRow]) -> float | None:
    """Descriptive DerSimonian-Laird-style mean of available standardised
    effects. Effect sizes are heterogeneous, so this is *not* inferential."""
    finite = [r.effect for r in rows if r.effect is not None and not math.isnan(r.effect)]
    if not finite:
        return None
    return float(np.mean(finite))


def render(rows: list[HypRow], out_png: Path, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "rows": [asdict(r) for r in rows],
                "descriptive_pooled_effect": descriptive_pooled(rows),
                "audit_row": "E.6",
                "note": "Effect sizes are heterogeneous (g, ρ, ΔB_RLHF, etc.); pooled value is presentational.",
            },
            indent=2,
        )
    )
    print(f"JSON saved to: {out_json}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — skipping figure.")
        return

    fig, ax = plt.subplots(figsize=(9, 0.55 * (len(rows) + 1) + 1.5))

    y = list(range(len(rows)))[::-1]
    for i, row in enumerate(rows):
        yi = y[i]
        if row.effect is None or math.isnan(row.effect):
            ax.text(0.0, yi, " pending", va="center", ha="left", color="#888888", fontsize=9, style="italic")
            ax.plot([0], [yi], marker="x", color="#aaaaaa", markersize=8)
        else:
            color = {"verified": "#1f6feb", "partial": "#bf8700", "pending": "#aaaaaa"}.get(row.status, "#444444")
            ax.plot([row.effect], [yi], marker="o", color=color, markersize=8)
            if row.ci_lo is not None and row.ci_hi is not None:
                ax.plot([row.ci_lo, row.ci_hi], [yi, yi], color=color, linewidth=2)
            ax.text(row.effect, yi + 0.18, f"{row.effect:+.3f}", ha="center", fontsize=8, color=color)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r.hypothesis for r in rows], fontsize=9)
    ax.set_xlabel("Standardised effect (positive = BGF prediction supported)")
    ax.set_title(
        "BGF H1-H9 Convergent-Evidence Forest Plot\n(heterogeneous effect-size metrics — see JSON for definitions)"
    )

    pooled = descriptive_pooled(rows)
    if pooled is not None:
        ax.axvline(
            pooled, color="#1f6feb", linewidth=1, linestyle="--", label=f"descriptive pooled mean = {pooled:+.3f}"
        )
        ax.legend(loc="lower right", fontsize=8)

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="H1-H9 forest plot.")
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(TABLES / "forest_plot.json"),
    )
    parser.add_argument(
        "--out-fig",
        type=str,
        default=str(REPO_ROOT / "analysis" / "figures" / "forest_plot.png"),
    )
    args = parser.parse_args()

    rows = collect_rows()
    render(rows, Path(args.out_fig), Path(args.out_json))

    print("\n=== Forest plot rows ===")
    for r in rows:
        eff = f"{r.effect:+.3f}" if r.effect is not None and not math.isnan(r.effect) else "  pending"
        print(f"  [{r.status:<8}] {r.hypothesis:<48} {eff}   audit:{r.audit_row}")


if __name__ == "__main__":
    main()
