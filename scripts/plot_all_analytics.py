"""
Publication-quality analytics plots from all experiment data.

Generates ~10 figures covering:
  1. LLM-alone vs ESS-grounded comparison (PRIORITY)
  2. Policy comparison heatmap
  3. Ablation effect chart
  4. Perturbation robustness
  5. Lorenz curves
  6. Distribution divergences (JSD, KL, Wasserstein)
  7. Calibration gap chart
  8. Network structure visualization
  9. Agent interaction graph
  10. Comprehensive results dashboard

Usage:
    python scripts/plot_all_analytics.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from metrics.inequality import gini_coefficient, lorenz_curve
from metrics.distribution import jensen_shannon_divergence, kl_divergence, wasserstein_distance
from metrics.calibration import calibration_evaluation_split

# ── Style ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#e8e8e8",
    "text.color": "#e8e8e8",
    "xtick.color": "#e8e8e8",
    "ytick.color": "#e8e8e8",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.4,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "figure.titlesize": 15,
})

COLORS = {
    "llm": "#e94560",
    "template": "#0f3460",
    "rule_based": "#533483",
    "random": "#16c79a",
    "no_persona": "#f5a623",
    "minimal_persona": "#50c4ed",
    "rich_persona": "#e94560",
    "ess": "#ff6b6b",
    "ungrounded": "#4ecdc4",
}

LABELS = {
    "llm": "LLM (ESS-Grounded)",
    "template": "Template (ESS)",
    "rule_based": "Rule-Based",
    "random": "Random",
    "no_persona": "LLM (No Persona)",
    "minimal_persona": "LLM (Minimal)",
    "rich_persona": "LLM (Rich Persona)",
}

OUTPUT_DIR = Path("analysis/figures")
EXPERIMENTS_DIR = Path("experiments")
TABLES_DIR = Path("analysis/tables")
DATA_PATH = Path("data/ess_clean.parquet")

POLICY_PREFIX = {"llm": "llm", "template": "template", "rule_based": "rule", "random": "random"}
SEEDS = [42, 123, 7]


def load_summary(exp_id: str) -> dict:
    path = EXPERIMENTS_DIR / exp_id / "summary.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.loads(f.read())


def load_wealth(policy: str, seeds=SEEDS) -> list[float]:
    prefix = POLICY_PREFIX.get(policy, policy)
    wealth = []
    for s in seeds:
        summary = load_summary(f"cmp_{prefix}_s{s}")
        wealth.extend(summary.get("wealth", {}).get("values", []))
    return wealth


def load_actions(policy: str, seeds=SEEDS) -> Counter:
    prefix = POLICY_PREFIX.get(policy, policy)
    actions = Counter()
    for s in seeds:
        summary = load_summary(f"cmp_{prefix}_s{s}")
        actions.update(summary.get("event_action_counts", {}))
    return actions


def load_ablation_wealth(mode: str, seeds=SEEDS) -> list[float]:
    wealth = []
    for s in seeds:
        summary = load_summary(f"ablation_{mode}_s{s}")
        wealth.extend(summary.get("wealth", {}).get("values", []))
    return wealth


# ══════════════════════════════════════════════════════════════════════════
# PLOT 1: LLM-ALONE VS ESS-GROUNDED (PRIORITY)
# ══════════════════════════════════════════════════════════════════════════

# how to fix: 

def plot_llm_grounding_comparison():
    """Compare LLM with no persona (ungrounded) vs LLM with ESS personas."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LLM-Alone vs ESS-Grounded Comparison", fontsize=16, fontweight="bold", y=0.98)

    # Load data
    grounded_wealth = load_wealth("llm")
    ungrounded_wealth = load_ablation_wealth("no_persona")
    minimal_wealth = load_ablation_wealth("minimal_persona")
    rich_wealth = load_ablation_wealth("rich_persona")

    grounded_actions = load_actions("llm")
    ungrounded_actions = Counter()
    for s in SEEDS:
        summary = load_summary(f"ablation_no_persona_s{s}")
        ungrounded_actions.update(summary.get("event_action_counts", {}))

    # Load ESS empirical data for reference
    ess_income = []
    if DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH)
        if "hinctnta" in df.columns:
            ess_income = df["hinctnta"].dropna().tolist()

    # ── Panel A: Wealth Distribution ──
    ax = axes[0, 0]
    bins = np.linspace(0, max(max(grounded_wealth, default=0), max(ungrounded_wealth, default=0)) * 1.1, 20)

    if grounded_wealth:
        ax.hist(grounded_wealth, bins=bins, alpha=0.6, color=COLORS["llm"],
                label=f"ESS-Grounded (μ={np.mean(grounded_wealth):.0f}, G={gini_coefficient(grounded_wealth):.3f})",
                edgecolor="white", linewidth=0.5)
    if ungrounded_wealth:
        ax.hist(ungrounded_wealth, bins=bins, alpha=0.6, color=COLORS["ungrounded"],
                label=f"Ungrounded (μ={np.mean(ungrounded_wealth):.0f}, G={gini_coefficient(ungrounded_wealth):.3f})",
                edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Final Wealth")
    ax.set_ylabel("Count")
    ax.set_title("A. Wealth Distribution")
    ax.legend(fontsize=8)

    # ── Panel B: Action Distribution ──
    ax = axes[0, 1]
    action_types = ["work", "save", "cooperate"]
    x = np.arange(len(action_types))
    width = 0.35

    grounded_total = max(sum(grounded_actions.values()), 1)
    ungrounded_total = max(sum(ungrounded_actions.values()), 1)
    grounded_pcts = [grounded_actions.get(a, 0) / grounded_total for a in action_types]
    ungrounded_pcts = [ungrounded_actions.get(a, 0) / ungrounded_total for a in action_types]

    bars1 = ax.bar(x - width/2, grounded_pcts, width, label="ESS-Grounded",
                   color=COLORS["llm"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, ungrounded_pcts, width, label="Ungrounded",
                   color=COLORS["ungrounded"], edgecolor="white", linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.0%}",
                        ha="center", va="bottom", fontsize=8, color="#e8e8e8")

    ax.set_xticks(x)
    ax.set_xticklabels(action_types)
    ax.set_ylabel("Proportion")
    ax.set_title("B. Action Distribution")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)

    # ── Panel C: Lorenz Curves ──
    ax = axes[1, 0]
    for data, label, color in [
        (grounded_wealth, "ESS-Grounded", COLORS["llm"]),
        (ungrounded_wealth, "Ungrounded", COLORS["ungrounded"]),
        (rich_wealth, "Rich Persona", COLORS["rich_persona"]),
        (minimal_wealth, "Minimal Persona", COLORS["minimal_persona"]),
    ]:
        if data:
            lc = lorenz_curve(data)
            ax.plot(lc["population_share"], lc["value_share"], label=label,
                    color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "--", color="#666", linewidth=1, label="Perfect Equality")
    ax.set_xlabel("Population Share")
    ax.set_ylabel("Wealth Share")
    ax.set_title("C. Lorenz Curves")
    ax.legend(fontsize=7, loc="upper left")

    # ── Panel D: Key Metrics Comparison ──
    ax = axes[1, 1]
    conditions = ["ESS-Grounded\n(Full LLM)", "No Persona\n(Ungrounded)", "Minimal\nPersona", "Rich\nPersona"]
    datasets = [grounded_wealth, ungrounded_wealth, minimal_wealth, rich_wealth]
    colors = [COLORS["llm"], COLORS["ungrounded"], COLORS["minimal_persona"], COLORS["rich_persona"]]

    metrics = {"Gini": [], "Mean Wealth\n(÷100)": [], "Coop Rate": []}

    all_action_sets = []
    for mode in [None, "no_persona", "minimal_persona", "rich_persona"]:
        ac = Counter()
        if mode is None:
            ac = grounded_actions
        else:
            for s in SEEDS:
                sm = load_summary(f"ablation_{mode}_s{s}")
                ac.update(sm.get("event_action_counts", {}))
        all_action_sets.append(ac)

    for i, (data, ac) in enumerate(zip(datasets, all_action_sets)):
        if data:
            metrics["Gini"].append(gini_coefficient(data))
            metrics["Mean Wealth\n(÷100)"].append(np.mean(data) / 100)
        else:
            metrics["Gini"].append(0)
            metrics["Mean Wealth\n(÷100)"].append(0)
        total = max(sum(ac.values()), 1)
        metrics["Coop Rate"].append(ac.get("cooperate", 0) / total)

    x = np.arange(len(conditions))
    width = 0.25
    for j, (metric_name, values) in enumerate(metrics.items()):
        bars = ax.bar(x + j * width - width, values, width, label=metric_name,
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="#e8e8e8")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_title("D. Key Metrics by Grounding Level")
    ax.legend(fontsize=7, loc="upper right")

    fig.subplots_adjust(top=0.90, hspace=0.30, wspace=0.25)
    out = OUTPUT_DIR / "llm_grounding_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")

    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# PLOT 2: POLICY COMPARISON HEATMAP
# ══════════════════════════════════════════════════════════════════════════

def plot_policy_heatmap():
    """Heatmap of all metrics across all policies."""
    policies = ["llm", "template", "rule_based", "random"]
    metric_names = ["Mean Wealth", "Gini", "Coop%", "Work%", "Save%", "Stress"]

    data_matrix = []
    for policy in policies:
        wealth = load_wealth(policy)
        actions = load_actions(policy)
        total_a = max(sum(actions.values()), 1)

        prefix = POLICY_PREFIX.get(policy, policy)
        stress_vals = []
        for s in SEEDS:
            sm = load_summary(f"cmp_{prefix}_s{s}")
            stress_vals.append(sm.get("stress", {}).get("mean", 0))

        row = [
            np.mean(wealth) if wealth else 0,
            gini_coefficient(wealth) if wealth else 0,
            actions.get("cooperate", 0) / total_a,
            actions.get("work", 0) / total_a,
            actions.get("save", 0) / total_a,
            np.mean(stress_vals) if stress_vals else 0,
        ]
        data_matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 5))
    mat = np.array(data_matrix)

    # Normalize each column to [0, 1] for color mapping
    col_min = mat.min(axis=0)
    col_max = mat.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    normalized = (mat - col_min) / col_range

    im = ax.imshow(normalized, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Annotate with actual values
    for i in range(len(policies)):
        for j in range(len(metric_names)):
            val = mat[i, j]
            fmt = f"{val:.0f}" if j == 0 else f"{val:.3f}" if j == 1 else f"{val:.1%}" if j < 5 else f"{val:.2f}"
            color = "white" if normalized[i, j] > 0.5 else "#e8e8e8"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels([LABELS.get(p, p) for p in policies])
    ax.set_title("Policy Comparison Heatmap", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Normalized Value", color="#e8e8e8")
    cbar.ax.yaxis.set_tick_params(color="#e8e8e8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#e8e8e8")

    plt.tight_layout()
    out = OUTPUT_DIR / "policy_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# PLOT 3: ABLATION EFFECT
# ══════════════════════════════════════════════════════════════════════════

def plot_ablation_effect():
    """Show how removing prompt components affects LLM behavior."""
    modes = ["rich_persona", "minimal_persona", "no_persona"]
    mode_labels = ["Rich Persona\n(Full ESS)", "Minimal Persona\n(Age+Gender)", "No Persona\n(Ungrounded)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Ablation Study: Effect of Persona Conditioning", fontsize=14, fontweight="bold")

    # Wealth distributions
    ax = axes[0]
    for mode, label in zip(modes, mode_labels):
        wealth = load_ablation_wealth(mode)
        if wealth:
            ax.hist(wealth, bins=15, alpha=0.5, label=f"{label.split(chr(10))[0]} (μ={np.mean(wealth):.0f})",
                    edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Final Wealth")
    ax.set_ylabel("Count")
    ax.set_title("Wealth Distribution")
    ax.legend(fontsize=8)

    # Action proportions
    ax = axes[1]
    action_types = ["work", "save", "cooperate"]
    x = np.arange(len(action_types))
    width = 0.25
    colors_list = [COLORS["rich_persona"], COLORS["minimal_persona"], COLORS["no_persona"]]

    for i, (mode, label) in enumerate(zip(modes, mode_labels)):
        actions = Counter()
        for s in SEEDS:
            sm = load_summary(f"ablation_{mode}_s{s}")
            actions.update(sm.get("event_action_counts", {}))
        total = max(sum(actions.values()), 1)
        pcts = [actions.get(a, 0) / total for a in action_types]
        ax.bar(x + i * width - width, pcts, width, label=label.split("\n")[0],
               color=colors_list[i], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(action_types)
    ax.set_ylabel("Proportion")
    ax.set_title("Action Distribution")
    ax.legend(fontsize=8)

    # Gini comparison
    ax = axes[2]
    ginis = []
    for mode in modes:
        wealth = load_ablation_wealth(mode)
        ginis.append(gini_coefficient(wealth) if wealth else 0)
    bars = ax.bar(range(len(modes)), ginis, color=colors_list, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([l.split("\n")[0] for l in mode_labels], fontsize=9)
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Inequality")
    for bar, g in zip(bars, ginis):
        ax.text(bar.get_x() + bar.get_width()/2, g + 0.005, f"{g:.3f}",
                ha="center", va="bottom", fontsize=9, color="#e8e8e8")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / "ablation_effect.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# PLOT 4: DISTRIBUTION DIVERGENCES
# ══════════════════════════════════════════════════════════════════════════

def plot_distribution_divergences():
    """JSD, KL, Wasserstein between policies and ESS-grounded LLM."""
    reference = load_wealth("llm")
    if not reference:
        print("  ⚠ No LLM wealth data for divergence plot")
        return None

    policies = ["template", "rule_based", "random"]
    ablations = ["no_persona", "minimal_persona", "rich_persona"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Distribution Divergences from ESS-Grounded LLM", fontsize=14, fontweight="bold")

    metric_funcs = {
        "Jensen-Shannon": jensen_shannon_divergence,
        "KL Divergence": kl_divergence,
        "Wasserstein": wasserstein_distance,
    }

    for idx, (metric_name, func) in enumerate(metric_funcs.items()):
        ax = axes[idx]
        all_labels = []
        all_values = []
        all_colors = []

        for policy in policies:
            wealth = load_wealth(policy)
            if wealth:
                val = func(reference, wealth)
                all_labels.append(LABELS.get(policy, policy))
                all_values.append(val)
                all_colors.append(COLORS.get(policy, "#999"))

        for mode in ablations:
            wealth = load_ablation_wealth(mode)
            if wealth:
                val = func(reference, wealth)
                all_labels.append(LABELS.get(mode, mode))
                all_values.append(val)
                all_colors.append(COLORS.get(mode, "#999"))

        bars = ax.barh(range(len(all_labels)), all_values, color=all_colors,
                       edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(all_labels)))
        ax.set_yticklabels(all_labels, fontsize=8)
        ax.set_xlabel(metric_name)
        ax.set_title(metric_name)

        for bar, val in zip(bars, all_values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=8, color="#e8e8e8")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / "distribution_divergences.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# PLOT 5: LORENZ CURVES (ALL POLICIES)
# ══════════════════════════════════════════════════════════════════════════

def plot_lorenz_all():
    """Lorenz curves for all policies."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("Lorenz Curves — Wealth Inequality", fontsize=14, fontweight="bold")

    for policy in ["llm", "template", "rule_based", "random"]:
        wealth = load_wealth(policy)
        if wealth:
            lc = lorenz_curve(wealth)
            g = gini_coefficient(wealth)
            ax.plot(lc["population_share"], lc["value_share"],
                    label=f"{LABELS[policy]} (G={g:.3f})",
                    color=COLORS[policy], linewidth=2.5)

    ax.plot([0, 1], [0, 1], "--", color="#666", linewidth=1, label="Perfect Equality")
    ax.set_xlabel("Cumulative Population Share")
    ax.set_ylabel("Cumulative Wealth Share")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_aspect("equal")

    plt.tight_layout()
    out = OUTPUT_DIR / "lorenz_curves_all.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# PLOT 6: CALIBRATION GAP
# ══════════════════════════════════════════════════════════════════════════

def plot_calibration_gap():
    """Calibration vs evaluation metrics gap."""
    policies = ["llm", "template", "rule_based", "random"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Calibration vs Evaluation Gap", fontsize=14, fontweight="bold")

    metric_keys = ["mean_wealth", "gini", "coop_rate"]
    metric_titles = ["Mean Wealth", "Gini Coefficient", "Cooperation Rate"]

    for idx, (key, title) in enumerate(zip(metric_keys, metric_titles)):
        ax = axes[idx]
        cal_vals = []
        eval_vals = []
        labels = []

        for policy in policies:
            result = calibration_evaluation_split(policy)
            cal_vals.append(result["calibration"].get(key, 0))
            eval_vals.append(result["evaluation"].get(key, 0))
            labels.append(LABELS.get(policy, policy).split("(")[0].strip())

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, cal_vals, width, label="Calibration\n(seeds 42, 123)",
               color=COLORS["llm"], alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.bar(x + width/2, eval_vals, width, label="Evaluation\n(seed 7)",
               color=COLORS["ungrounded"], alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / "calibration_gap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# PLOT 7: PERTURBATION ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════

def plot_perturbation_robustness():
    """Show LLM sensitivity to prompt perturbations."""
    modes = ["rephrase", "shuffle", "noise"]
    mode_labels = ["Rephrase", "Shuffle", "Noise"]

    # Check what perturbation data exists
    pert_data = {}
    for mode in modes:
        wealth = []
        actions = Counter()
        for s in SEEDS:
            # Try pert_{mode}_s{seed} naming
            sm = load_summary(f"pert_{mode}_s{s}")
            if not sm:
                sm = load_summary(f"pert_llm_s{s}")  # Fallback
            if sm:
                wealth.extend(sm.get("wealth", {}).get("values", []))
                actions.update(sm.get("event_action_counts", {}))
        if wealth:
            pert_data[mode] = {"wealth": wealth, "actions": actions}

    if not pert_data:
        print("  ⚠ No perturbation data found, skipping plot")
        return None

    # Also load baseline LLM for comparison
    base_wealth = load_wealth("llm")
    base_actions = load_actions("llm")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Prompt Perturbation Robustness", fontsize=14, fontweight="bold")

    # Wealth comparison
    ax = axes[0]
    all_data = [("Baseline", base_wealth, COLORS["llm"])]
    for mode, label in zip(modes, mode_labels):
        if mode in pert_data:
            all_data.append((label, pert_data[mode]["wealth"], COLORS.get(mode, "#999")))

    bp = ax.boxplot([d[1] for d in all_data if d[1]], tick_labels=[d[0] for d in all_data if d[1]],
                    patch_artist=True, medianprops=dict(color="white", linewidth=2))
    for patch, (_, _, color) in zip(bp["boxes"], [d for d in all_data if d[1]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Final Wealth")
    ax.set_title("Wealth Distribution")

    # Action comparison
    ax = axes[1]
    action_types = ["work", "save", "cooperate"]
    x = np.arange(len(action_types))
    width = 0.2
    offset = 0
    base_total = max(sum(base_actions.values()), 1)
    bars = ax.bar(x - 1.5 * width, [base_actions.get(a, 0)/base_total for a in action_types],
                  width, label="Baseline", color=COLORS["llm"], edgecolor="white", linewidth=0.5)
    for mode, label in zip(modes, mode_labels):
        if mode in pert_data:
            offset += 1
            total = max(sum(pert_data[mode]["actions"].values()), 1)
            pcts = [pert_data[mode]["actions"].get(a, 0)/total for a in action_types]
            ax.bar(x + (offset - 1.5) * width, pcts, width, label=label,
                   edgecolor="white", linewidth=0.5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(action_types)
    ax.set_ylabel("Proportion")
    ax.set_title("Action Distribution")
    ax.legend(fontsize=8)

    # Gini comparison
    ax = axes[2]
    ginis = [gini_coefficient(base_wealth)] if base_wealth else [0]
    labels_g = ["Baseline"]
    colors_g = [COLORS["llm"]]
    for mode, label in zip(modes, mode_labels):
        if mode in pert_data:
            ginis.append(gini_coefficient(pert_data[mode]["wealth"]))
            labels_g.append(label)
            colors_g.append(COLORS.get(mode, "#999"))

    bars = ax.bar(range(len(ginis)), ginis, color=colors_g, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(ginis)))
    ax.set_xticklabels(labels_g)
    ax.set_ylabel("Gini")
    ax.set_title("Inequality Sensitivity")
    for bar, g in zip(bars, ginis):
        ax.text(bar.get_x() + bar.get_width()/2, g + 0.005, f"{g:.3f}",
                ha="center", va="bottom", fontsize=9, color="#e8e8e8")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / "perturbation_robustness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# PLOT 8: V0-V5 PROMPT ABLATION LADDER (LLM BEHAVIOR RECOVERY)
# ══════════════════════════════════════════════════════════════════════════

def plot_ladder_ablation():
    """Show the effect of the V0->V5 structured ablation ladder on LLM behavior collapse."""
    levels = list(range(6))
    labels = ["V0: Base", "V1: Stress\nSalience", "V2: Coop\nIncentives", "V3: Trust\nMemory", "V4: Balanced\nPhrasing", "V5: T=0.7\nDecoder"]
    
    ginis = []
    coop_rates = []
    work_rates = []
    save_rates = []
    
    data_found = False
    for lvl in levels:
        wealth = []
        actions = Counter()
        for s in SEEDS:
            sm = load_summary(f"abl_v{lvl}_llm_s{s}")
            if sm:
                data_found = True
                wealth.extend(sm.get("wealth", {}).get("values", []))
                actions.update(sm.get("event_action_counts", {}))
        
        ginis.append(gini_coefficient(wealth) if wealth else 0)
        tot = max(sum(actions.values()), 1)
        coop_rates.append(actions.get("cooperate", 0) / tot)
        work_rates.append(actions.get("work", 0) / tot)
        save_rates.append(actions.get("save", 0) / tot)
    
    if not data_found:
        print("  ⚠ No V0-V5 ablation ladder data found, skipping plot")
        return None
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("V0-V5 Ablation Ladder: Resolving Action Collapse", fontsize=14, fontweight="bold")
    
    # Action Rates (Stacked Bar)
    ax = axes[0]
    x = np.arange(len(levels))
    
    ax.bar(x, work_rates, label="Work", color="#e94560", edgecolor="white", linewidth=0.5)
    ax.bar(x, save_rates, bottom=work_rates, label="Save", color="#0f3460", edgecolor="white", linewidth=0.5)
    ax.bar(x, coop_rates, bottom=np.array(work_rates)+np.array(save_rates), label="Cooperate", color="#16c79a", edgecolor="white", linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Action Proportion")
    ax.set_title("A. Action Distribution Recovery")
    ax.legend()
    
    # Equality
    ax = axes[1]
    ax.plot(x, ginis, marker="o", color="#50c4ed", linewidth=2, markersize=8)
    for i, g in enumerate(ginis):
        ax.text(i, g + 0.01, f"{g:.3f}", ha="center", va="bottom", fontsize=9, color="#e8e8e8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("B. Wealth Inequality Dynamics")
    ax.set_ylim(0, max(ginis)*1.2 if max(ginis) > 0 else 1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / "ladder_ablation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# PLOT 9: COMPREHENSIVE RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

def plot_results_dashboard():
    """Comprehensive 2x3 dashboard combining key results."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    fig.suptitle("BGF Comprehensive Results Dashboard", fontsize=16, fontweight="bold", y=0.98)

    policies = ["llm", "template", "rule_based", "random"]

    # Panel 1: Mean Wealth Bar
    ax = fig.add_subplot(gs[0, 0])
    means = [np.mean(load_wealth(p)) if load_wealth(p) else 0 for p in policies]
    stds = [np.std(load_wealth(p)) if load_wealth(p) else 0 for p in policies]
    colors_list = [COLORS[p] for p in policies]
    bars = ax.bar(range(len(policies)), means, yerr=stds, color=colors_list,
                  edgecolor="white", linewidth=0.5, capsize=3)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([LABELS[p].split("(")[0].strip() for p in policies], fontsize=9)
    ax.set_ylabel("Mean Wealth")
    ax.set_title("A. Mean Wealth ± σ")

    # Panel 2: Gini Coefficients
    ax = fig.add_subplot(gs[0, 1])
    ginis = [gini_coefficient(load_wealth(p)) if load_wealth(p) else 0 for p in policies]
    bars = ax.bar(range(len(policies)), ginis, color=colors_list,
                  edgecolor="white", linewidth=0.5)
    for bar, g in zip(bars, ginis):
        ax.text(bar.get_x() + bar.get_width()/2, g + 0.005, f"{g:.3f}",
                ha="center", va="bottom", fontsize=9, color="#e8e8e8")
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([LABELS[p].split("(")[0].strip() for p in policies], fontsize=9)
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("B. Inequality (Gini)")

    # Panel 3: Action Distributions (stacked)
    ax = fig.add_subplot(gs[0, 2])
    action_types = ["work", "save", "cooperate"]
    action_colors = ["#e94560", "#0f3460", "#16c79a"]
    bottoms = np.zeros(len(policies))
    for j, action in enumerate(action_types):
        vals = []
        for p in policies:
            ac = load_actions(p)
            total = max(sum(ac.values()), 1)
            vals.append(ac.get(action, 0) / total)
        ax.bar(range(len(policies)), vals, bottom=bottoms, label=action,
               color=action_colors[j], edgecolor="white", linewidth=0.5)
        bottoms += np.array(vals)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([LABELS[p].split("(")[0].strip() for p in policies], fontsize=9)
    ax.set_ylabel("Proportion")
    ax.set_title("C. Action Distribution")
    ax.legend(fontsize=8, loc="upper right")

    # Panel 4: Lorenz curves
    ax = fig.add_subplot(gs[1, 0])
    for p in policies:
        wealth = load_wealth(p)
        if wealth:
            lc = lorenz_curve(wealth)
            ax.plot(lc["population_share"], lc["value_share"],
                    color=COLORS[p], linewidth=2, label=LABELS[p].split("(")[0].strip())
    ax.plot([0, 1], [0, 1], "--", color="#666", linewidth=1)
    ax.set_xlabel("Population Share")
    ax.set_ylabel("Wealth Share")
    ax.set_title("D. Lorenz Curves")
    ax.legend(fontsize=8)

    # Panel 5: Ablation effect
    ax = fig.add_subplot(gs[1, 1])
    abl_modes = ["rich_persona", "minimal_persona", "no_persona"]
    abl_labels = ["Rich\nPersona", "Minimal\nPersona", "No\nPersona"]
    abl_ginis = []
    abl_means = []
    for mode in abl_modes:
        w = load_ablation_wealth(mode)
        abl_ginis.append(gini_coefficient(w) if w else 0)
        abl_means.append(np.mean(w) if w else 0)

    x = np.arange(len(abl_modes))
    ax.bar(x - 0.2, [m / 100 for m in abl_means], 0.4, label="Wealth (÷100)",
           color=COLORS["llm"], edgecolor="white", linewidth=0.5)
    ax.bar(x + 0.2, abl_ginis, 0.4, label="Gini",
           color=COLORS["ungrounded"], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(abl_labels, fontsize=9)
    ax.set_title("E. Ablation: Persona Effect")
    ax.legend(fontsize=8)

    # Panel 6: Calibration gap
    ax = fig.add_subplot(gs[1, 2])
    gap_data = {}
    for p in policies:
        result = calibration_evaluation_split(p)
        avg_gap = np.mean(list(result["gap"].values()))
        gap_data[p] = avg_gap

    bars = ax.bar(range(len(policies)), list(gap_data.values()),
                  color=colors_list, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([LABELS[p].split("(")[0].strip() for p in policies], fontsize=9)
    ax.set_ylabel("Average Gap (%)")
    ax.set_title("F. Calibration–Evaluation Gap")
    ax.axhline(y=25, color="#e94560", linestyle="--", linewidth=1, alpha=0.7, label="HIGH threshold")
    ax.axhline(y=10, color="#f5a623", linestyle="--", linewidth=1, alpha=0.7, label="MEDIUM threshold")
    ax.legend(fontsize=7)
    for bar, g in zip(bars, gap_data.values()):
        ax.text(bar.get_x() + bar.get_width()/2, g + 1, f"{g:.1f}%",
                ha="center", va="bottom", fontsize=9, color="#e8e8e8")

    fig.subplots_adjust(top=0.92, hspace=0.32, wspace=0.28)

    out = OUTPUT_DIR / "results_dashboard.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Publication Plots")
    print("=" * 60)

    figures = []

    print("\n1. LLM-Alone vs ESS-Grounded (PRIORITY)...")
    figures.append(plot_llm_grounding_comparison())

    print("\n2. Policy Comparison Heatmap...")
    figures.append(plot_policy_heatmap())

    print("\n3. Ablation Effect...")
    figures.append(plot_ablation_effect())

    print("\n4. Distribution Divergences...")
    figures.append(plot_distribution_divergences())

    print("\n5. Lorenz Curves...")
    figures.append(plot_lorenz_all())

    print("\n6. Calibration Gap...")
    figures.append(plot_calibration_gap())

    print("\n7. Perturbation Robustness...")
    figures.append(plot_perturbation_robustness())

    print("\n8. V0-V5 Ablation Ladder...")
    figures.append(plot_ladder_ablation())

    print("\n9. Results Dashboard...")
    figures.append(plot_results_dashboard())

    generated = [f for f in figures if f is not None]
    print(f"\n{'=' * 60}")
    print(f"Done! Generated {len(generated)} figures in {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
