"""
Generate publication-quality policy comparison diagrams.

Compares LLM, template, rule-based, and random policies
against ESS empirical baselines.

Produces 4 diagrams saved to analysis/figures/:
  1. policy_wealth_comparison.png     — Wealth distributions by policy vs ESS
  2. policy_behavior_comparison.png   — Action distributions + cooperation rates
  3. policy_dynamics_comparison.png   — Gini, wealth, cooperation over rounds
  4. policy_summary_radar.png         — Radar chart of aggregate metrics

Usage:
    python scripts/plot_policy_comparison_full.py

Prerequisite:
    python scripts/run_experiment_suite.py --suite-config configs/comparison_suite.yaml
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.io import set_global_seed

# ── Style ────────────────────────────────────────────────────────────────────

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

POLICY_COLORS = {
    "llm": "#e94560",
    "template": "#16c79a",
    "rule_based": "#533483",
    "random": "#f5a623",
}
POLICY_LABELS = {
    "llm": "LLM (Mistral-7B)",
    "template": "Template (ESS Archetype)",
    "rule_based": "Rule-Based",
    "random": "Random",
}
ESS_COLOR = "#50c4ed"
OUTPUT_DIR = Path("analysis/figures")
EXPERIMENTS_DIR = Path("experiments")
ESS_PATH = Path("data/ess_clean.parquet")
SEEDS = [42, 123, 7]
# Map policy key → experiment ID prefix (must match comparison_suite.yaml)
POLICY_EXP_PREFIX = {
    "llm": "llm",
    "template": "template",
    "rule_based": "rule",
    "random": "random",
}


# ── Correct Gini ─────────────────────────────────────────────────────────────

def gini_coefficient(values: list[float]) -> float:
    """
    Proper Gini coefficient: G = Σ_i Σ_j |x_i - x_j| / (2 * n^2 * x̄)
    """
    arr = np.array(values, dtype=float)
    arr = arr[arr >= 0]  # filter negatives
    if len(arr) < 2 or arr.sum() == 0:
        return 0.0
    n = len(arr)
    diff_sum = sum(abs(arr[i] - arr[j]) for i in range(n) for j in range(n))
    return float(diff_sum / (2 * n * n * arr.mean()))


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_experiment_events(experiment_id: str) -> list[dict]:
    """Load events from an experiment's events.jsonl file."""
    events_path = EXPERIMENTS_DIR / experiment_id / "events.jsonl"
    if not events_path.exists():
        return []
    events = []
    with events_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def load_experiment_summary(experiment_id: str) -> dict:
    """Load summary from an experiment's summary.json file."""
    summary_path = EXPERIMENTS_DIR / experiment_id / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open() as f:
        return json.loads(f.read())


def gather_policy_data(policy: str) -> dict:
    """
    Gather data across seeds for a given policy.
    Reads BOTH summary.json (for final wealth) AND events.jsonl (for dynamics).
    """
    all_events = []
    all_final_wealths = []
    all_action_counts = Counter()
    per_seed_events = {}

    for seed in SEEDS:
        exp_id = f"cmp_{POLICY_EXP_PREFIX[policy]}_s{seed}"

        # Load events
        events = load_experiment_events(exp_id)
        if events:
            all_events.extend(events)
            per_seed_events[seed] = events

        # Load summary — final wealth from summary.json
        summary = load_experiment_summary(exp_id)
        if summary:
            # wealth.values contains final wealth array
            wealth_data = summary.get("wealth", {})
            wealth_values = wealth_data.get("values", [])
            if wealth_values:
                all_final_wealths.extend(wealth_values)

            # event_action_counts is the ground truth for action distribution
            eac = summary.get("event_action_counts", {})
            all_action_counts.update(eac)
        elif events:
            # Fallback: extract from events if no summary
            for e in events:
                action = e.get("action", {}).get("action_type")
                if action:
                    all_action_counts[action] += 1
            # Extract final wealth from last round's state_after
            max_round = max(e.get("round_id", 0) for e in events)
            for e in events:
                if e.get("round_id") == max_round:
                    w = e.get("state_after", {}).get("wealth")
                    if w is not None:
                        all_final_wealths.append(w)

    return {
        "events": all_events,
        "per_seed_events": per_seed_events,
        "final_wealths": all_final_wealths,
        "action_counts": dict(all_action_counts),
    }


def extract_round_metrics(events: list[dict]) -> dict:
    """Extract per-round metrics from events using correct Gini formula."""
    round_data = defaultdict(lambda: {"wealths": [], "actions": []})

    for e in events:
        r = e.get("round_id", 0)
        state = e.get("state_after", {})
        action = e.get("action", {}).get("action_type", "unknown")
        wealth = state.get("wealth", 0.0)
        round_data[r]["wealths"].append(wealth)
        round_data[r]["actions"].append(action)

    ginis = []
    wealth_means = []
    coop_rates = []

    for r in sorted(round_data.keys()):
        w = round_data[r]["wealths"]
        a = round_data[r]["actions"]
        total_a = max(len(a), 1)

        wealth_means.append(float(np.mean(w)) if w else 0.0)
        ginis.append(gini_coefficient(w))
        coop_rates.append(a.count("cooperate") / total_a)

    return {
        "rounds": sorted(round_data.keys()),
        "ginis": ginis,
        "wealth_means": wealth_means,
        "coop_rates": coop_rates,
    }


def load_ess_wealth_proxy() -> np.ndarray:
    """Load ESS income data as a wealth proxy for comparison."""
    if not ESS_PATH.exists():
        return np.array([])
    df = pd.read_parquet(ESS_PATH)
    if "household_income" in df.columns:
        vals = df["household_income"].dropna().values
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            # Scale to simulation range
            vals = (vals - vmin) / (vmax - vmin) * 200 + 50
        return vals
    return np.array([])


# ── Figure 1: Wealth Distribution Comparison ─────────────────────────────────

def plot_wealth_comparison(policy_data: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Final Wealth Distributions by Policy vs ESS Empirical",
                 fontweight="bold", y=1.01)

    ess_wealth = load_ess_wealth_proxy()
    policies = ["llm", "template", "rule_based", "random"]

    for idx, policy in enumerate(policies):
        ax = axes[idx // 2][idx % 2]
        data = policy_data.get(policy, {})
        final_w = data.get("final_wealths", [])

        has_data = False

        if len(ess_wealth) > 0:
            ax.hist(ess_wealth, bins=20, alpha=0.5, label="ESS Empirical",
                    color=ESS_COLOR, edgecolor="#1a1a2e", density=True)
            has_data = True

        if len(final_w) > 0:
            ax.hist(final_w, bins=20, alpha=0.7, label=POLICY_LABELS[policy],
                    color=POLICY_COLORS[policy], edgecolor="#1a1a2e", density=True)
            mean_w = np.mean(final_w)
            gini = gini_coefficient(final_w)
            ax.axvline(mean_w, color=POLICY_COLORS[policy],
                       linestyle="--", linewidth=1.5,
                       label=f"μ={mean_w:.1f}, Gini={gini:.3f}")
            has_data = True
        else:
            ax.text(0.5, 0.5, "No data available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="#e94560", fontweight="bold")

        ax.set_title(POLICY_LABELS[policy])
        ax.set_xlabel("Wealth")
        ax.set_ylabel("Density")
        if has_data:
            ax.legend(fontsize=8)
        ax.grid(True)

    fig.tight_layout()
    path = OUTPUT_DIR / "policy_wealth_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Figure 2: Behavior Comparison ────────────────────────────────────────────

def plot_behavior_comparison(policy_data: dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Behavioral Comparison Across Policies", fontweight="bold", y=1.02)

    policies = ["llm", "template", "rule_based", "random"]
    action_types = ["work", "save", "cooperate"]
    action_colors = {"work": "#e94560", "save": "#533483", "cooperate": "#16c79a"}

    # Panel 1: Stacked action distribution (from action_counts)
    ax = axes[0]
    x = np.arange(len(policies))
    bottoms = np.zeros(len(policies))

    for action in action_types:
        rates = []
        for policy in policies:
            counts = policy_data.get(policy, {}).get("action_counts", {})
            total = max(sum(counts.values()), 1)
            rates.append(counts.get(action, 0) / total)

        ax.bar(x, rates, width=0.6, bottom=bottoms,
               label=action.capitalize(), color=action_colors[action],
               edgecolor="#1a1a2e", alpha=0.85)

        # Add text labels in each segment
        for i, rate in enumerate(rates):
            if rate > 0.05:
                ax.text(x[i], bottoms[i] + rate / 2,
                        f"{rate:.0%}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="#e8e8e8")

        bottoms += np.array(rates)

    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies], fontsize=8, rotation=10)
    ax.set_ylabel("Action Rate")
    ax.set_title("Action Distribution (all rounds × all seeds)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y")

    # Panel 2: Cooperation rate per seed with error bars
    ax = axes[1]
    coop_means = []
    coop_stds = []

    for policy in policies:
        seed_coop = []
        for seed in SEEDS:
            exp_id = f"cmp_{POLICY_EXP_PREFIX[policy]}_s{seed}"
            summary = load_experiment_summary(exp_id)
            if summary:
                eac = summary.get("event_action_counts", {})
                total = max(sum(eac.values()), 1)
                seed_coop.append(eac.get("cooperate", 0) / total)
            else:
                events = load_experiment_events(exp_id)
                if events:
                    actions = [e.get("action", {}).get("action_type") for e in events]
                    total = max(len(actions), 1)
                    seed_coop.append(actions.count("cooperate") / total)

        if seed_coop:
            coop_means.append(np.mean(seed_coop))
            coop_stds.append(np.std(seed_coop))
        else:
            coop_means.append(0)
            coop_stds.append(0)

    bars = ax.bar(x, coop_means, yerr=coop_stds, capsize=5, width=0.6,
                  color=[POLICY_COLORS[p] for p in policies],
                  edgecolor="#1a1a2e", alpha=0.85)

    for bar, mean_val in zip(bars, coop_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean_val:.2f}", ha="center", fontweight="bold",
                color="#e8e8e8", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies], fontsize=8, rotation=10)
    ax.set_ylabel("Cooperation Rate")
    ax.set_title("Mean Cooperation Rate (± std across seeds)")
    ax.set_ylim(0, max(max(coop_means) + 0.15, 0.5))
    ax.grid(True, axis="y")

    fig.tight_layout()
    path = OUTPUT_DIR / "policy_behavior_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Figure 3: Dynamics Comparison ────────────────────────────────────────────

def plot_dynamics_comparison(policy_data: dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Simulation Dynamics — Policy Comparison (10 Rounds)",
                 fontweight="bold", y=1.02)

    policies = ["llm", "template", "rule_based", "random"]
    markers = ["o", "s", "D", "^"]

    for idx, policy in enumerate(policies):
        # Average across seeds
        seed_metrics = []
        for seed in SEEDS:
            exp_id = f"cmp_{POLICY_EXP_PREFIX[policy]}_s{seed}"
            events = load_experiment_events(exp_id)
            if events:
                seed_metrics.append(extract_round_metrics(events))

        if not seed_metrics:
            continue

        # Find common round range
        max_rounds = max(len(m["rounds"]) for m in seed_metrics)
        avg_ginis = []
        avg_wealth = []
        avg_coop = []

        for r_idx in range(max_rounds):
            g_vals = [m["ginis"][r_idx] for m in seed_metrics if r_idx < len(m["ginis"])]
            w_vals = [m["wealth_means"][r_idx] for m in seed_metrics if r_idx < len(m["wealth_means"])]
            c_vals = [m["coop_rates"][r_idx] for m in seed_metrics if r_idx < len(m["coop_rates"])]

            avg_ginis.append(np.mean(g_vals) if g_vals else 0)
            avg_wealth.append(np.mean(w_vals) if w_vals else 0)
            avg_coop.append(np.mean(c_vals) if c_vals else 0)

        rounds = range(1, max_rounds + 1)

        # Gini
        axes[0].plot(rounds, avg_ginis,
                     color=POLICY_COLORS[policy], linewidth=2,
                     marker=markers[idx], markersize=5,
                     label=POLICY_LABELS[policy])

        # Wealth
        axes[1].plot(rounds, avg_wealth,
                     color=POLICY_COLORS[policy], linewidth=2,
                     marker=markers[idx], markersize=5,
                     label=POLICY_LABELS[policy])

        # Cooperation
        axes[2].plot(rounds, avg_coop,
                     color=POLICY_COLORS[policy], linewidth=2,
                     marker=markers[idx], markersize=5,
                     label=POLICY_LABELS[policy])

    axes[0].set_title("Wealth Inequality (Gini)")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Gini Coefficient")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].set_title("Mean Wealth")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Wealth")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    axes[2].set_title("Cooperation Rate")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Rate")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(fontsize=8)
    axes[2].grid(True)

    fig.tight_layout()
    path = OUTPUT_DIR / "policy_dynamics_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Figure 4: Radar Summary ─────────────────────────────────────────────────

def plot_radar_summary(policy_data: dict):
    policies = ["llm", "template", "rule_based", "random"]
    metrics_names = ["Mean Wealth", "Cooperation", "Action Diversity",
                     "Wealth Equality", "Stability"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]

    for policy in policies:
        data = policy_data.get(policy, {})
        final_w = data.get("final_wealths", [])
        counts = data.get("action_counts", {})
        total_actions = max(sum(counts.values()), 1)

        if not final_w and not counts:
            continue

        # 1. Mean wealth (normalized to 0-1, max ~300)
        mean_wealth = np.mean(final_w) / 300.0 if final_w else 0.0
        mean_wealth = min(mean_wealth, 1.0)

        # 2. Cooperation rate
        coop_rate = counts.get("cooperate", 0) / total_actions

        # 3. Action diversity (Shannon entropy / max entropy)
        probs = np.array([counts.get(a, 0) / total_actions for a in ["work", "save", "cooperate"]])
        probs = probs[probs > 0]
        max_ent = np.log2(3)
        diversity = float(-np.sum(probs * np.log2(probs)) / max_ent) if len(probs) > 0 else 0.0

        # 4. Wealth equality (1 - gini)
        equality = 1.0 - gini_coefficient(final_w) if final_w else 0.5

        # 5. Stability (low variance in per-seed cooperation rates)
        seed_coop = []
        for seed in SEEDS:
            exp_id = f"cmp_{POLICY_EXP_PREFIX[policy]}_s{seed}"
            summary = load_experiment_summary(exp_id)
            if summary:
                eac = summary.get("event_action_counts", {})
                t = max(sum(eac.values()), 1)
                seed_coop.append(eac.get("cooperate", 0) / t)
        if len(seed_coop) > 1:
            stability = 1.0 - min(np.std(seed_coop) * 5, 1.0)
        else:
            stability = 0.5

        values = [mean_wealth, coop_rate, diversity, equality, stability]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, color=POLICY_COLORS[policy],
                label=POLICY_LABELS[policy])
        ax.fill(angles, values, alpha=0.15, color=POLICY_COLORS[policy])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=10, color="#e8e8e8")
    ax.set_ylim(0, 1)
    ax.set_title("Policy Performance Radar", fontweight="bold", pad=20, color="#e8e8e8")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    ax.spines["polar"].set_color("#e94560")
    ax.tick_params(colors="#e8e8e8")
    ax.yaxis.grid(True, color="#2a2a4a", alpha=0.5)
    ax.xaxis.grid(True, color="#2a2a4a", alpha=0.5)

    path = OUTPUT_DIR / "policy_summary_radar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(42)

    print("=" * 60)
    print("Policy Comparison — Diagram Generation")
    print("=" * 60)

    # Check experiments
    policies = ["llm", "template", "rule_based", "random"]
    available = {}
    missing = []
    for policy in policies:
        for seed in SEEDS:
            exp_id = f"cmp_{POLICY_EXP_PREFIX[policy]}_s{seed}"
            exp_dir = EXPERIMENTS_DIR / exp_id
            if exp_dir.exists() and (exp_dir / "events.jsonl").exists():
                available[exp_id] = True
            else:
                missing.append(exp_id)

    print(f"\nAvailable: {len(available)}/12 experiments")
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("Run: python scripts/run_experiment_suite.py --suite-config configs/comparison_suite.yaml\n")

    # Gather data
    print("Loading experiment data...")
    policy_data = {}
    for policy in policies:
        data = gather_policy_data(policy)
        policy_data[policy] = data
        n_events = len(data["events"])
        n_wealth = len(data["final_wealths"])
        ac = data["action_counts"]
        print(f"  {POLICY_LABELS[policy]:30s}: {n_events:4d} events, "
              f"{n_wealth:3d} wealth values, actions={ac}")

    print()
    print("Generating figures...")

    plot_wealth_comparison(policy_data)
    plot_behavior_comparison(policy_data)
    plot_dynamics_comparison(policy_data)
    plot_radar_summary(policy_data)

    print()
    print(f"All figures saved to: {OUTPUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
