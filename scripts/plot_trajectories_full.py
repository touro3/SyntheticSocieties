"""
Advanced trajectory plotting across multiple seeds.
Generates wealth/stress flows with CI bands, action frequency areas,
and diversity/collapse metrics (entropy + Gini over time).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metrics.inequality import gini_coefficient
from metrics.trajectories import aggregate_seeds

# ── Style ──────────────────────────────────────────────────────────────────

plt.rcParams.update(
    {
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
    }
)

COLORS = {
    "llm": "#e94560",
    "template": "#0f3460",
    "rule_based": "#533483",
    "random": "#16c79a",
    "work": "#e94560",
    "save": "#0f3460",
    "cooperate": "#16c79a",
    "steal": "#ff6b6b",
}

# Optional suffix appended after the seed in experiment IDs (e.g. "_condA").
# Set by --id-suffix CLI flag in main().
ID_SUFFIX = ""


# ══════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════


def plot_wealth_stress_trajectories(policies: list[str], seeds: list[int], output_path: Path):
    """Plot Mean +/- Std Wealth and Stress trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for policy in policies:
        data = aggregate_seeds(policy, seeds)
        if not data:
            continue

        rounds = data["rounds"]
        color = COLORS.get(policy, "#999")
        label = policy.replace("_", " ").title()

        # Wealth
        axes[0].plot(rounds, data["wealth_mean"], label=label, color=color, linewidth=2.5)
        axes[0].fill_between(
            rounds,
            data["wealth_mean"] - data["wealth_std"],
            data["wealth_mean"] + data["wealth_std"],
            color=color,
            alpha=0.15,
        )

        # Stress
        axes[1].plot(rounds, data["stress_mean"], label=label, color=color, linewidth=2.5)
        axes[1].fill_between(
            rounds,
            data["stress_mean"] - data["stress_std"],
            data["stress_mean"] + data["stress_std"],
            color=color,
            alpha=0.15,
        )

    axes[0].set_title("Wealth Trajectories (Mean +/- 1 sigma)", fontweight="bold")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Wealth")
    axes[0].legend(fontsize=9)
    axes[0].grid(True)

    axes[1].set_title("Stress Trajectories (Mean +/- 1 sigma)", fontweight="bold")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Stress")
    axes[1].grid(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


def plot_per_agent_ci_bands(policy: str, seeds: list[int], output_path: Path, n_agents_show: int = 5):
    """Plot per-agent wealth trajectories with 95% CI bands across seeds.

    Shows the first `n_agents_show` agents' individual trajectories
    with mean and a shaded 95% CI band (2.5th-97.5th percentile).
    """
    from metrics.trajectories import extract_trajectories

    # Collect per-agent wealth across seeds
    prefix_candidates = ["cmp_", "ablation_", "pert_"]
    name_map = {"llm": "llm", "template": "template", "rule_based": "rule", "random": "random"}
    prefix_name = name_map.get(policy, policy)

    agent_seed_data: dict[str, list[list[float]]] = {}
    for seed in seeds:
        for pref in prefix_candidates:
            exp_id = f"{pref}{prefix_name}_s{seed}{ID_SUFFIX}"
            exp_dir = Path("experiments") / exp_id
            if exp_dir.exists():
                data = extract_trajectories(exp_dir)
                if data:
                    for a_id, traj in data["agent_trajectories"].items():
                        agent_seed_data.setdefault(a_id, []).append(traj["wealth"])
                    break

    if not agent_seed_data:
        print(f"  (no data for {policy})")
        return

    # Pick agents with most seeds of data
    agents = sorted(agent_seed_data.keys(), key=lambda a: len(agent_seed_data[a]), reverse=True)
    agents = agents[:n_agents_show]

    fig, axes = plt.subplots(1, len(agents), figsize=(5 * len(agents), 5), sharey=True)
    if len(agents) == 1:
        axes = [axes]

    for ax, agent_id in zip(axes, agents):
        runs = agent_seed_data[agent_id]
        # Trim to min length
        min_len = min(len(r) for r in runs)
        arr = np.array([r[:min_len] for r in runs])  # [Seeds x Rounds]
        rounds = np.arange(min_len)
        mean = arr.mean(axis=0)
        ci_low = np.percentile(arr, 2.5, axis=0)
        ci_high = np.percentile(arr, 97.5, axis=0)

        ax.plot(rounds, mean, color="#e94560", linewidth=2)
        ax.fill_between(rounds, ci_low, ci_high, color="#e94560", alpha=0.2)
        # Individual seed traces
        for row in arr:
            ax.plot(rounds, row, color="#e94560", alpha=0.15, linewidth=0.5)
        ax.set_title(agent_id, fontsize=10)
        ax.set_xlabel("Round")
        ax.grid(True)

    axes[0].set_ylabel("Wealth")
    fig.suptitle(
        f"Per-Agent Wealth Trajectories ({policy.upper()}, {len(runs)} seeds, 95% CI)",
        fontweight="bold",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


def plot_action_frequencies_area(policy: str, seeds: list[int], output_path: Path):
    """Stacked area chart of action frequencies for a single policy."""
    data = aggregate_seeds(policy, seeds)
    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    rounds = data["rounds"]
    freqs = data["action_freqs"].T  # [3 x Rounds]
    labels = data["action_labels"]

    ax.stackplot(
        rounds,
        freqs,
        labels=labels,
        colors=[COLORS[l] for l in labels],
        alpha=0.8,
    )

    ax.set_title(
        f"Action Frequency Evolution -- {policy.upper()}",
        fontweight="bold",
        fontsize=14,
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Proportion of Actions")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(min(rounds), max(rounds))
    ax.legend(loc="upper right", fontsize=10)

    for i, r in enumerate(rounds):
        if i % 5 == 0 or i == len(rounds) - 1:
            cumulative = 0
            for j, val in enumerate(freqs[:, i]):
                if val > 0.05:
                    ax.text(
                        r,
                        cumulative + val / 2,
                        f"{val:.0%}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                        fontweight="bold",
                    )
                cumulative += val

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


def _compute_gini_trajectory(pool_wealth: np.ndarray) -> np.ndarray:
    """Compute Gini coefficient at each round from pooled wealth matrix.

    Parameters
    ----------
    pool_wealth : ndarray of shape [Rounds x Agents]

    Returns
    -------
    ndarray of shape [Rounds] with Gini values.
    """
    n_rounds = pool_wealth.shape[0]
    gini_vals = np.zeros(n_rounds)
    for r in range(n_rounds):
        w = pool_wealth[r]
        if len(w) > 1 and w.sum() > 0:
            gini_vals[r] = gini_coefficient(w)
    return gini_vals


def plot_diversity_collapse(policies: list[str], seeds: list[int], output_path: Path):
    """Plot Action Entropy and Gini coefficient over time."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for policy in policies:
        data = aggregate_seeds(policy, seeds)
        if not data:
            continue

        rounds = data["rounds"]
        color = COLORS.get(policy, "#999")
        label = policy.replace("_", " ").title()

        # Action Entropy: H = -sum(p * log2(p))
        freqs = data["action_freqs"]  # [Rounds x 3]
        safe_freqs = np.clip(freqs, 1e-9, 1.0)
        entropy = -np.sum(freqs * np.log2(safe_freqs), axis=1)
        axes[0].plot(rounds, entropy, label=label, color=color, linewidth=2.5)

        # Gini Coefficient over rounds (from pooled wealth matrix)
        pool_wealth = data.get("pool_wealth")
        if pool_wealth is not None:
            gini_traj = _compute_gini_trajectory(pool_wealth)
            axes[1].plot(rounds, gini_traj, label=label, color=color, linewidth=2.5)

    axes[0].set_title("Behavioral Entropy (Action Diversity)", fontweight="bold")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Entropy (Bits)")
    axes[0].set_ylim(0, 1.6)  # max for 3 actions is ~1.58 bits
    axes[0].grid(True)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Wealth Gini Coefficient over Rounds", fontweight="bold")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Gini Index")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Advanced Trajectory Plotting")
    parser.add_argument("--seeds", type=str, default="42,123,7,1,2", help="Comma-sep seeds")
    parser.add_argument("--out-dir", type=str, default="analysis/figures", help="Output dir")
    parser.add_argument(
        "--id-suffix",
        type=str,
        default="",
        help="Optional suffix appended to experiment IDs (e.g. '_condA').",
    )
    args = parser.parse_args()

    global ID_SUFFIX
    ID_SUFFIX = args.id_suffix

    if "," in args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        val = int(args.seeds)
        seeds = [42, 123, 7, 1, 2, 88, 99, 101, 102, 103][:val] if val <= 20 else [val]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = ["llm", "template", "rule_based", "random"]

    print("=" * 60)
    print(f"Generating Trajectory Plots (Seeds: {len(seeds)})")
    print("=" * 60)

    print("\n1. Wealth & Stress Trajectories...")
    plot_wealth_stress_trajectories(policies, seeds, out_dir / "wealth_stress_trajectories.png")

    print("\n2. Per-Agent CI Bands...")
    for policy in policies:
        plot_per_agent_ci_bands(policy, seeds, out_dir / f"per_agent_ci_{policy}.png")

    print("\n3. Action Frequency Areas...")
    for policy in policies:
        plot_action_frequencies_area(policy, seeds, out_dir / f"action_area_{policy}.png")

    print("\n4. Diversity & Collapse (Entropy + Gini)...")
    plot_diversity_collapse(policies, seeds, out_dir / "diversity_collapse.png")

    print("\n" + "=" * 60)
    print(f"Done! Trajectory plots saved in {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
