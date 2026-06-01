"""
Generate empirical analysis diagrams from ESS data and simulation results.

Produces 6 diagnostic plots saved to analysis/figures/:
  1. ess_demographics.png        — Age/gender distributions
  2. ess_trust_politics.png      — Trust, political orientation, satisfaction
  3. ess_behavioral_profiles.png — Behavioral proxy + feature correlations
  4. ess_population_heatmap.png  — Correlation heatmap of ESS variables
  5. empirical_vs_synthetic.png  — Side-by-side comparison of population modes
  6. empirical_simulation_results.png — Empirical run: actions, Gini, cooperation

Usage:
    python scripts/plot_empirical_analysis.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.io import set_global_seed

# ── Style ────────────────────────────────────────────────────────────────────

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
        "axes.titlesize": 13,
        "figure.titlesize": 15,
    }
)

COLORS = ["#e94560", "#0f3460", "#533483", "#16c79a", "#f5a623", "#50c4ed"]
OUTPUT_DIR = Path("analysis/figures")
DATA_PATH = Path("data/ess_clean.parquet")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Run: python scripts/ingest_ess.py")
        sys.exit(1)
    return pd.read_parquet(DATA_PATH)


# ── Figure 1: Demographics ───────────────────────────────────────────────────


def plot_demographics(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ESS Demographics — Austrian Respondents", fontweight="bold", y=1.02)

    # Age distribution
    ax = axes[0]
    ages = df["age"].dropna()
    ax.hist(ages, bins=25, color=COLORS[0], alpha=0.85, edgecolor="#1a1a2e")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("Age Distribution")
    ax.axvline(ages.median(), color=COLORS[3], linestyle="--", label=f"Median: {ages.median():.0f}")
    ax.legend()
    ax.grid(True)

    # Gender distribution
    ax = axes[1]
    if "gender" in df.columns:
        gender_map = {1: "Male", 2: "Female"}
        gender_counts = df["gender"].dropna().map(gender_map).value_counts()
        bars = ax.bar(gender_counts.index, gender_counts.values, color=[COLORS[1], COLORS[0]], edgecolor="#1a1a2e")
        for bar, val in zip(bars, gender_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(val),
                ha="center",
                fontweight="bold",
                color="#e8e8e8",
            )
        ax.set_title("Gender Distribution")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y")

    # Education level
    ax = axes[2]
    if "education_level" in df.columns:
        edu = df["education_level"].dropna()
        edu_labels = {
            1: "< Lower\nSec",
            2: "Lower\nSec",
            3: "Upper\nSec",
            4: "Post\nSec",
            5: "Short\nTertiary",
            6: "Bachelor",
            7: "Master+",
        }
        edu_counts = edu.value_counts().sort_index()
        labels = [edu_labels.get(int(k), str(int(k))) for k in edu_counts.index]
        ax.bar(labels, edu_counts.values, color=COLORS[2], alpha=0.85, edgecolor="#1a1a2e")
        ax.set_title("Education Level (ES-ISCED)")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0, labelsize=8)
        ax.grid(True, axis="y")

    fig.tight_layout()
    path = OUTPUT_DIR / "ess_demographics.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Figure 2: Trust & Politics ───────────────────────────────────────────────


def plot_trust_politics(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("ESS Trust, Politics & Satisfaction", fontweight="bold", y=1.01)

    vars_info = [
        ("trust_people", "Interpersonal Trust", COLORS[0]),
        ("trust_parliament", "Trust in Parliament", COLORS[1]),
        ("trust_police", "Trust in Police", COLORS[2]),
        ("left_right", "Political Orientation\n(Left → Right)", COLORS[3]),
        ("life_satisfaction", "Life Satisfaction", COLORS[4]),
        ("happiness", "Happiness", COLORS[5]),
    ]

    for idx, (col, title, color) in enumerate(vars_info):
        ax = axes[idx // 3][idx % 3]
        if col in df.columns:
            vals = df[col].dropna()
            ax.hist(vals, bins=20, color=color, alpha=0.85, edgecolor="#1a1a2e")
            ax.axvline(vals.mean(), color="#ffffff", linestyle="--", linewidth=1.5, label=f"μ={vals.mean():.2f}")
            ax.set_title(title)
            ax.set_xlabel("Normalized [0-1]")
            ax.set_ylabel("Count")
            ax.legend(fontsize=9)
            ax.grid(True)

    fig.tight_layout()
    path = OUTPUT_DIR / "ess_trust_politics.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Figure 3: Behavioral Profiles ────────────────────────────────────────────


def plot_behavioral_profiles(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("ESS Behavioral & Personality Profiles", fontweight="bold", y=1.02)

    # Behavioral proxy distribution
    ax = axes[0]
    behavior_path = Path("data/behavior/ess_behavior_dataset.csv")
    if behavior_path.exists():
        bdf = pd.read_csv(behavior_path)
        if "action" in bdf.columns:
            counts = bdf["action"].value_counts()
            bars = ax.bar(counts.index, counts.values, color=[COLORS[3], COLORS[0], COLORS[1]], edgecolor="#1a1a2e")
            for bar, val in zip(bars, counts.values):
                pct = val / len(bdf) * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    f"{pct:.1f}%",
                    ha="center",
                    fontweight="bold",
                    color="#e8e8e8",
                )
            ax.set_title("Behavioral Proxy Distribution")
            ax.set_ylabel("Count")
            ax.grid(True, axis="y")

    # Risk vs Competitiveness
    ax = axes[1]
    risk_cols = ["risk_taking", "competitiveness"]
    if all(c in df.columns for c in risk_cols):
        valid = df[risk_cols].dropna()
        ax.scatter(valid["risk_taking"], valid["competitiveness"], alpha=0.4, s=15, c=COLORS[0], edgecolors="none")
        ax.set_xlabel("Risk Taking")
        ax.set_ylabel("Competitiveness")
        ax.set_title("Risk vs Competitiveness")
        ax.grid(True)

    # Trust vs Social Activity
    ax = axes[2]
    social_cols = ["trust_people", "social_meeting_freq"]
    if all(c in df.columns for c in social_cols):
        valid = df[social_cols].dropna()
        ax.scatter(valid["trust_people"], valid["social_meeting_freq"], alpha=0.4, s=15, c=COLORS[2], edgecolors="none")
        ax.set_xlabel("Trust in People")
        ax.set_ylabel("Social Meeting Frequency")
        ax.set_title("Trust vs Social Activity")
        ax.grid(True)

    fig.tight_layout()
    path = OUTPUT_DIR / "ess_behavioral_profiles.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Figure 4: Correlation Heatmap ────────────────────────────────────────────


def plot_heatmap(df: pd.DataFrame):
    # Select key variables for the heatmap
    heatmap_cols = [
        "age",
        "trust_people",
        "trust_parliament",
        "trust_police",
        "left_right",
        "life_satisfaction",
        "happiness",
        "risk_taking",
        "competitiveness",
        "social_meeting_freq",
        "self_rated_health",
        "immigration_same_ethnicity",
        "reduce_inequality",
        "satisfaction_economy",
    ]
    available = [c for c in heatmap_cols if c in df.columns]
    subset = df[available].dropna()

    if len(subset) < 10:
        print("  ⚠ Not enough valid rows for heatmap")
        return

    corr = subset.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    labels = [c.replace("_", "\n") for c in available]
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    for i in range(len(available)):
        for j in range(len(available)):
            val = corr.values[i, j]
            color = "#1a1a2e" if abs(val) > 0.5 else "#e8e8e8"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson Correlation")
    ax.set_title("ESS Variable Correlation Heatmap", fontweight="bold", pad=15)

    fig.tight_layout()
    path = OUTPUT_DIR / "ess_population_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Figure 5: Empirical vs Synthetic Comparison ──────────────────────────────


def plot_empirical_vs_synthetic():
    """
    Compare empirical and synthetic population modes by running quick simulations.
    """
    from decision.mock_policy import MockPolicy
    from population.generator import generate_empirical_population, generate_population
    from utils.config import load_config

    config = load_config("configs/base_config.yaml")
    set_global_seed(42)
    policy = MockPolicy()

    # Synthetic population
    config_syn = dict(config)
    config_syn["simulation"] = dict(config["simulation"])
    config_syn["simulation"]["population_size"] = 50
    syn_agents = generate_population(config_syn, policy)

    # Empirical population
    config_emp = dict(config)
    config_emp["simulation"] = dict(config["simulation"])
    config_emp["simulation"]["population_size"] = 50
    emp_agents = generate_empirical_population(config_emp, policy)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Synthetic vs Empirical Population Comparison", fontweight="bold", y=1.01)

    # Wealth distribution
    ax = axes[0][0]
    syn_wealth = [a.state.wealth for a in syn_agents]
    emp_wealth = [a.state.wealth for a in emp_agents]
    ax.hist(syn_wealth, bins=15, alpha=0.7, label="Synthetic", color=COLORS[1], edgecolor="#1a1a2e")
    ax.hist(emp_wealth, bins=15, alpha=0.7, label="Empirical", color=COLORS[0], edgecolor="#1a1a2e")
    ax.set_title("Initial Wealth Distribution")
    ax.set_xlabel("Wealth")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True)

    # Age distribution
    ax = axes[0][1]
    syn_ages = [a.profile.age for a in syn_agents]
    emp_ages = [a.profile.age for a in emp_agents]
    ax.hist(syn_ages, bins=15, alpha=0.7, label="Synthetic", color=COLORS[1], edgecolor="#1a1a2e")
    ax.hist(emp_ages, bins=15, alpha=0.7, label="Empirical", color=COLORS[0], edgecolor="#1a1a2e")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True)

    # Trust distribution (empirical only)
    ax = axes[1][0]
    emp_trust = [a.profile.trust_people for a in emp_agents if a.profile.trust_people is not None]
    syn_trust = [0.5] * len(syn_agents)  # synthetic default
    ax.hist(syn_trust, bins=15, alpha=0.7, label="Synthetic (default)", color=COLORS[1], edgecolor="#1a1a2e")
    ax.hist(emp_trust, bins=15, alpha=0.7, label="Empirical (ESS)", color=COLORS[0], edgecolor="#1a1a2e")
    ax.set_title("Trust in People")
    ax.set_xlabel("Trust Level [0-1]")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True)

    # Risk tolerance
    ax = axes[1][1]
    syn_risk = [a.profile.risk_tolerance for a in syn_agents]
    emp_risk = [a.profile.risk_tolerance for a in emp_agents if a.profile.risk_tolerance is not None]
    ax.hist(syn_risk, bins=15, alpha=0.7, label="Synthetic", color=COLORS[1], edgecolor="#1a1a2e")
    ax.hist(emp_risk, bins=15, alpha=0.7, label="Empirical", color=COLORS[0], edgecolor="#1a1a2e")
    ax.set_title("Risk Tolerance")
    ax.set_xlabel("Risk Tolerance [0-1]")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    path = OUTPUT_DIR / "empirical_vs_synthetic.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Figure 6: Empirical Simulation Results ───────────────────────────────────


def plot_simulation_results():
    """
    Run a quick empirical simulation and visualize results.
    """
    from bgf_logging.event_logger import EventLogger
    from decision.rule_based_policy import RuleBasedPolicy
    from environment.institutions import InstitutionManager
    from environment.network import NetworkManager
    from environment.world import World
    from environment.world_state import WorldState
    from metrics.inequality import gini_coefficient
    from metrics.summary import summarize_agents
    from population.generator import generate_empirical_population
    from simulation.kernel import SimulationKernel
    from utils.config import load_config

    config = load_config("configs/base_config.yaml")
    set_global_seed(42)

    config_emp = dict(config)
    config_emp["simulation"] = dict(config["simulation"])
    config_emp["simulation"]["population_size"] = 50
    config_emp["simulation"]["rounds"] = 10

    policy = RuleBasedPolicy()
    agents = generate_empirical_population(config_emp, policy)

    env_cfg = config.get("environment", {})
    world_state = WorldState(
        public_signal=env_cfg.get("public_signal", {}),
        prices=env_cfg.get("prices", {"food": 1.0}),
        resources=env_cfg.get("resources", {"jobs": 100.0}),
    )

    net_cfg = config.get("network", {})
    agent_ids = [a.profile.agent_id for a in agents]
    network_manager = NetworkManager.random_graph(
        agent_ids=agent_ids,
        edge_prob=net_cfg.get("edge_prob", 0.3),
        seed=42,
    )

    institution_manager = InstitutionManager()
    world = World(state=world_state, institution_manager=institution_manager, network_manager=network_manager)

    # Temp log file
    tmp_log = Path("/tmp/bgf_empirical_sim.jsonl")
    logger = EventLogger(str(tmp_log))

    kernel = SimulationKernel(
        agents=agents,
        world=world,
        logger=logger,
    )

    # Track per-round metrics
    round_ginis = []
    round_coop_rates = []
    round_wealth_means = []

    for r in range(10):
        kernel.run_round()
        wealths = [a.state.wealth for a in agents]
        round_ginis.append(gini_coefficient(wealths))
        coop = sum(1 for a in agents if a.state.last_action == "cooperate") / len(agents)
        round_coop_rates.append(coop)
        round_wealth_means.append(np.mean(wealths))

    # Final summary
    summary = summarize_agents(agents)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Empirical Population — Simulation Results (10 rounds, rule-based)", fontweight="bold", y=1.01)

    # Final action distribution
    ax = axes[0][0]
    actions = summary["actions"]
    bars = ax.bar(
        actions.keys(),
        actions.values(),
        color=[COLORS[i % len(COLORS)] for i in range(len(actions))],
        edgecolor="#1a1a2e",
    )
    for bar, val in zip(bars, actions.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val),
            ha="center",
            fontweight="bold",
            color="#e8e8e8",
        )
    ax.set_title("Final Action Distribution")
    ax.set_ylabel("Agent Count")
    ax.grid(True, axis="y")

    # Gini over rounds
    ax = axes[0][1]
    ax.plot(range(1, 11), round_ginis, color=COLORS[0], linewidth=2, marker="o", markersize=5)
    ax.set_title("Wealth Inequality (Gini) Over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Gini Coefficient")
    ax.grid(True)

    # Cooperation rate over rounds
    ax = axes[1][0]
    ax.plot(range(1, 11), round_coop_rates, color=COLORS[3], linewidth=2, marker="s", markersize=5)
    ax.fill_between(range(1, 11), round_coop_rates, alpha=0.2, color=COLORS[3])
    ax.set_title("Cooperation Rate Over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cooperation Rate")
    ax.set_ylim(0, 1)
    ax.grid(True)

    # Wealth trajectory
    ax = axes[1][1]
    ax.plot(range(1, 11), round_wealth_means, color=COLORS[4], linewidth=2, marker="D", markersize=5)
    ax.set_title("Mean Wealth Over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean Wealth")
    ax.grid(True)

    fig.tight_layout()
    path = OUTPUT_DIR / "empirical_simulation_results.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")

    # Clean up temp file
    if tmp_log.exists():
        tmp_log.unlink()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(42)

    print("=" * 60)
    print("Empirical Analysis — Diagram Generation")
    print("=" * 60)

    df = load_data()
    print(f"Loaded ESS data: {df.shape}")
    print()

    print("Generating figures...")
    plot_demographics(df)
    plot_trust_politics(df)
    plot_behavioral_profiles(df)
    plot_heatmap(df)
    plot_empirical_vs_synthetic()
    plot_simulation_results()

    print()
    print(f"All figures saved to: {OUTPUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
