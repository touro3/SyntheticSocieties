"""Mechanism analysis: action-transition matrices + per-round JSD across
Condition A vs Condition B. CPU-only; reads events.jsonl directly.

Two mechanism signals are computed:

1. **Action-transition matrices.** For each condition, build the 3×3
   stochastic matrix P[i,j] = Pr(next action = j | current action = i)
   pooled across all agents and all consecutive-round transitions.
   Report row entropy, stationary distribution, off-diagonal mass.
   Mode collapse under Condition A appears as P[cooperate, cooperate] → 1
   and zero off-diagonal mass; grounded heterogeneity appears as more
   uniform off-diagonal mass.

2. **Per-round Jensen-Shannon divergence trajectory.** For each round t,
   compute JSD(π_A(t) ‖ π_uniform) and JSD(π_B(t) ‖ π_uniform).
   The trajectory tests whether grounding stabilises the action
   distribution over time (flatter JSD) or whether mode collapse
   sharpens (JSD growing toward 2/3).

Outputs:
  - analysis/tables/action_transitions.json
  - analysis/tables/per_round_jsd.json
  - analysis/figures/action_transitions.png
  - analysis/figures/per_round_jsd.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXP_DIR = REPO / "experiments"
TABLES = REPO / "analysis" / "tables"
FIGURES = REPO / "analysis" / "figures"

ACTIONS = ["work", "save", "cooperate"]


def load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def action_transition_matrix(events: list[dict]) -> np.ndarray:
    """3×3 stochastic transition matrix pooled across agents."""
    by_agent: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for ev in events:
        a = ev.get("action", {}).get("action_type")
        if a in ACTIONS:
            by_agent[ev["agent_id"]].append((ev["round_id"], a))

    counts = np.zeros((3, 3), dtype=float)
    for agent_id, history in by_agent.items():
        history.sort()
        for (r0, a0), (r1, a1) in zip(history, history[1:]):
            if r1 == r0 + 1 and a0 in ACTIONS and a1 in ACTIONS:
                counts[ACTIONS.index(a0), ACTIONS.index(a1)] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


def per_round_action_dist(events: list[dict]) -> dict[int, np.ndarray]:
    rounds: dict[int, list[str]] = defaultdict(list)
    for ev in events:
        a = ev.get("action", {}).get("action_type")
        if a in ACTIONS:
            rounds[ev["round_id"]].append(a)
    out = {}
    for r, acts in rounds.items():
        dist = np.array([acts.count(a) for a in ACTIONS], dtype=float)
        s = dist.sum()
        out[r] = dist / s if s > 0 else dist
    return out


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    m = 0.5 * (p + q)

    def kl(x, y):
        return np.sum(np.where(x > 0, x * np.log2((x + eps) / (y + eps)), 0))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def stationary(P: np.ndarray) -> np.ndarray:
    """Left eigenvector for eigenvalue 1, normalised."""
    eig_vals, eig_vecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eig_vals - 1))
    v = np.real(eig_vecs[:, idx])
    v = v / v.sum()
    return v


def summarise_transition(P: np.ndarray) -> dict:
    eps = 1e-12
    row_entropy = -np.nansum(P * np.log2(P + eps), axis=1)
    off_diag = float(P.sum() - np.trace(P))
    return {
        "matrix": P.tolist(),
        "row_entropy_bits": row_entropy.tolist(),
        "mean_row_entropy": float(row_entropy.mean()),
        "off_diagonal_mass": off_diag,
        "stationary_dist": stationary(P).tolist(),
        "diag_dominance": float(np.trace(P) / 3.0),
    }


def main() -> None:
    pairs = [
        (
            "A_ungrounded",
            [
                "pure_llm_ess_persona_s42",
                "pure_llm_ess_persona_s43",
                "pure_llm_ess_persona_s44",
            ],
        ),
        (
            "B_grounded",
            [
                "grounded_llm_ess_persona_s42",
                "grounded_llm_ess_persona_s43",
                "grounded_llm_ess_persona_s44",
            ],
        ),
    ]

    transitions: dict[str, dict] = {}
    per_round: dict[str, dict[int, list[float]]] = {}
    for label, exp_names in pairs:
        all_events: list[dict] = []
        for n in exp_names:
            ev = load_events(EXP_DIR / n / "events.jsonl")
            all_events.extend(ev)
        if not all_events:
            print(f"  {label}: no events found")
            continue
        P = action_transition_matrix(all_events)
        transitions[label] = summarise_transition(P)
        transitions[label]["n_events"] = int(len(all_events))
        transitions[label]["actions"] = ACTIONS
        per_round[label] = {int(r): d.tolist() for r, d in per_round_action_dist(all_events).items()}
        print(f"\n{label}:  n_events={len(all_events)}")
        print("  transition matrix (rows = current action, cols = next):")
        for i, a in enumerate(ACTIONS):
            print(f"    {a:<10} -> [{P[i, 0]:.3f} {P[i, 1]:.3f} {P[i, 2]:.3f}]")
        print(f"  mean row entropy = {transitions[label]['mean_row_entropy']:.3f} bits")
        print(f"  off-diagonal mass = {transitions[label]['off_diagonal_mass']:.3f}")

    out1 = {
        "transition_matrices": transitions,
        "interpretation": (
            "Off-diagonal mass measures behavioural diversity; row entropy "
            "measures unpredictability of the next action. Mode-collapsed "
            "Condition A is expected to have low off-diagonal mass and "
            "low row entropy on the cooperate row."
        ),
        "audit_row": "C.transitions (new)",
    }
    (TABLES / "action_transitions.json").write_text(json.dumps(out1, indent=2))
    print(f"\n✓ {TABLES / 'action_transitions.json'}")

    # Per-round JSD vs uniform
    pi_uniform = np.array([1 / 3, 1 / 3, 1 / 3])
    jsd_traj: dict[str, dict[int, float]] = {}
    for label, rounds in per_round.items():
        jsd_traj[label] = {int(r): float(jsd(np.array(d), pi_uniform)) for r, d in sorted(rounds.items())}
    out2 = {
        "jsd_vs_uniform": jsd_traj,
        "interpretation": (
            "JSD(π_round, π_uniform) per round. Higher JSD = larger "
            "deviation from uniform (more biased). A flat trajectory means "
            "the action distribution is stable; a rising trajectory means "
            "mode collapse intensifies over time."
        ),
        "audit_row": "C.jsd (new)",
    }
    (TABLES / "per_round_jsd.json").write_text(json.dumps(out2, indent=2))
    print(f"✓ {TABLES / 'per_round_jsd.json'}")

    # Figures
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, (label, summary) in zip(axes, transitions.items()):
            P = np.array(summary["matrix"])
            im = ax.imshow(P, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(3))
            ax.set_xticklabels(ACTIONS)
            ax.set_yticks(range(3))
            ax.set_yticklabels(ACTIONS)
            ax.set_xlabel("Next action")
            ax.set_ylabel("Current action")
            ax.set_title(f"{label}\nrow entropy = {summary['mean_row_entropy']:.2f} bits")
            for i in range(3):
                for j in range(3):
                    ax.text(
                        j,
                        i,
                        f"{P[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if P[i, j] < 0.6 else "black",
                        fontsize=9,
                    )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Action-Transition Matrices — Mechanism of Grounding", fontsize=11)
        fig.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / "action_transitions.png", dpi=150)
        plt.close(fig)
        print(f"✓ {FIGURES / 'action_transitions.png'}")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = {"A_ungrounded": "#c1121f", "B_grounded": "#1e6091"}
        for label, traj in jsd_traj.items():
            rs = sorted(traj.keys())
            ys = [traj[r] for r in rs]
            ax.plot(rs, ys, marker="o", label=label, color=colors.get(label, "#444"))
        ax.set_xlabel("Round")
        ax.set_ylabel("JSD(π_round ‖ π_uniform)  (bits)")
        ax.set_title("Per-Round Action-Distribution Bias")
        ax.axhline(0, ls="--", color="#aaa", lw=0.6)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "per_round_jsd.png", dpi=150)
        plt.close(fig)
        print(f"✓ {FIGURES / 'per_round_jsd.png'}")
    except Exception as e:
        print(f"figure skipped: {e}")


if __name__ == "__main__":
    main()
