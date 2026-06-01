"""
Bias & failure diagnostics for LLM agent simulations.

Provides 4 diagnostic analyses:
  - Subgroup analysis: compare outcomes by demographic groups
  - Persona drift detection: measure JSD between initial persona and behavior
  - Response diversity: Shannon entropy of action distributions
  - Alignment bias detection: check systematic action preferences
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.stats import entropy


def subgroup_analysis(
    events: list[dict],
    agents: list,
    group_by: str = "gender",
) -> dict:
    """
    Compare outcomes across demographic subgroups.

    Args:
        events: List of event dicts from simulation log.
        agents: List of Agent objects.
        group_by: Profile attribute to group by (e.g., 'gender', 'country').

    Returns:
        Dict with per-group wealth, cooperation rate, and action distribution.
    """
    groups: dict[str, list] = defaultdict(list)

    for agent in agents:
        group_val = getattr(agent.profile, group_by, "unknown")
        groups[str(group_val)].append(agent)

    results = {}
    for group_name, group_agents in groups.items():
        group_ids = {a.profile.agent_id for a in group_agents}

        wealths = [a.state.wealth for a in group_agents]
        group_events = [e for e in events if e.get("agent_id") in group_ids]
        actions = [e.get("action", {}).get("action_type", "unknown") for e in group_events if "action" in e]

        action_counts = Counter(actions)
        total = max(len(actions), 1)

        results[group_name] = {
            "n_agents": len(group_agents),
            "mean_wealth": float(np.mean(wealths)) if wealths else 0.0,
            "std_wealth": float(np.std(wealths)) if wealths else 0.0,
            "cooperation_rate": action_counts.get("cooperate", 0) / total,
            "action_distribution": {k: v / total for k, v in action_counts.items()},
        }

    return results


def persona_drift_detection(
    events: list[dict],
    agents: list,
    window_size: int = 5,
) -> dict:
    """
    Detect persona drift by comparing early vs late behavior per agent.

    Measures Jensen–Shannon divergence between action distributions
    in the first and last `window_size` rounds for each agent.
    """
    agent_actions: dict[str, dict[str, list]] = defaultdict(lambda: {"early": [], "late": []})

    # Group by agent and round
    for e in events:
        agent_id = e.get("agent_id")
        round_id = e.get("round_id", 0)
        action = e.get("action", {}).get("action_type")
        if agent_id and action:
            agent_actions[agent_id]["all"] = agent_actions[agent_id].get("all", [])
            agent_actions[agent_id]["all"].append((round_id, action))

    max_round = max(
        (r for aa in agent_actions.values() for r, _ in aa.get("all", [(0, "")])),
        default=0,
    )

    results = {}
    all_actions = sorted(set(a for aa in agent_actions.values() for _, a in aa.get("all", [])))

    for agent_id, data in agent_actions.items():
        rounds_actions = sorted(data.get("all", []), key=lambda x: x[0])

        early = [a for r, a in rounds_actions if r <= window_size]
        late = [a for r, a in rounds_actions if r > max_round - window_size]

        if not early or not late:
            results[agent_id] = {"drift_jsd": 0.0, "drifted": False}
            continue

        early_dist = np.array([early.count(a) for a in all_actions], dtype=float)
        late_dist = np.array([late.count(a) for a in all_actions], dtype=float)

        eps = 1e-10
        p = (early_dist + eps) / (early_dist + eps).sum()
        q = (late_dist + eps) / (late_dist + eps).sum()
        m = (p + q) / 2
        jsd = float(entropy(m) - (entropy(p) + entropy(q)) / 2)

        results[agent_id] = {
            "drift_jsd": jsd,
            "drifted": jsd > 0.1,
            "early_actions": dict(Counter(early)),
            "late_actions": dict(Counter(late)),
        }

    return results


def response_diversity(events: list[dict]) -> dict:
    """
    Measure response diversity per agent via Shannon entropy.

    Higher entropy = more diverse action selection.
    """
    agent_actions: dict[str, list] = defaultdict(list)

    for e in events:
        agent_id = e.get("agent_id")
        action = e.get("action", {}).get("action_type")
        if agent_id and action:
            agent_actions[agent_id].append(action)

    results = {}
    for agent_id, actions in agent_actions.items():
        counts = Counter(actions)
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        ent = float(entropy(probs, base=2))

        results[agent_id] = {
            "entropy": ent,
            "max_entropy": float(np.log2(max(len(counts), 1))),
            "normalized_entropy": ent / max(np.log2(max(len(counts), 1)), 1e-10),
            "action_counts": dict(counts),
            "n_unique_actions": len(counts),
        }

    # Aggregate
    entropies = [r["entropy"] for r in results.values()]
    results["_aggregate"] = {
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "std_entropy": float(np.std(entropies)) if entropies else 0.0,
        "min_entropy": float(np.min(entropies)) if entropies else 0.0,
        "max_entropy": float(np.max(entropies)) if entropies else 0.0,
    }

    return results


def alignment_bias_detection(
    events: list[dict],
    agents: list,
) -> dict:
    """
    Check if the LLM systematically favors certain actions regardless of persona.

    Computes per-persona-attribute correlation between attributes and action choices.
    A high bias score means the LLM ignores persona variation.
    """
    agent_action_rates: dict[str, dict] = {}

    for agent in agents:
        aid = agent.profile.agent_id
        agent_events = [e for e in events if e.get("agent_id") == aid]
        actions = [e.get("action", {}).get("action_type") for e in agent_events if "action" in e]
        total = max(len(actions), 1)
        agent_action_rates[aid] = {
            "work_rate": actions.count("work") / total,
            "save_rate": actions.count("save") / total,
            "cooperate_rate": actions.count("cooperate") / total,
            "trust": getattr(agent.profile, "trust_people", None),
            "risk": getattr(agent.profile, "risk_tolerance", None),
            "competitiveness": getattr(agent.profile, "competitiveness", None),
        }

    # Check if high-trust agents actually cooperate more
    trust_values = []
    coop_values = []
    for rates in agent_action_rates.values():
        if rates["trust"] is not None:
            trust_values.append(rates["trust"])
            coop_values.append(rates["cooperate_rate"])

    trust_coop_corr = float(np.corrcoef(trust_values, coop_values)[0, 1]) if len(trust_values) > 2 else 0.0

    # Check action distribution uniformity (bias = low variance across agents)
    all_work = [r["work_rate"] for r in agent_action_rates.values()]
    all_coop = [r["cooperate_rate"] for r in agent_action_rates.values()]

    return {
        "trust_cooperation_correlation": trust_coop_corr,
        "persona_responsive": abs(trust_coop_corr) > 0.3,
        "work_rate_variance": float(np.var(all_work)) if all_work else 0.0,
        "cooperate_rate_variance": float(np.var(all_coop)) if all_coop else 0.0,
        "potential_bias": float(np.var(all_work)) < 0.01 and float(np.var(all_coop)) < 0.01,
        "per_agent_rates": agent_action_rates,
    }
