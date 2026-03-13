from __future__ import annotations

from agents.agent import Agent
from metrics.descriptive import maximum, mean, median, minimum, variance
from metrics.inequality import gini_coefficient, lorenz_curve


def extract_agent_wealths(agents: list[Agent]) -> list[float]:
    return [float(agent.state.wealth) for agent in agents]


def extract_agent_stress(agents: list[Agent]) -> list[float]:
    return [float(agent.state.stress) for agent in agents]


def action_counts(agents: list[Agent]) -> dict[str, int]:
    counts: dict[str, int] = {}

    for agent in agents:
        action = agent.state.last_action or "none"
        counts[action] = counts.get(action, 0) + 1

    return counts


def cooperation_rate(agents: list[Agent]) -> float:
    counts = action_counts(agents)
    total = sum(counts.values())

    if total == 0:
        return 0.0

    cooperative_actions = counts.get("cooperate", 0)
    return cooperative_actions / total


def defection_rate(agents: list[Agent]) -> float:
    counts = action_counts(agents)
    total = sum(counts.values())

    if total == 0:
        return 0.0

    defect_actions = counts.get("decline_help", 0)
    return defect_actions / total


def summarize_agents(agents: list[Agent]) -> dict:
    wealths = extract_agent_wealths(agents)
    stress = extract_agent_stress(agents)
    counts = action_counts(agents)

    return {
        "num_agents": len(agents),
        "wealth": {
            "values": wealths,
            "mean": mean(wealths),
            "median": median(wealths),
            "variance": variance(wealths),
            "min": minimum(wealths),
            "max": maximum(wealths),
            "gini": gini_coefficient(wealths),
            "lorenz_curve": lorenz_curve(wealths),
        },
        "stress": {
            "values": stress,
            "mean": mean(stress),
            "median": median(stress),
            "variance": variance(stress),
            "min": minimum(stress),
            "max": maximum(stress),
        },
        "actions": counts,
        "behavior": {
            "cooperation_rate": cooperation_rate(agents),
            "defection_rate": defection_rate(agents),
        },
        "final_state": {
            agent.profile.agent_id: {
                "wealth": agent.state.wealth,
                "stress": agent.state.stress,
                "satisfaction": agent.state.satisfaction,
                "last_action": agent.state.last_action,
            }
            for agent in agents
        },
    }

def merge_behavior_summary(summary: dict, behavior_summary: dict) -> dict:
    merged = dict(summary)
    merged.update(behavior_summary)
    return merged