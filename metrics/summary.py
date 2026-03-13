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


def summarize_agents(agents: list[Agent]) -> dict:
    wealths = extract_agent_wealths(agents)
    stress = extract_agent_stress(agents)

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
        "actions": action_counts(agents),
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