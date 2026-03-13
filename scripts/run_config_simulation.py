import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.agent import Agent
from bgf_logging.event_logger import EventLogger
from environment.institutions import InstitutionManager
from environment.network import NetworkManager
from environment.world import World
from environment.world_state import WorldState
from metrics.summary import summarize_agents
from population.generator import generate_population
from simulation.kernel import SimulationKernel
from utils.config import load_config
from utils.io import ensure_dir, save_json, save_yaml, set_global_seed
from metrics.event_metrics import behavior_summary_from_events, load_events
from metrics.summary import merge_behavior_summary, summarize_agents
from decision.mock_policy import MockPolicy
from decision.random_policy import RandomPolicy
from decision.rule_based_policy import RuleBasedPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a BGF simulation from config.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def build_network(config: dict, agents: list[Agent]) -> NetworkManager:
    network_cfg = config["network"]
    agent_ids = [agent.profile.agent_id for agent in agents]

    if network_cfg["type"] == "fully_connected":
        return NetworkManager.fully_connected(agent_ids)

    if network_cfg["type"] == "random":
        return NetworkManager.random_graph(
            agent_ids=agent_ids,
            edge_prob=network_cfg["edge_prob"],
            seed=config["project"]["seed"],
        )

    if network_cfg["type"] == "small_world":
        return NetworkManager.small_world(
            agent_ids=agent_ids,
            k=network_cfg["k"],
            rewiring_prob=network_cfg["rewiring_prob"],
            seed=config["project"]["seed"],
        )

    raise ValueError(f"Unsupported network type: {network_cfg['type']}")


def build_world(config: dict, network_manager: NetworkManager) -> World:
    return World(
        state=WorldState(
            public_signal=config["environment"]["public_signal"],
            prices=config["environment"]["prices"],
            resources=config["environment"]["resources"],
        ),
        institution_manager=InstitutionManager(),
        network_manager=network_manager,
    )

def build_policy(config: dict):
    policy_type = config["policy"]["type"]

    if policy_type == "mock":
        return MockPolicy()

    if policy_type == "random":
        return RandomPolicy()

    if policy_type == "rule_based":
        return RuleBasedPolicy()

    raise ValueError(f"Unsupported policy type: {policy_type}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment_id = config["project"]["experiment_id"]
    seed = config["project"]["seed"]

    set_global_seed(seed)

    run_dir = ensure_dir(Path("experiments") / experiment_id)

    save_yaml(config, run_dir / "config.yaml")

    metadata = {
        "project_name": config["project"]["name"],
        "experiment_id": experiment_id,
        "seed": seed,
        "policy_type": config["policy"]["type"],
        "population_size": config["simulation"]["population_size"],
        "rounds": config["simulation"]["rounds"],
        "network_type": config["network"]["type"],
        "network_edge_prob": config["network"].get("edge_prob"),
    }

    save_json(metadata, run_dir / "metadata.json")

    policy = build_policy(config)
    agents = generate_population(config, policy)

    network_manager = build_network(config, agents)

    world = build_world(config, network_manager)

    logger = EventLogger(run_dir / "events.jsonl", overwrite=True)

    kernel = SimulationKernel(
        agents=agents,
        world=world,
        logger=logger,
    )

    kernel.run(num_rounds=config["simulation"]["rounds"])

    summary = summarize_agents(agents)

    events = load_events(run_dir / "events.jsonl")
    event_behavior = behavior_summary_from_events(events)

    summary = merge_behavior_summary(summary, event_behavior)

    save_json(summary, run_dir / "summary.json")

    print(f"Experiment completed: {experiment_id}")
    print(f"Artifacts saved in: {run_dir}")

    for agent in agents:
        print(agent.profile.agent_id, agent.state)


if __name__ == "__main__":
    main()