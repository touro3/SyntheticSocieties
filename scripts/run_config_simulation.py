import argparse
import sys
from pathlib import Path
from metrics.summary import summarize_agents

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from bgf_logging.event_logger import EventLogger
from decision.mock_policy import MockPolicy
from environment.institutions import InstitutionManager
from environment.world import World
from environment.world_state import WorldState
from simulation.kernel import SimulationKernel
from utils.config import load_config
from utils.io import ensure_dir, save_json, save_yaml, set_global_seed
from population.generator import generate_population

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a BGF simulation from config.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def build_agents(config: dict) -> list[Agent]:
    n = config["simulation"]["population_size"]
    defaults = config["agent_defaults"]

    agents = []

    for i in range(n):
        profile = AgentProfile(
            agent_id=f"agent_{i}",
            age=25 + i,
            income=1000 + i * 100,
            education=defaults["education"],
            occupation=defaults["occupation"],
            location=defaults["location"],
            political_preference=defaults["political_preference"],
            risk_tolerance=defaults["risk_tolerance"],
            social_class=defaults["social_class"],
        )

        state = AgentState(
            wealth=defaults["initial_wealth"] + i * defaults["wealth_step"]
        )

        memory = MemoryBuffer(max_items=defaults["memory_size"])
        policy = MockPolicy()

        agents.append(
            Agent(
                profile=profile,
                state=state,
                memory=memory,
                policy=policy,
            )
        )

    return agents


def build_world(config: dict) -> World:
    return World(
        state=WorldState(
            public_signal=config["environment"]["public_signal"],
            prices=config["environment"]["prices"],
            resources=config["environment"]["resources"],
        ),
        institution_manager=InstitutionManager(),
    )


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
    }
    save_json(metadata, run_dir / "metadata.json")

    agents = generate_population(config)
    world = build_world(config)

    logger = EventLogger(run_dir / "events.jsonl", overwrite=True)

    kernel = SimulationKernel(
        agents=agents,
        world=world,
        logger=logger,
    )

    kernel.run(num_rounds=config["simulation"]["rounds"])

    summary = summarize_agents(agents)
    save_json(summary, run_dir / "summary.json")

    print(f"Experiment completed: {experiment_id}")
    print(f"Artifacts saved in: {run_dir}")

    for agent in agents:
        print(agent.profile.agent_id, agent.state)


if __name__ == "__main__":
    main()
