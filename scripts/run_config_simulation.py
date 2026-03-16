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
from population.generator import generate_population, generate_empirical_population
from simulation.kernel import SimulationKernel
from utils.config import load_config
from utils.io import ensure_dir, save_json, save_yaml, set_global_seed
from metrics.event_metrics import behavior_summary_from_events, load_events
from metrics.summary import merge_behavior_summary, summarize_agents
from decision.mock_policy import MockPolicy
from decision.random_policy import RandomPolicy
from decision.rule_based_policy import RuleBasedPolicy
from decision.data_driven_policy import DataDrivenPolicy



def parse_args():
    parser = argparse.ArgumentParser(description="Run a BGF simulation from config.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to YAML config file.",
    )
    # Capture any unknown arguments for dot-notation overrides
    args, overrides = parser.parse_known_args()
    return args, overrides


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply dot-notation overrides (e.g., 'policy.type=llm') to config dict."""
    for override in overrides:
        if "=" not in override:
            continue
        key_path, value = override.split("=", 1)
        
        # Try to parse value as int, float, or bool
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass # Keep as string
        
        # Traverse and set
        parts = key_path.split(".")
        curr = config
        for i, part in enumerate(parts[:-1]):
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
        curr[parts[-1]] = value
    return config


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

    if policy_type == "data_driven":
        return DataDrivenPolicy()

    if policy_type == "llm":
        from decision.llm_backend import LLMBackend
        from decision.llm_policy import LLMPolicy
        from bgf_logging.prompt_logger import PromptLogger

        llm_cfg = config.get("llm", {})
        experiment_id = config["project"]["experiment_id"]

        backend = LLMBackend.get_instance(
            model_id=llm_cfg.get("model_id", "mistralai/Mistral-7B-Instruct-v0.3"),
            dtype=llm_cfg.get("dtype", "float16"),
            device_map=llm_cfg.get("device_map", "auto"),
            max_new_tokens=llm_cfg.get("max_new_tokens", 256),
            temperature=llm_cfg.get("temperature", 0.7),
            cache_dir=llm_cfg.get("cache_dir"),
        )
        backend.load()

        prompt_logger = PromptLogger(
            output_path=Path("experiments") / experiment_id / "prompts.jsonl"
        )

        return LLMPolicy(
            backend=backend,
            memory_window=llm_cfg.get("memory_window", 5),
            temperature=llm_cfg.get("temperature", 0.7),
            max_retries=llm_cfg.get("max_retries", 2),
            prompt_logger=prompt_logger,
            perturbation_mode=config.get("perturbation", {}).get("mode"),
        )

    if policy_type == "template":
        from decision.template_policy import TemplatePolicy
        return TemplatePolicy()

    if policy_type == "ablated_llm":
        from decision.llm_backend import LLMBackend
        from decision.ablated_llm_policy import AblatedLLMPolicy
        from bgf_logging.prompt_logger import PromptLogger

        llm_cfg = config.get("llm", {})
        ablation_cfg = config.get("ablation", {})
        experiment_id = config["project"]["experiment_id"]

        backend = LLMBackend.get_instance(
            model_id=llm_cfg.get("model_id", "mistralai/Mistral-7B-Instruct-v0.3"),
            dtype=llm_cfg.get("dtype", "float16"),
            device_map=llm_cfg.get("device_map", "auto"),
            max_new_tokens=llm_cfg.get("max_new_tokens", 256),
            temperature=llm_cfg.get("temperature", 0.7),
            cache_dir=llm_cfg.get("cache_dir"),
        )
        backend.load()

        prompt_logger = PromptLogger(
            output_path=Path("experiments") / experiment_id / "prompts.jsonl"
        )

        return AblatedLLMPolicy(
            backend=backend,
            ablation=ablation_cfg.get("mode", "no_persona"),
            memory_window=llm_cfg.get("memory_window", 5),
            temperature=llm_cfg.get("temperature", 0.7),
            max_retries=llm_cfg.get("max_retries", 2),
            prompt_logger=prompt_logger,
        )

    raise ValueError(f"Unsupported policy type: {policy_type}")


def main() -> None:
    args, overrides = parse_args()
    config = load_config(args.config)
    
    if overrides:
        config = apply_overrides(config, overrides)
        print(f"Applied CLI overrides: {overrides}")

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
    pop_source = config.get("population", {}).get("source", "synthetic")

    if pop_source == "empirical":
        agents = generate_empirical_population(config, policy)
        print(f"Generated empirical population: {len(agents)} agents from ESS data")
    else:
        agents = generate_population(config, policy)
        print(f"Generated synthetic population: {len(agents)} agents")

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