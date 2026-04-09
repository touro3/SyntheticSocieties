from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bgf_logging.event_logger import EventLogger
from bgf_logging.prompt_logger import PromptLogger
from decision.auditable_random_policy import AuditableRandomPolicy
from decision.conditioned_llm_policy import ConditionedLLMPolicy
from decision.graph_rag import GraphRAG
from decision.llm_backend import LLMBackend
from environment.institutions import InstitutionManager
from environment.network import NetworkManager
from environment.world import World
from environment.world_state import WorldState
from metrics.event_metrics import behavior_summary_from_events, load_events
from metrics.summary import merge_behavior_summary, summarize_agents
from population.persona_synthesizer import load_persona_records, persona_records_to_agents
from simulation.kernel import SimulationKernel
from utils.io import ensure_dir, save_json, save_yaml, set_global_seed

PRIMARY_CONDITIONS = {
    "auditable_random_ess_persona",
    "pure_llm_ess_persona",
    "grounded_llm_ess_persona",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM grounding comparison matrix.")
    parser.add_argument("--artifacts-dir", required=True, type=str, help="Directory created by build_society_from_prompt.py")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 42,43,44")
    parser.add_argument("--model-id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--memory-window", type=int, default=5)
    parser.add_argument("--network-type", type=str, default="random")
    parser.add_argument("--edge-prob", type=float, default=0.4)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--rewiring-prob", type=float, default=0.3)
    parser.add_argument("--skip-random", action="store_true", help="Skip auditable random baseline.")
    return parser.parse_args()


def parse_seed_list(args) -> list[int]:
    if args.seeds:
        return [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    return [args.seed]


def build_network(agent_ids: list[str], args, seed: int) -> NetworkManager:
    if args.network_type == "fully_connected":
        return NetworkManager.fully_connected(agent_ids)
    if args.network_type == "random":
        return NetworkManager.random_graph(agent_ids=agent_ids, edge_prob=args.edge_prob, seed=seed)
    if args.network_type == "small_world":
        return NetworkManager.small_world(
            agent_ids=agent_ids,
            k=args.k,
            rewiring_prob=args.rewiring_prob,
            seed=seed,
        )
    raise ValueError(f"Unsupported network type: {args.network_type}")


def build_world(network_manager: NetworkManager) -> World:
    return World(
        state=WorldState(
            public_signal={"economy": "stable"},
            prices={"food": 1.0},
            resources={"jobs": 100.0},
        ),
        institution_manager=InstitutionManager(),
        network_manager=network_manager,
    )


def build_backend(args):
    backend = LLMBackend.get_instance(
        model_id=args.model_id,
        dtype="float16",
        device_map="auto",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        cache_dir=args.cache_dir,
        inference_timeout=120,
        max_retries=2,
    )
    backend.load()
    return backend


def condition_specs(include_random: bool) -> list[dict]:
    specs = [
        {
            "key": "pure_llm_ess_persona",
            "policy_family": "llm",
            "persona_source": "ess",
            "use_population_context": False,
            "use_social_context": False,
            "use_memory_context": False,
            "system_prompt_mode": "base",
            "include_balancing_hint": False,
        },
        {
            "key": "grounded_llm_ess_persona",
            "policy_family": "llm",
            "persona_source": "ess",
            "use_population_context": True,
            "use_social_context": True,
            "use_memory_context": True,
            "system_prompt_mode": "balanced",
            "include_balancing_hint": True,
        },
        {
            "key": "pure_llm_synth_persona",
            "policy_family": "llm",
            "persona_source": "synthetic",
            "use_population_context": False,
            "use_social_context": False,
            "use_memory_context": False,
            "system_prompt_mode": "base",
            "include_balancing_hint": False,
        },
        {
            "key": "grounded_llm_synth_persona",
            "policy_family": "llm",
            "persona_source": "synthetic",
            "use_population_context": True,
            "use_social_context": True,
            "use_memory_context": True,
            "system_prompt_mode": "balanced",
            "include_balancing_hint": True,
        },
    ]
    if include_random:
        specs.insert(
            0,
            {
                "key": "auditable_random_ess_persona",
                "policy_family": "auditable_random",
                "persona_source": "ess",
                "use_population_context": True,
                "use_social_context": False,
                "use_memory_context": False,
                "system_prompt_mode": "none",
                "include_balancing_hint": False,
            },
        )
    return specs


def build_policy(spec: dict, backend, population_context: str, grounding_report: dict, run_dir: Path, seed: int, args):
    if spec["policy_family"] == "auditable_random":
        return AuditableRandomPolicy(
            seed=seed,
            audit_path=run_dir / "random_audit.jsonl",
            cohort_summary=grounding_report.get("variable_summaries", {}),
            condition_name=spec["key"],
        )

    graph_rag = GraphRAG() if spec["use_social_context"] else None
    prompt_logger = PromptLogger(run_dir / "prompts.jsonl")
    extra_guidance = None
    if spec["key"].startswith("grounded_"):
        extra_guidance = "Use demographic grounding as a soft empirical prior rather than a rigid stereotype."

    return ConditionedLLMPolicy(
        backend=backend,
        memory_window=args.memory_window,
        temperature=args.temperature,
        max_retries=2,
        prompt_logger=prompt_logger,
        graph_rag=graph_rag,
        fixed_population_context=population_context,
        use_population_context=spec["use_population_context"],
        use_social_context=spec["use_social_context"],
        use_memory_context=spec["use_memory_context"],
        system_prompt_mode=spec["system_prompt_mode"],
        include_balancing_hint=spec["include_balancing_hint"],
        extra_guidance=extra_guidance,
        condition_name=spec["key"],
    )


def run_one_condition(spec: dict, seed: int, artifacts_dir: Path, backend, args, registry: list[dict]) -> None:
    experiment_id = f"{spec['key']}_s{seed}"
    run_dir = ensure_dir(Path("experiments") / experiment_id)

    with (artifacts_dir / "population_context.txt").open("r", encoding="utf-8") as f:
        population_context = f.read().strip()

    grounding_report = json.loads((artifacts_dir / "grounding_report.json").read_text(encoding="utf-8"))
    society_spec = json.loads((artifacts_dir / "society_spec.json").read_text(encoding="utf-8"))

    persona_file = "ess_personas.jsonl" if spec["persona_source"] == "ess" else "synthetic_personas.jsonl"
    persona_records = load_persona_records(artifacts_dir / persona_file)

    policy = build_policy(
        spec=spec,
        backend=backend,
        population_context=population_context,
        grounding_report=grounding_report,
        run_dir=run_dir,
        seed=seed,
        args=args,
    )
    agents = persona_records_to_agents(persona_records, policy, memory_size=10)

    network_manager = build_network([agent.profile.agent_id for agent in agents], args, seed)
    world = build_world(network_manager)
    logger = EventLogger(run_dir / "events.jsonl", overwrite=True)

    config = {
        "project": {
            "name": "grounding_matrix",
            "experiment_id": experiment_id,
            "seed": seed,
        },
        "condition": {
            **spec,
            "is_primary": spec["key"] in PRIMARY_CONDITIONS,
        },
        "simulation": {
            "rounds": args.rounds,
            "population_size": len(agents),
        },
        "artifacts": {
            "artifacts_dir": str(artifacts_dir),
            "persona_records_path": str(artifacts_dir / persona_file),
        },
        "network": {
            "type": args.network_type,
            "edge_prob": args.edge_prob,
            "k": args.k,
            "rewiring_prob": args.rewiring_prob,
        },
        "llm": {
            "model_id": args.model_id,
            "temperature": args.temperature,
            "memory_window": args.memory_window,
        },
    }

    save_yaml(config, run_dir / "config.yaml")
    save_json(society_spec, run_dir / "society_spec.json")
    save_json(grounding_report, run_dir / "grounding_report.json")

    kernel = SimulationKernel(agents=agents, world=world, logger=logger)
    kernel.run(num_rounds=args.rounds)

    summary = summarize_agents(agents)
    events = load_events(run_dir / "events.jsonl")
    event_behavior = behavior_summary_from_events(events)
    summary = merge_behavior_summary(summary, event_behavior)
    save_json(summary, run_dir / "summary.json")

    registry.append(
        {
            "experiment_id": experiment_id,
            "condition_key": spec["key"],
            "policy_family": spec["policy_family"],
            "persona_source": spec["persona_source"],
            "use_population_context": spec["use_population_context"],
            "use_social_context": spec["use_social_context"],
            "use_memory_context": spec["use_memory_context"],
            "system_prompt_mode": spec["system_prompt_mode"],
            "include_balancing_hint": spec["include_balancing_hint"],
            "seed": seed,
            "run_dir": str(run_dir),
            "is_primary": spec["key"] in PRIMARY_CONDITIONS,
        }
    )

    print(f"Condition completed: {experiment_id}")
    print(f"Artifacts saved in: {run_dir}")


def main():
    args = parse_args()
    seeds = parse_seed_list(args)
    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")

    include_random = not args.skip_random
    specs = condition_specs(include_random=include_random)

    needs_backend = any(spec["policy_family"] == "llm" for spec in specs)
    backend = build_backend(args) if needs_backend else None

    registry: list[dict] = []
    for seed in seeds:
        set_global_seed(seed)
        for spec in specs:
            run_one_condition(
                spec=spec,
                seed=seed,
                artifacts_dir=artifacts_dir,
                backend=backend,
                args=args,
                registry=registry,
            )

    save_json(registry, Path("experiments") / "grounding_matrix_registry.json")
    print("Condition matrix complete.")


if __name__ == "__main__":
    main()
