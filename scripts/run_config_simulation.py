import argparse
import inspect
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.agent import Agent
from bgf_logging.event_logger import EventLogger
from decision.data_driven_policy import DataDrivenPolicy
from decision.mock_policy import MockPolicy
from decision.random_policy import RandomPolicy
from decision.rule_based_ess_policy import RuleBasedESSPolicy
from decision.rule_based_policy import RuleBasedPolicy
from environment.institutions import InstitutionManager
from environment.network import NetworkManager
from environment.world import World
from environment.world_state import WorldState
from metrics.event_metrics import behavior_summary_from_events, load_events
from metrics.summary import merge_behavior_summary, summarize_agents
from population.generator import generate_empirical_population, generate_population
from simulation.kernel import SimulationKernel
from utils.config import load_config
from utils.io import ensure_dir, save_json, save_yaml, set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run a BGF simulation from config.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="EXP_ID",
        help=(
            "Resume an interrupted experiment. Pass the experiment ID "
            "(e.g. cmp_llm_s2). Loads experiments/<EXP_ID>/checkpoint.json "
            "and continues from the saved round."
        ),
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    for override in overrides:
        if "=" not in override:
            continue

        key_path, value = override.split("=", 1)

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
                pass

        parts = key_path.split(".")
        curr = config
        for part in parts[:-1]:
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


def _get_ess_clean_path(config: dict) -> str:
    return config.get("data", {}).get("ess_clean_path", "data/ess_clean.parquet")


_OPENAI_COMPAT_PROVIDERS = {
    # provider  → (base_url or None, env-var for key, key-fallback)
    "openai": (None,                                    "OPENAI_API_KEY",  None),
    "groq":   ("https://api.groq.com/openai/v1",       "GROQ_API_KEY",    None),
    "ollama": ("http://localhost:11434/v1",              None,              "ollama"),
}


def _build_llm_backend(llm_cfg: dict):
    import os as _os
    backend_type = llm_cfg.get("backend_type", "huggingface")

    if backend_type in _OPENAI_COMPAT_PROVIDERS:
        from decision.openai_backend import OpenAIBackend
        base_url, env_var, key_fallback = _OPENAI_COMPAT_PROVIDERS[backend_type]
        api_key = (
            llm_cfg.get("api_key")
            or ((_os.environ.get(env_var) or None) if env_var else None)
            or key_fallback
        )
        backend = OpenAIBackend(
            model_id=llm_cfg.get("model_id", "gpt-4o-mini"),
            max_new_tokens=llm_cfg.get("max_new_tokens", 256),
            temperature=llm_cfg.get("temperature", 0.7),
            max_retries=llm_cfg.get("max_retries", 2),
            api_key=api_key,
            base_url=base_url,
        )
        backend.load()
        return backend

    from decision.llm_backend import LLMBackend

    backend = LLMBackend.get_instance(
        model_id=llm_cfg.get("model_id", "mistralai/Mistral-7B-Instruct-v0.3"),
        dtype=llm_cfg.get("dtype", "float16"),
        device_map=llm_cfg.get("device_map", "auto"),
        max_new_tokens=llm_cfg.get("max_new_tokens", 128),
        temperature=llm_cfg.get("temperature", 0.7),
        cache_dir=llm_cfg.get("cache_dir"),
        inference_timeout=llm_cfg.get("inference_timeout", 120),
        max_retries=llm_cfg.get("max_retries", 2),
        quantization=llm_cfg.get("quantization", None),
    )
    if "max_batch_size" in llm_cfg:
        backend._max_batch_size = int(llm_cfg["max_batch_size"])
    backend.load()
    return backend


def _build_prompt_logger(experiment_id: str):
    from bgf_logging.prompt_logger import PromptLogger

    return PromptLogger(output_path=Path("experiments") / experiment_id / "prompts.jsonl")


def _build_rag_components(config: dict):
    from decision.graph_rag import GraphRAG
    from decision.sql_rag import SQLRAG

    graph_rag = GraphRAG()
    sql_rag = SQLRAG(data_path=_get_ess_clean_path(config))
    return graph_rag, sql_rag


def _attach_optional_rag_kwargs(policy_cls, policy_kwargs: dict, graph_rag, sql_rag) -> dict:
    params = inspect.signature(policy_cls.__init__).parameters
    if "graph_rag" in params:
        policy_kwargs["graph_rag"] = graph_rag
    if "sql_rag" in params:
        policy_kwargs["sql_rag"] = sql_rag
    return policy_kwargs


def build_policy(config: dict):
    policy_type = config["policy"]["type"]

    if policy_type == "mock":
        return MockPolicy()

    if policy_type == "random":
        return RandomPolicy()

    if policy_type == "rule_based":
        return RuleBasedPolicy()

    if policy_type == "rule_based_ess":
        return RuleBasedESSPolicy()

    if policy_type == "data_driven":
        return DataDrivenPolicy()

    if policy_type == "template":
        from decision.template_policy import TemplatePolicy

        return TemplatePolicy()

    if policy_type == "llm":
        from decision.llm_policy import LLMPolicy

        llm_cfg = config.get("llm", {})
        experiment_id = config["project"]["experiment_id"]

        backend = _build_llm_backend(llm_cfg)
        prompt_logger = _build_prompt_logger(experiment_id)
        graph_rag, sql_rag = _build_rag_components(config)

        return LLMPolicy(
            backend=backend,
            memory_window=llm_cfg.get("memory_window", 5),
            temperature=llm_cfg.get("temperature", 0.7),
            max_retries=llm_cfg.get("max_retries", 2),
            prompt_logger=prompt_logger,
            perturbation_mode=config.get("perturbation", {}).get("mode"),
            graph_rag=graph_rag,
            sql_rag=sql_rag,
        )

    if policy_type == "padded_ablation":
        from decision.padded_ablation_policy import PaddedAblationPolicy

        llm_cfg = config.get("llm", {})
        padded_cfg = config.get("padded", {})
        experiment_id = config["project"]["experiment_id"]

        backend = _build_llm_backend(llm_cfg)
        prompt_logger = _build_prompt_logger(experiment_id)

        return PaddedAblationPolicy(
            backend=backend,
            temperature=llm_cfg.get("temperature", 0.7),
            max_retries=llm_cfg.get("max_retries", 2),
            prompt_logger=prompt_logger,
            memory_window=llm_cfg.get("memory_window", 5),
            target_token_count=padded_cfg.get("target_token_count"),
        )

    if policy_type == "ablated_llm":
        from decision.ablated_llm_policy import AblatedLLMPolicy

        llm_cfg = config.get("llm", {})
        ablation_cfg = config.get("ablation", {})
        experiment_id = config["project"]["experiment_id"]

        backend = _build_llm_backend(llm_cfg)
        prompt_logger = _build_prompt_logger(experiment_id)
        graph_rag, sql_rag = _build_rag_components(config)

        policy_kwargs = {
            "backend": backend,
            "ablation": ablation_cfg.get("mode", "no_persona"),
            "memory_window": llm_cfg.get("memory_window", 5),
            "temperature": llm_cfg.get("temperature", 0.7),
            "max_retries": llm_cfg.get("max_retries", 2),
            "prompt_logger": prompt_logger,
            "perturbation_mode": config.get("perturbation", {}).get("mode"),
        }
        policy_kwargs = _attach_optional_rag_kwargs(
            AblatedLLMPolicy,
            policy_kwargs,
            graph_rag,
            sql_rag,
        )
        return AblatedLLMPolicy(**policy_kwargs)

    raise ValueError(f"Unsupported policy type: {policy_type}")


def run_simulation(config_path: str, overrides: list[str] | None = None, resume_exp_id: str | None = None) -> None:
    config = load_config(config_path)

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
        "ess_clean_path": _get_ess_clean_path(config),
    }
    save_json(metadata, run_dir / "metadata.json")

    # Write run_state.json immediately so /status can see the experiment
    # even while the policy / LLM model is still loading.
    from simulation.crash_recovery import RunStateManager as _EarlyRSM
    _early_run_mgr = _EarlyRSM(run_dir)
    if not (run_dir / "run_state.json").exists():
        _early_run_mgr.start(
            total_rounds=config["simulation"]["rounds"],
            experiment_id=experiment_id,
        )

    try:
        policy = build_policy(config)
    except Exception as exc:
        _early_run_mgr.fail(str(exc))
        raise

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
    from agents.collective_memory import CollectiveMemory
    from environment.social_env import SocialEnvironment

    collective_memory = CollectiveMemory()
    social_env = SocialEnvironment(state=world.state, network_manager=network_manager)

    kernel = SimulationKernel(
        agents=agents,
        world=world,
        logger=logger,
        heartbeat_path=run_dir / "heartbeat.json",
        collective_memory=collective_memory,
        social_env=social_env,
    )

    num_rounds = config["simulation"]["rounds"]
    start_round = 0

    if resume_exp_id:
        checkpoint_path = Path("experiments") / resume_exp_id / "checkpoint.json"
        if checkpoint_path.exists():
            start_round = kernel.load_checkpoint(checkpoint_path)
            print(f"Resumed from checkpoint: round {start_round} / {num_rounds}")
        else:
            print(f"Warning: no checkpoint found at {checkpoint_path}, starting fresh.")

    # ── Graph-enriched persona injection (post-resume) ────────────────────────
    # After a resume, the GraphRAG is rebuilt from events.jsonl by the policy
    # before run() starts.  We enrich all agent personas with social context
    # from the graph so the LLM receives relationship-aware persona blocks.
    if start_round > 0 and hasattr(policy, "graph_rag") and policy.graph_rag is not None:
        try:
            from population.persona_synthesizer import PersonaRecord, enrich_persona_from_graph

            for agent in agents:
                p = agent.profile
                rec = PersonaRecord(
                    agent_id=p.agent_id,
                    age=p.age,
                    income=getattr(p, "income", 0.0),
                    education=getattr(p, "education", ""),
                    occupation=getattr(p, "occupation", "worker"),
                    location=getattr(p, "location", "urban"),
                    political_preference=str(getattr(p, "political_preference", "center")),
                    risk_tolerance=getattr(p, "risk_tolerance", 0.5) or 0.5,
                    social_class=getattr(p, "social_class", "middle"),
                    initial_wealth=agent.state.wealth,
                    gender=getattr(p, "gender", None),
                    country=getattr(p, "country", None),
                    trust_people=getattr(p, "trust_people", None),
                    trust_institutions=getattr(p, "trust_institutions", None),
                    life_satisfaction=getattr(p, "life_satisfaction", None),
                    social_activity=getattr(p, "social_activity", None),
                )
                enriched = enrich_persona_from_graph(rec, policy.graph_rag)
                if hasattr(p, "persona_text"):
                    p.persona_text = enriched
            print(f"Graph-enriched personas applied to {len(agents)} agents.")
        except Exception as exc:
            print(f"Warning: graph persona enrichment skipped ({exc})")

    # ── Graceful shutdown wiring ──────────────────────────────────────────────
    from simulation.crash_recovery import RunStateManager
    from simulation.signal_handler import GracefulShutdown

    run_mgr = _early_run_mgr  # reuse the manager that wrote the early run_state.json
    if start_round == 0 and run_mgr._state is None:
        run_mgr.start(total_rounds=num_rounds, experiment_id=experiment_id)
    shutdown = GracefulShutdown()
    shutdown.register()

    from simulation.ipc import SimulationIPCServer

    ipc_server = SimulationIPCServer(
        agents=kernel.agent_lookup,
        base_dir=run_dir,
        current_round_fn=lambda: world.state.round_id,
        world_state=world.state,
    )
    ipc_server.start()

    try:
        completed = kernel.run(num_rounds=num_rounds, start_round=start_round, stop_flag=shutdown)
        run_mgr.tick(start_round + completed)
        if shutdown.requested:
            run_mgr.fail(f"Interrupted by {shutdown.signal_name}")
            print(f"Run interrupted after round {start_round + completed} — checkpoint saved.")
        else:
            run_mgr.complete()
    except Exception as exc:
        run_mgr.fail(str(exc))
        raise
    finally:
        ipc_server.stop()
        shutdown.unregister()

    summary = summarize_agents(agents)
    events = load_events(run_dir / "events.jsonl")
    event_behavior = behavior_summary_from_events(events)
    summary = merge_behavior_summary(summary, event_behavior)

    if social_env is not None:
        from metrics.social_metrics import engagement_rate, network_amplification, post_diversity

        summary["social_metrics"] = {
            "engagement_rate": round(engagement_rate(social_env), 4),
            "post_diversity": round(post_diversity(social_env), 4),
            "network_amplification": round(network_amplification(social_env), 4),
        }

    save_json(summary, run_dir / "summary.json")

    print(f"Experiment completed: {experiment_id}")
    print(f"Artifacts saved in: {run_dir}")

    for agent in agents:
        print(agent.profile.agent_id, agent.state)


def main() -> None:
    args, overrides = parse_args()
    run_simulation(args.config, overrides, resume_exp_id=args.resume)


if __name__ == "__main__":
    main()
