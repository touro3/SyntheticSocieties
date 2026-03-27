"""Event-driven simulation kernel.

Orchestrates round execution by gathering agent proposals (sequentially or
batched) and delegating processing to RoundProcessor. The kernel owns only
orchestration; RoundProcessor owns validate → execute → update → log.
"""

from __future__ import annotations

from simulation.round_processor import RoundProcessor


class SimulationKernel:
    def __init__(self, agents: list, world, logger) -> None:
        self.agents = agents
        self.world = world
        self.logger = logger
        self.agent_lookup = {agent.profile.agent_id: agent for agent in agents}
        self._processor = RoundProcessor(
            world=world, agent_lookup=self.agent_lookup, logger=logger,
        )

    def run(self, num_rounds: int) -> None:
        for _ in range(num_rounds):
            if self._can_use_batched_mode():
                self.run_round_batched()
            else:
                self.run_round()

    def _can_use_batched_mode(self) -> bool:
        try:
            from decision.llm_policy import LLMPolicy
        except ImportError:
            return False

        if not self.agents or not hasattr(self.agents[0], "policy"):
            return False

        policy = self.agents[0].policy
        backend = getattr(policy, "backend", None)
        return isinstance(policy, LLMPolicy) and hasattr(backend, "generate_batch")

    def run_round(self) -> None:
        self.world.apply_exogenous_updates()
        round_id = self.world.state.round_id

        for agent in self.agents:
            world_context = self.world.get_agent_context(agent.profile.agent_id)
            perception = agent.perceive(
                world_snapshot=world_context,
                local_network={"neighbors": world_context.get("neighbors", [])},
            )

            proposed_action = agent.decide(context=perception, round_id=round_id)
            self._processor.process_agent_action(
                agent, proposed_action, round_id, perception=perception,
            )

    def run_round_batched(self) -> None:
        from decision.llm_policy import LLMPolicy
        from decision.output_parser import parse_llm_output
        from decision.prompt_builder import build_prompt, build_prompt_text

        sample_policy = self.agents[0].policy if self.agents and hasattr(self.agents[0], "policy") else None
        backend = getattr(sample_policy, "backend", None)

        if not isinstance(sample_policy, LLMPolicy) or not hasattr(backend, "generate_batch"):
            return self.run_round()

        policy = sample_policy

        self.world.apply_exogenous_updates()
        round_id = self.world.state.round_id

        agent_data = []
        messages_list = []

        for agent in self.agents:
            world_context = self.world.get_agent_context(agent.profile.agent_id)
            perception = agent.perceive(
                world_snapshot=world_context,
                local_network={"neighbors": world_context.get("neighbors", [])},
            )

            social_context = None
            if getattr(policy, "graph_rag", None):
                social_context = policy.graph_rag.get_social_context(agent.profile.agent_id)

            pop_context = None
            if getattr(policy, "sql_rag", None):
                pop_context = policy.sql_rag.get_peer_group_context(
                    age=agent.profile.age,
                    gender=agent.profile.gender,
                    country=agent.profile.country,
                )

            messages = build_prompt(
                profile=agent.profile,
                state=agent.state,
                memory=agent.memory,
                context=perception,
                round_id=round_id,
                memory_window=policy.memory_window,
                social_context=social_context,
                population_context=pop_context,
                ablation_level=getattr(policy, "ablation_level", 5),
            )

            if policy.perturbation_mode:
                from decision.prompt_perturbation import apply_perturbation

                seed = hash((round_id, agent.profile.agent_id)) % (2**31)
                messages = apply_perturbation(messages, mode=policy.perturbation_mode, seed=seed)

            agent_data.append(
                {
                    "agent": agent,
                    "perception": perception,
                    "neighbors": world_context.get("neighbors", []),
                    "social_context": social_context,
                    "pop_context": pop_context,
                }
            )
            messages_list.append(messages)

        batch_results = policy.backend.generate_batch(
            messages_list=messages_list,
            temperature=policy.temperature,
        )

        for i, (raw_text, latency) in enumerate(batch_results):
            agent = agent_data[i]["agent"]
            perception = agent_data[i]["perception"]
            neighbors = agent_data[i]["neighbors"]
            social_ctx = agent_data[i]["social_context"]
            pop_ctx = agent_data[i]["pop_context"]

            action, parse_meta = parse_llm_output(raw_text, neighbors)

            if action is None:
                action = policy._fallback_action(agent.state, neighbors)
                parse_meta["fallback"] = True

            if policy.prompt_logger:
                prompt_text = build_prompt_text(
                    agent.profile,
                    agent.state,
                    agent.memory,
                    perception,
                    round_id,
                    policy.memory_window,
                    social_context=social_ctx,
                    population_context=pop_ctx,
                    ablation_level=getattr(policy, "ablation_level", 5),
                )
                policy.prompt_logger.log(
                    round_id=round_id,
                    agent_id=agent.profile.agent_id,
                    prompt=prompt_text,
                    raw_output=raw_text,
                    parsed_action=action.model_dump() if action else None,
                    latency_ms=latency * 1000,
                    parse_metadata=parse_meta,
                )

            self._processor.process_agent_action(
                agent, action, round_id, perception=perception,
            )
