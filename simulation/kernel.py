from agents.memory import MemoryItem


class SimulationKernel:
    def __init__(self, agents: list, world, logger) -> None:
        self.agents = agents
        self.world = world
        self.logger = logger
        self.agent_lookup = {agent.profile.agent_id: agent for agent in agents}

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

    def _record_memory(self, agent, proposed_action, executed_event, round_id: int) -> None:
        agent.memory.add(
            MemoryItem(
                round_id=round_id,
                partner_id=proposed_action.target_agent_id,
                event_type=proposed_action.action_type,
                content=proposed_action.reasoning_summary,
                outcome=executed_event,
            )
        )

    def _update_graph_rag(self, policy, agent, proposed_action, round_id: int) -> None:
        graph_rag = getattr(policy, "graph_rag", None)
        if graph_rag is None:
            return

        if proposed_action.action_type != "cooperate" or not proposed_action.target_agent_id:
            return

        graph_rag.add_event(
            {
                "round_id": round_id,
                "agent_id": agent.profile.agent_id,
                "action": proposed_action.model_dump(),
            }
        )

    def _log_event(self, round_id: int, agent, perception, proposed_action, validation, executed_event) -> None:
        self.logger.log_event(
            {
                "round_id": round_id,
                "agent_id": agent.profile.agent_id,
                "perception": perception,
                "action": proposed_action.model_dump(),
                "validation": validation.model_dump(),
                "result": executed_event,
                "state_after": {
                    "wealth": agent.state.wealth,
                    "stress": agent.state.stress,
                    "satisfaction": agent.state.satisfaction,
                    "last_action": agent.state.last_action,
                },
            }
        )

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
            validation = self.world.validate_action(proposed_action, agent, self.agent_lookup)

            if validation.valid:
                executed_event = self.world.execute_action(proposed_action, agent, self.agent_lookup)
                agent.apply_local_update(executed_event)
                self._record_memory(agent, proposed_action, executed_event, round_id)
                self._update_graph_rag(agent.policy, agent, proposed_action, round_id)
            else:
                executed_event = {
                    "agent_id": agent.profile.agent_id,
                    "action_type": "rejected",
                    "reason": validation.reason,
                    "round_id": round_id,
                }

            self._log_event(
                round_id=round_id,
                agent=agent,
                perception=perception,
                proposed_action=proposed_action,
                validation=validation,
                executed_event=executed_event,
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

            proposed_action = action
            validation = self.world.validate_action(proposed_action, agent, self.agent_lookup)

            if validation.valid:
                executed_event = self.world.execute_action(proposed_action, agent, self.agent_lookup)
                agent.apply_local_update(executed_event)
                self._record_memory(agent, proposed_action, executed_event, round_id)
                self._update_graph_rag(policy, agent, proposed_action, round_id)
            else:
                executed_event = {
                    "agent_id": agent.profile.agent_id,
                    "action_type": "rejected",
                    "reason": validation.reason,
                    "round_id": round_id,
                }

            self._log_event(
                round_id=round_id,
                agent=agent,
                perception=perception,
                proposed_action=proposed_action,
                validation=validation,
                executed_event=executed_event,
            )
