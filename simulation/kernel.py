from agents.memory import MemoryItem


class SimulationKernel:
    def __init__(self, agents: list, world, logger) -> None:
        self.agents = agents
        self.world = world
        self.logger = logger
        self.agent_lookup = {agent.profile.agent_id: agent for agent in agents}

    def run(self, num_rounds: int) -> None:
        # Auto-detect if we can use batched mode (LLM policy)
        use_batch = False
        try:
            from decision.llm_policy import LLMPolicy
            if self.agents and hasattr(self.agents[0], "policy"):
                if isinstance(self.agents[0].policy, LLMPolicy):
                    use_batch = True
        except ImportError:
            pass

        for _ in range(num_rounds):
            if use_batch:
                self.run_round_batched()
            else:
                self.run_round()

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

                agent.memory.add(
                    MemoryItem(
                        round_id=round_id,
                        partner_id=proposed_action.target_agent_id,
                        event_type=proposed_action.action_type,
                        content=proposed_action.reasoning_summary,
                        outcome=executed_event,
                    )
                )
            else:
                executed_event = {
                    "agent_id": agent.profile.agent_id,
                    "action_type": "rejected",
                    "reason": validation.reason,
                    "round_id": round_id,
                }

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

    def run_round_batched(self) -> None:
        """
        Batched round execution for LLM policies.

        Phase 1: Collect all agent perceptions and build prompts
        Phase 2: Batch-generate all LLM decisions in one forward pass
        Phase 3: Apply all actions sequentially (maintains consistency)

        This is 3-5x faster than sequential for LLM experiments.
        """
        from decision.llm_policy import LLMPolicy
        from decision.prompt_builder import build_prompt, build_prompt_text
        from decision.output_parser import parse_llm_output

        self.world.apply_exogenous_updates()
        round_id = self.world.state.round_id

        # Check if any agent uses LLM policy with batch support
        sample_policy = self.agents[0].policy if hasattr(self.agents[0], "policy") else None
        if not isinstance(sample_policy, LLMPolicy) or not hasattr(sample_policy.backend, "generate_batch"):
            # Fallback to sequential
            return self.run_round()

        policy = sample_policy  # All agents share same policy

        # Phase 1: Collect perceptions and build prompts
        agent_data = []
        messages_list = []

        for agent in self.agents:
            world_context = self.world.get_agent_context(agent.profile.agent_id)
            perception = agent.perceive(
                world_snapshot=world_context,
                local_network={"neighbors": world_context.get("neighbors", [])},
            )

            messages = build_prompt(
                profile=agent.profile,
                state=agent.state,
                memory=agent.memory,
                context=perception,
                round_id=round_id,
                memory_window=policy.memory_window,
            )

            # Apply perturbation if configured
            if policy.perturbation_mode:
                from decision.prompt_perturbation import apply_perturbation
                seed = hash((round_id, agent.profile.agent_id)) % (2**31)
                messages = apply_perturbation(messages, mode=policy.perturbation_mode, seed=seed)

            agent_data.append({
                "agent": agent,
                "perception": perception,
                "messages": messages,
                "neighbors": world_context.get("neighbors", []),
            })
            messages_list.append(messages)

        # Phase 2: Batch generate all decisions
        batch_results = policy.backend.generate_batch(
            messages_list=messages_list,
            temperature=policy.temperature,
        )

        # Phase 3: Apply results sequentially
        for i, (raw_text, latency) in enumerate(batch_results):
            agent = agent_data[i]["agent"]
            perception = agent_data[i]["perception"]
            neighbors = agent_data[i]["neighbors"]

            action, parse_meta = parse_llm_output(raw_text, neighbors)

            if action is None:
                action = policy._fallback_action(agent.state, neighbors)
                parse_meta["fallback"] = True

            # Log prompt + output
            if policy.prompt_logger:
                prompt_text = build_prompt_text(
                    agent.profile, agent.state, agent.memory, perception,
                    round_id, policy.memory_window,
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

                agent.memory.add(
                    MemoryItem(
                        round_id=round_id,
                        partner_id=proposed_action.target_agent_id,
                        event_type=proposed_action.action_type,
                        content=proposed_action.reasoning_summary,
                        outcome=executed_event,
                    )
                )
            else:
                executed_event = {
                    "agent_id": agent.profile.agent_id,
                    "action_type": "rejected",
                    "reason": validation.reason,
                    "round_id": round_id,
                }

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