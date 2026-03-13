from agents.memory import MemoryItem


class SimulationKernel:
    def __init__(self, agents: list, world, logger) -> None:
        self.agents = agents
        self.world = world
        self.logger = logger
        self.agent_lookup = {agent.profile.agent_id: agent for agent in agents}

    def run(self, num_rounds: int) -> None:
        for _ in range(num_rounds):
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