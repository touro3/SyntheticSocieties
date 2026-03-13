from environment.world_state import WorldState


class World:
    def __init__(self, state: WorldState, institution_manager, network_manager=None) -> None:
        self.state = state
        self.institution_manager = institution_manager
        self.network_manager = network_manager

    def get_agent_context(self, agent_id: str) -> dict:
        neighbors = []
        if self.network_manager is not None:
            neighbors = self.network_manager.get_neighbors(agent_id)

        return {
            "public_signal": self.state.public_signal,
            "prices": self.state.prices,
            "resources": self.state.resources,
            "neighbors": neighbors,
        }

    def validate_action(self, action, agent, agent_lookup):
        return self.institution_manager.validate(action, agent, self.state, agent_lookup)

    def execute_action(self, action, agent, agent_lookup) -> dict:
        return self.institution_manager.execute(action, agent, self.state, agent_lookup)

    def apply_exogenous_updates(self) -> None:
        self.state.round_id += 1