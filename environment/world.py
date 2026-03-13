from environment.world_state import WorldState


class World:
    def __init__(self, state: WorldState, institution_manager) -> None:
        self.state = state
        self.institution_manager = institution_manager

    def get_agent_context(self, agent_id: str) -> dict:
        return {
            "public_signal": self.state.public_signal,
            "prices": self.state.prices,
            "resources": self.state.resources,
            "neighbors": [],
        }

    def validate_action(self, action, agent):
        return self.institution_manager.validate(action, agent, self.state)

    def execute_action(self, action, agent) -> dict:
        return self.institution_manager.execute(action, agent, self.state)

    def apply_exogenous_updates(self) -> None:
        self.state.round_id += 1