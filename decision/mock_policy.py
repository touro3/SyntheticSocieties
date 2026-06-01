from decision.prompt_builder import get_neighbors
from decision.schemas import ProposedAction


class MockPolicy:
    def propose_action(self, profile, state, memory, context, round_id: int) -> ProposedAction:
        neighbors = get_neighbors(context)

        if neighbors and state.wealth >= 100:
            target = neighbors[0]
            return ProposedAction(
                action_type="cooperate",
                target_agent_id=target,
                amount=5.0,
                reasoning_summary="Agent cooperates by transferring resources to a neighbor.",
                confidence=0.85,
            )

        if state.wealth > 60:
            return ProposedAction(
                action_type="save",
                amount=5.0,
                reasoning_summary="Agent chooses to save part of current wealth.",
                confidence=0.9,
            )

        return ProposedAction(
            action_type="work",
            amount=10.0,
            reasoning_summary="Agent chooses to work to increase resources.",
            confidence=0.8,
        )
