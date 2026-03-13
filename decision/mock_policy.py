from decision.schemas import ProposedAction


class MockPolicy:
    def propose_action(self, profile, state, memory, context, round_id: int) -> ProposedAction:
        if state.wealth > 100:
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