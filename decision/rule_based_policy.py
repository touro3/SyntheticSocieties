from decision.schemas import ProposedAction


class RuleBasedPolicy:
    def propose_action(self, profile, state, memory, context, round_id: int) -> ProposedAction:
        neighbors = context.get("network", {}).get("neighbors", [])

        if state.wealth < 70:
            return ProposedAction(
                action_type="work",
                amount=10.0,
                reasoning_summary="Rule-based policy works when wealth is low.",
                confidence=0.9,
            )

        if neighbors and state.wealth >= 100:
            return ProposedAction(
                action_type="cooperate",
                target_agent_id=neighbors[0],
                amount=5.0,
                reasoning_summary="Rule-based policy cooperates when wealth is high.",
                confidence=0.9,
            )

        return ProposedAction(
            action_type="save",
            amount=5.0,
            reasoning_summary="Rule-based policy saves under intermediate conditions.",
            confidence=0.9,
        )