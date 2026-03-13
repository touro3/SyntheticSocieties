import random

from decision.schemas import ProposedAction


class RandomPolicy:
    def propose_action(self, profile, state, memory, context, round_id: int) -> ProposedAction:
        neighbors = context.get("network", {}).get("neighbors", [])

        possible_actions = ["work", "save"]

        if neighbors and state.wealth >= 5.0:
            possible_actions.append("cooperate")

        action_type = random.choice(possible_actions)

        if action_type == "cooperate":
            target = random.choice(neighbors)
            return ProposedAction(
                action_type="cooperate",
                target_agent_id=target,
                amount=5.0,
                reasoning_summary="Random baseline selected cooperation.",
                confidence=0.5,
            )

        if action_type == "save":
            return ProposedAction(
                action_type="save",
                amount=5.0,
                reasoning_summary="Random baseline selected saving.",
                confidence=0.5,
            )

        return ProposedAction(
            action_type="work",
            amount=10.0,
            reasoning_summary="Random baseline selected work.",
            confidence=0.5,
        )