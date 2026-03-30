import random

from decision.constants import (
    DEFAULT_COOPERATE_AMOUNT,
    DEFAULT_RANDOM_CONFIDENCE,
    DEFAULT_WORK_AMOUNT,
    MIN_COOPERATE_WEALTH,
)
from decision.prompt_builder import get_neighbors
from decision.schemas import ProposedAction


class RandomPolicy:
    def propose_action(self, profile, state, memory, context, round_id: int) -> ProposedAction:
        neighbors = get_neighbors(context)

        possible_actions = ["work", "save"]

        if neighbors and state.wealth >= MIN_COOPERATE_WEALTH:
            possible_actions.append("cooperate")

        action_type = random.choice(possible_actions)

        if action_type == "cooperate":
            target = random.choice(neighbors)
            return ProposedAction(
                action_type="cooperate",
                target_agent_id=target,
                amount=DEFAULT_COOPERATE_AMOUNT,
                reasoning_summary="Random baseline selected cooperation.",
                confidence=DEFAULT_RANDOM_CONFIDENCE,
            )

        if action_type == "save":
            return ProposedAction(
                action_type="save",
                amount=DEFAULT_COOPERATE_AMOUNT,
                reasoning_summary="Random baseline selected saving.",
                confidence=DEFAULT_RANDOM_CONFIDENCE,
            )

        return ProposedAction(
            action_type="work",
            amount=DEFAULT_WORK_AMOUNT,
            reasoning_summary="Random baseline selected work.",
            confidence=DEFAULT_RANDOM_CONFIDENCE,
        )
