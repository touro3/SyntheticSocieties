from decision.constants import (
    COOPERATE_WEALTH_THRESHOLD,
    DEFAULT_COOPERATE_AMOUNT,
    DEFAULT_RULE_CONFIDENCE,
    DEFAULT_WORK_AMOUNT,
    WORK_WEALTH_THRESHOLD,
)
from decision.prompt_builder import get_neighbors
from decision.schemas import ProposedAction


class RuleBasedPolicy:
    def propose_action(self, profile, state, memory, context, round_id: int) -> ProposedAction:
        neighbors = get_neighbors(context)

        if state.wealth < WORK_WEALTH_THRESHOLD:
            return ProposedAction(
                action_type="work",
                amount=DEFAULT_WORK_AMOUNT,
                reasoning_summary="Rule-based policy works when wealth is low.",
                confidence=DEFAULT_RULE_CONFIDENCE,
            )

        if neighbors and state.wealth >= COOPERATE_WEALTH_THRESHOLD:
            return ProposedAction(
                action_type="cooperate",
                target_agent_id=neighbors[0],
                amount=DEFAULT_COOPERATE_AMOUNT,
                reasoning_summary="Rule-based policy cooperates when wealth is high.",
                confidence=DEFAULT_RULE_CONFIDENCE,
            )

        return ProposedAction(
            action_type="save",
            amount=DEFAULT_COOPERATE_AMOUNT,
            reasoning_summary="Rule-based policy saves under intermediate conditions.",
            confidence=DEFAULT_RULE_CONFIDENCE,
        )
