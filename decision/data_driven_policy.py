from decision.schemas import ProposedAction
from models.behavioral_model import BehavioralModel


class DataDrivenPolicy:

    def __init__(self):
        self.model = BehavioralModel()

    def propose_action(self, profile, state, memory, context, round_id):

        world = context["world"]

        utilities = self.model.utilities(profile, state, world)

        probs = self.model.probabilities(utilities)

        action = self.model.choose(probs)

        if action == "work":
            return ProposedAction(
                action_type="work",
                amount=10.0,
                reasoning_summary="utility maximization",
                confidence=max(probs),
            )

        if action == "save":
            return ProposedAction(
                action_type="save",
                amount=5.0,
                reasoning_summary="utility maximization",
                confidence=max(probs),
            )

        neighbors = context["network"]["neighbors"]

        target = neighbors[0] if neighbors else None

        return ProposedAction(
            action_type="cooperate",
            target_agent_id=target,
            amount=5.0,
            reasoning_summary="utility maximization",
            confidence=max(probs),
        )