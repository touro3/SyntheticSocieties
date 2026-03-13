from pydantic import BaseModel

from decision.schemas import ProposedAction


class ValidationResult(BaseModel):
    valid: bool
    reason: str | None = None


class InstitutionManager:
    def validate(self, action: ProposedAction, agent, world_state) -> ValidationResult:
        if action.amount is not None and action.amount < 0:
            return ValidationResult(valid=False, reason="negative_amount")
        return ValidationResult(valid=True)

    def execute(self, action: ProposedAction, agent, world_state) -> dict:
        event = {
            "agent_id": agent.profile.agent_id,
            "action_type": action.action_type,
            "wealth_delta": 0.0,
            "stress_delta": 0.0,
            "round_id": world_state.round_id,
        }

        if action.action_type == "work":
            event["wealth_delta"] = float(action.amount or 10.0)
            event["stress_delta"] = 1.0
        elif action.action_type == "save":
            event["wealth_delta"] = 0.0
            event["stress_delta"] = -0.2

        return event