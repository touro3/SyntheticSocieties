from pydantic import BaseModel

from decision.schemas import ProposedAction
from environment.payoffs import GamePayoffs, DEFAULT_PAYOFFS


class ValidationResult(BaseModel):
    valid: bool
    reason: str | None = None


class InstitutionManager:
    def __init__(self, payoffs: GamePayoffs | None = None) -> None:
        self.payoffs = payoffs or DEFAULT_PAYOFFS

    def validate(self, action: ProposedAction, agent, world_state, agent_lookup) -> ValidationResult:
        if action.amount is not None and action.amount < 0:
            return ValidationResult(valid=False, reason="negative_amount")

        if action.action_type == "cooperate":
            if not action.target_agent_id:
                return ValidationResult(valid=False, reason="missing_target")

            if action.target_agent_id not in agent_lookup:
                return ValidationResult(valid=False, reason="unknown_target")

            if action.target_agent_id == agent.profile.agent_id:
                return ValidationResult(valid=False, reason="self_target_not_allowed")

            if action.amount is None or action.amount <= 0:
                return ValidationResult(valid=False, reason="invalid_transfer_amount")

            if agent.state.wealth < action.amount:
                return ValidationResult(valid=False, reason="insufficient_wealth")

        if action.action_type == "communicate":
            if not action.target_agent_id:
                return ValidationResult(valid=False, reason="missing_target")

            if action.target_agent_id not in agent_lookup:
                return ValidationResult(valid=False, reason="unknown_target")

            if action.target_agent_id == agent.profile.agent_id:
                return ValidationResult(valid=False, reason="self_target_not_allowed")

        return ValidationResult(valid=True)

    def execute(self, action: ProposedAction, agent, world_state, agent_lookup) -> dict:
        event = {
            "agent_id": agent.profile.agent_id,
            "action_type": action.action_type,
            "target_agent_id": action.target_agent_id,
            "wealth_delta": 0.0,
            "stress_delta": 0.0,
            "satisfaction_delta": 0.0,
            "target_wealth_delta": 0.0,
            "interaction_type": None,
            "round_id": world_state.round_id,
        }

        if action.action_type == "work":
            event["wealth_delta"] = float(action.amount or self.payoffs.work_income)
            event["stress_delta"] = self.payoffs.work_stress_increase
            # Earning provides mild satisfaction, reduced when stress is high
            event["satisfaction_delta"] = 0.05 if agent.state.stress < 0.5 else -0.05

        elif action.action_type == "save":
            event["wealth_delta"] = self.payoffs.save_wealth_delta
            event["stress_delta"] = self.payoffs.save_stress_relief
            # Resting and saving provides security-based satisfaction
            event["satisfaction_delta"] = 0.08

        elif action.action_type == "cooperate":
            amount = float(action.amount or 0.0)

            event["wealth_delta"] = -amount
            event["target_wealth_delta"] = amount * self.payoffs.cooperation_multiplier
            event["stress_delta"] = self.payoffs.cooperate_stress_relief
            # Social cooperation provides the strongest satisfaction boost
            event["satisfaction_delta"] = 0.12
            event["interaction_type"] = "cooperation"

        elif action.action_type == "communicate":
            # Communication has no economic effect — purely informational.
            event["interaction_type"] = "communication"
            event["message_summary"] = action.reasoning_summary

        return event
