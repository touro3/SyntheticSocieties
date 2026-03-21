from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Optional

from decision.schemas import ProposedAction


class AuditableRandomPolicy:
    def __init__(
        self,
        seed: int = 42,
        audit_path: str | Path | None = None,
        cohort_summary: Optional[dict] = None,
        base_weights: Optional[dict] = None,
        condition_name: str = "auditable_random",
    ):
        self.seed = seed
        self.audit_path = Path(audit_path) if audit_path else None
        self.cohort_summary = cohort_summary or {}
        self.base_weights = base_weights or {
            "work": 0.34,
            "save": 0.33,
            "cooperate": 0.33,
        }
        self.condition_name = condition_name

        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def propose_action(self, profile, state, memory, context: dict, round_id: int) -> ProposedAction:
        neighbors = context.get("network", {}).get("neighbors", [])
        rng = random.Random(self._derive_seed(profile.agent_id, round_id))

        weights, adjustments = self._compute_weights(state=state, neighbors=neighbors)
        draw = rng.random()
        action_type = self._weighted_choice(weights, draw)

        target_agent_id = None
        if action_type == "cooperate" and neighbors:
            target_agent_id = rng.choice(neighbors)

        amount = self._sample_amount(action_type, state, rng)
        reasoning_summary = self._reasoning_summary(weights, adjustments)

        action = ProposedAction(
            action_type=action_type,
            target_agent_id=target_agent_id,
            amount=amount,
            reasoning_summary=reasoning_summary,
            confidence=0.4,
        )

        self._log_audit(
            round_id=round_id,
            agent_id=profile.agent_id,
            state=state,
            neighbors=neighbors,
            weights=weights,
            adjustments=adjustments,
            draw=draw,
            action=action,
        )
        return action

    def _derive_seed(self, agent_id: str, round_id: int) -> int:
        key = f"{self.seed}:{agent_id}:{round_id}:{self.condition_name}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        return int(digest[:16], 16)

    def _compute_weights(self, state, neighbors: list[str]) -> tuple[dict, list[str]]:
        weights = dict(self.base_weights)
        adjustments: list[str] = []

        age_mean = self.cohort_summary.get("age_mean")
        trust_people_mean = self.cohort_summary.get("trust_people_mean")
        social_activity_mean = self.cohort_summary.get("social_activity_mean")

        if age_mean is not None and age_mean >= 65:
            weights["save"] += 0.05
            weights["work"] -= 0.03
            weights["cooperate"] -= 0.02
            adjustments.append("cohort_age_mean>=65 -> save+0.05")

        if trust_people_mean is not None and trust_people_mean <= 0.35:
            weights["cooperate"] -= 0.08
            weights["save"] += 0.04
            weights["work"] += 0.04
            adjustments.append("low_cohort_trust_people -> cooperate-0.08")

        if social_activity_mean is not None and social_activity_mean >= 0.60:
            weights["cooperate"] += 0.06
            weights["save"] -= 0.03
            weights["work"] -= 0.03
            adjustments.append("high_cohort_social_activity -> cooperate+0.06")

        if state.wealth < 70:
            weights["work"] += 0.20
            weights["save"] -= 0.10
            weights["cooperate"] -= 0.10
            adjustments.append("low_wealth -> work+0.20")
        elif state.wealth > 110:
            weights["cooperate"] += 0.08
            weights["save"] += 0.02
            weights["work"] -= 0.10
            adjustments.append("high_wealth -> cooperate+0.08")

        if state.stress > 0.60:
            weights["save"] += 0.22
            weights["work"] -= 0.15
            weights["cooperate"] -= 0.07
            adjustments.append("high_stress -> save+0.22")
        elif state.stress < -0.40:
            weights["work"] += 0.08
            weights["save"] -= 0.04
            weights["cooperate"] -= 0.04
            adjustments.append("very_low_stress -> work+0.08")

        if not neighbors:
            weights["work"] += max(0.0, weights["cooperate"]) * 0.50
            weights["save"] += max(0.0, weights["cooperate"]) * 0.50
            weights["cooperate"] = 0.0
            adjustments.append("no_neighbors -> cooperate=0")

        if state.wealth < 10:
            weights["cooperate"] -= 0.15
            weights["work"] += 0.10
            weights["save"] += 0.05
            adjustments.append("very_low_wealth -> cooperate-0.15")

        weights = {key: max(0.001, value) for key, value in weights.items()}
        total = sum(weights.values())
        weights = {key: value / total for key, value in weights.items()}
        return weights, adjustments

    def _weighted_choice(self, weights: dict, draw: float) -> str:
        cumulative = 0.0
        for action_type in ["work", "save", "cooperate"]:
            cumulative += weights[action_type]
            if draw <= cumulative:
                return action_type
        return "save"

    def _sample_amount(self, action_type: str, state, rng: random.Random) -> float:
        if action_type == "work":
            return float(rng.randint(5, 15))
        if action_type == "save":
            return float(rng.randint(5, 10))
        if action_type == "cooperate":
            upper = min(10, max(1, int(state.wealth)))
            lower = min(5, upper)
            return float(rng.randint(lower, upper))
        return 5.0

    def _reasoning_summary(self, weights: dict, adjustments: list[str]) -> str:
        weight_str = ", ".join(f"{k}={v:.2f}" for k, v in weights.items())
        if adjustments:
            return f"[Auditable random] weighted sample with {weight_str}. Key adjustments: {'; '.join(adjustments[:3])}."
        return f"[Auditable random] weighted sample with {weight_str}."

    def _log_audit(self, round_id: int, agent_id: str, state, neighbors: list[str], weights: dict, adjustments: list[str], draw: float, action: ProposedAction) -> None:
        if self.audit_path is None:
            return

        record = {
            "round_id": round_id,
            "agent_id": agent_id,
            "weights": weights,
            "adjustments": adjustments,
            "draw": draw,
            "neighbors": neighbors,
            "state": {
                "wealth": state.wealth,
                "stress": state.stress,
                "satisfaction": state.satisfaction,
            },
            "action": action.model_dump(),
            "condition_name": self.condition_name,
            "base_seed": self.seed,
        }
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
