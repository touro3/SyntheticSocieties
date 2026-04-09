from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentState:
    wealth: float
    trust: dict[str, float] = field(default_factory=dict)
    stress: float = 0.0
    satisfaction: float = 0.0
    last_action: Optional[str] = None

    def __post_init__(self) -> None:
        if self.wealth < 0.0:
            self.wealth = 0.0

    @property
    def trust_network(self) -> dict[str, float]:
        """Alias for trust dict — used by prompt_builder's TRUST_SURFACED level."""
        return self.trust

    def update_trust_from_cooperation(
        self, partner_id: str, was_reciprocated: bool, decay: float = 0.95
    ) -> None:
        """Update trust toward a partner based on cooperation outcomes.

        Trust increases when cooperation is reciprocated and decreases
        when it is not. A decay factor is applied to existing trust each
        update to model recency weighting.

        Args:
            partner_id: ID of the cooperation partner.
            was_reciprocated: Whether the partner reciprocated this round.
            decay: Recency decay applied to existing trust (0-1).
        """
        current = self.trust.get(partner_id, 0.0)
        delta = 0.1 if was_reciprocated else -0.05
        self.trust[partner_id] = max(0.0, min(1.0, current * decay + delta))

    def clamp(self) -> None:
        """Enforce domain bounds on mutable state fields."""
        if self.wealth < 0.0:
            self.wealth = 0.0
        self.stress = max(-1.0, min(1.0, self.stress))
        self.satisfaction = max(0.0, min(1.0, self.satisfaction))

    def snapshot(self) -> dict:
        """Return a frozen dict copy of current state for logging."""
        return {
            "wealth": self.wealth,
            "stress": self.stress,
            "satisfaction": self.satisfaction,
            "last_action": self.last_action,
            "trust": dict(self.trust) if self.trust else {},
        }

