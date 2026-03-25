from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AgentState:
    wealth: float
    trust: Dict[str, float] = field(default_factory=dict)
    stress: float = 0.0
    satisfaction: float = 0.0
    last_action: Optional[str] = None

    def __post_init__(self) -> None:
        if self.wealth < 0.0:
            self.wealth = 0.0

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
        }
