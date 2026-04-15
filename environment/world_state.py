from dataclasses import dataclass, field


@dataclass
class WorldState:
    round_id: int = 0
    public_signal: dict[str, str] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)
    resources: dict[str, float] = field(default_factory=dict)
    shock_active: bool = False
    shock_magnitude: float = 0.0
