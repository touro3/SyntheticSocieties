from dataclasses import dataclass, field
from typing import Dict


@dataclass
class WorldState:
    round_id: int = 0
    public_signal: Dict[str, str] = field(default_factory=dict)
    prices: Dict[str, float] = field(default_factory=dict)
    resources: Dict[str, float] = field(default_factory=dict)