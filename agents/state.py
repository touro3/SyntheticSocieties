from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AgentState:
    wealth: float
    trust: Dict[str, float] = field(default_factory=dict)
    stress: float = 0.0
    satisfaction: float = 0.0
    last_action: Optional[str] = None