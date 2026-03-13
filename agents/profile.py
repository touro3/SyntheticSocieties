from dataclasses import dataclass

@dataclass
class AgentProfile:
    agent_id: str
    age: int
    income: float
    education: str
    occupation: str
    location: str
    political_preference: str
    risk_tolerance: float
    social_class: str