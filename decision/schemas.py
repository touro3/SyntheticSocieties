from typing import Optional
from pydantic import BaseModel, Field


class ProposedAction(BaseModel):
    action_type: str = Field(...)
    target_agent_id: Optional[str] = None
    amount: Optional[float] = None
    reasoning_summary: str = Field(...)
    confidence: Optional[float] = None