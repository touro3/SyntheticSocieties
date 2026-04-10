from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ProposedAction(BaseModel):
    action_type: Literal["work", "save", "cooperate", "communicate"] = Field(...)
    target_agent_id: Optional[str] = None
    amount: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    reasoning_summary: str = Field(...)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def cooperate_requires_target(self) -> "ProposedAction":
        if self.action_type in ("cooperate", "communicate") and not self.target_agent_id:
            raise ValueError(f"{self.action_type} action requires a target_agent_id")
        return self
