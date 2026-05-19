from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ProposedAction(BaseModel):
    action_type: Literal["work", "save", "cooperate", "communicate", "steal"] = Field(...)
    target_agent_id: Optional[str] = None
    amount: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    reasoning_summary: str = Field(...)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # Audit trail of decision-altering rewrites the harness parser applied to
    # the raw model output (e.g. invalid target swapped, action downgraded,
    # action inferred from keywords). Empty when the model's output was used
    # as-is. Surfaced in events.jsonl via model_dump() so model intent vs.
    # harness repair is always distinguishable post-hoc.
    substitutions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def cooperate_requires_target(self) -> "ProposedAction":
        if self.action_type in ("cooperate", "communicate", "steal") and not self.target_agent_id:
            raise ValueError(f"{self.action_type} action requires a target_agent_id")
        return self
