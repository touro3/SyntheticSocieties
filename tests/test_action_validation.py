import pytest
from pydantic import ValidationError

from decision.schemas import ProposedAction


def test_negative_amount_is_rejected_at_schema_level():
    """Negative amounts are now caught by Pydantic validation at construction time."""
    with pytest.raises(ValidationError):
        ProposedAction(
            action_type="work",
            amount=-10.0,
            reasoning_summary="invalid negative amount",
            confidence=0.1,
        )
