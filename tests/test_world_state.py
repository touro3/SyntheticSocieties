"""Tests for WorldState enrichment."""

from environment.world_state import WorldState


class TestWorldStateMacroShock:
    def test_default_no_shock(self):
        ws = WorldState()
        assert ws.shock_active is False
        assert ws.shock_magnitude == 0.0

    def test_shock_can_be_activated(self):
        ws = WorldState(shock_active=True, shock_magnitude=0.5)
        assert ws.shock_active is True
        assert ws.shock_magnitude == 0.5


class TestWorldStateRoundId:
    def test_default_round_id_is_zero(self):
        ws = WorldState()
        assert ws.round_id == 0

    def test_round_id_increments(self):
        ws = WorldState()
        ws.round_id += 1
        assert ws.round_id == 1
