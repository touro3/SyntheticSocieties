from api.app import create_app
from environment.institutions import InstitutionManager
from environment.world import World
from environment.world_state import WorldState
from simulation.ipc import SimulationIPCServer


def test_world_drains_pending_injections_into_state():
    state = WorldState(
        round_id=0,
        public_signal={"economy": "stable"},
        prices={"food": 1.0},
        resources={"jobs": 100.0},
        pending_injections=[
            {"event_type": "signal_update", "payload": {"signal": {"economy": "strained"}}},
            {"event_type": "wealth_shock", "payload": {"magnitude": 0.4, "content": "A large shock hit."}},
        ],
    )
    world = World(state=state, institution_manager=InstitutionManager())

    applied = world.apply_exogenous_updates()

    assert len(applied) == 2
    assert state.pending_injections == []
    assert state.round_id == 1
    assert state.public_signal["economy"] == "strained"
    assert state.shock_active is True
    assert state.shock_magnitude == 0.4


def test_ipc_inject_event_queues_on_world_state(tmp_path):
    state = WorldState(round_id=7)
    server = SimulationIPCServer(
        agents={},
        base_dir=tmp_path,
        current_round_fn=lambda: state.round_id,
        world_state=state,
    )

    result = server._dispatch("inject_event", {"event_type": "narrative", "payload": {"content": "A rumor spreads."}})

    assert result == {"status": "ok", "round": 7}
    assert state.pending_injections == [{"event_type": "narrative", "payload": {"content": "A rumor spreads."}}]


def test_inject_endpoint_returns_404_for_unknown_experiment(tmp_path):
    app = create_app(experiments_root=str(tmp_path), configs_root=str(tmp_path))
    client = app.test_client()

    response = client.post(
        "/inject/missing_exp",
        json={"event_type": "narrative", "payload": {"content": "A major shock occurred"}},
    )

    assert response.status_code == 404
