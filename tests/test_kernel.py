from unittest.mock import MagicMock, patch

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from bgf_logging.event_logger import EventLogger
from decision.ablated_llm_policy import AblatedLLMPolicy
from decision.llm_policy import LLMPolicy
from decision.mock_policy import MockPolicy
from environment.institutions import InstitutionManager
from environment.world import World
from environment.world_state import WorldState
from simulation.kernel import SimulationKernel


def build_test_agents() -> list[Agent]:
    agents = []
    for i in range(2):
        agents.append(
            Agent(
                profile=AgentProfile(
                    agent_id=f"agent_{i}",
                    age=30,
                    income=1000,
                    education="college",
                    occupation="worker",
                    location="italy",
                    political_preference="center",
                    risk_tolerance=0.5,
                    social_class="middle",
                ),
                state=AgentState(wealth=50.0),
                memory=MemoryBuffer(max_items=5),
                policy=MockPolicy(),
            )
        )
    return agents


def _make_world():
    return World(
        state=WorldState(
            public_signal={"economy": "stable"},
            prices={"food": 1.0},
            resources={"jobs": 100.0},
        ),
        institution_manager=InstitutionManager(),
    )


def test_kernel_runs(tmp_path):
    agents = build_test_agents()
    world = World(
        state=WorldState(
            public_signal={"economy": "stable"},
            prices={"food": 1.0},
            resources={"jobs": 100.0},
        ),
        institution_manager=InstitutionManager(),
    )
    logger = EventLogger(tmp_path / "events.jsonl", overwrite=True)

    kernel = SimulationKernel(agents=agents, world=world, logger=logger)
    kernel.run(num_rounds=2)

    assert world.state.round_id == 2
    assert agents[0].state.wealth > 50.0
    assert (tmp_path / "events.jsonl").exists()


def test_kernel_writes_validation_manifest(tmp_path):
    """Kernel must emit validation.json so downstream tooling can detect a
    Condition B run that silently degraded to Condition A (audit A1.5)."""
    import json

    agents = build_test_agents()
    world = _make_world()
    logger = EventLogger(tmp_path / "events.jsonl", overwrite=True)

    kernel = SimulationKernel(agents=agents, world=world, logger=logger, heartbeat_path=tmp_path / "heartbeat.json")
    kernel.run(num_rounds=1)

    manifest_path = tmp_path / "validation.json"
    assert manifest_path.exists(), "validation.json not written"
    payload = json.loads(manifest_path.read_text())
    # MockPolicy has neither sql_rag nor graph_rag — both should be absent.
    assert payload["sql_rag_status"] == "absent"
    assert payload["sql_rag_active"] is False
    assert payload["graph_rag_active"] is False
    assert payload["n_agents"] == 2
    assert payload["policy_type"] == "MockPolicy"


def test_kernel_validation_manifest_reports_rag_active(tmp_path):
    """When the policy has a working SQLRAG handle, the manifest must report
    sql_rag_active=True."""
    import json

    from decision.sql_rag import SQLRAG

    agents = build_test_agents()
    # Attach a working SQLRAG (real ess_clean.parquet) to each agent's policy.
    rag = SQLRAG(data_path="data/ess_clean.parquet")
    for agent in agents:
        agent.policy.sql_rag = rag  # MockPolicy is a permissive container

    world = _make_world()
    logger = EventLogger(tmp_path / "events.jsonl", overwrite=True)
    kernel = SimulationKernel(agents=agents, world=world, logger=logger, heartbeat_path=tmp_path / "heartbeat.json")
    kernel.run(num_rounds=1)

    payload = json.loads((tmp_path / "validation.json").read_text())
    assert payload["sql_rag_status"] == "ok"
    assert payload["sql_rag_active"] is True


def test_batched_mode_respects_ablation_level(tmp_path):
    """run_round_batched must pass policy.ablation_level to build_prompt, not silently default to 5."""
    agents = build_test_agents()

    # Wire each agent with an LLMPolicy that has a non-default ablation_level.
    fake_backend = MagicMock()
    fake_backend.generate_batch.return_value = [
        ('{"action_type": "work", "amount": 8, "reasoning_summary": "test", "confidence": 0.9}', 0.1) for _ in agents
    ]

    policy = LLMPolicy(
        backend=fake_backend,
        ablation_level=2,  # non-default; default would be 5
    )
    for agent in agents:
        agent.policy = policy

    world = _make_world()
    logger = EventLogger(tmp_path / "events.jsonl", overwrite=True)
    kernel = SimulationKernel(agents=agents, world=world, logger=logger)

    captured_calls = []
    original_build = __import__("decision.prompt_builder", fromlist=["build_prompt"]).build_prompt

    def spy_build_prompt(*args, **kwargs):
        captured_calls.append(kwargs.get("ablation_level"))
        return original_build(*args, **kwargs)

    with patch("decision.prompt_builder.build_prompt", side_effect=spy_build_prompt):
        kernel.run_round_batched()

    assert len(captured_calls) == len(agents), "build_prompt should be called once per agent"
    assert all(level == 2 for level in captured_calls), (
        f"Expected ablation_level=2 for all agents, got: {captured_calls}"
    )


def test_ablated_policy_runs_via_sequential_path(tmp_path):
    """AblatedLLMPolicy is not an LLMPolicy subclass so it always uses run_round().

    kernel._can_use_batched_mode gates on isinstance(policy, LLMPolicy).
    This test verifies that AblatedLLMPolicy completes a full round correctly
    via the sequential path, and that ablation_level=2 is set for no_network.
    """
    agents = build_test_agents()

    fake_backend = MagicMock()
    fake_backend.generate.return_value = (
        '{"action_type": "work", "amount": 8, "reasoning_summary": "t", "confidence": 0.9}',
        0.1,
    )

    policy = AblatedLLMPolicy(backend=fake_backend, ablation="no_network")
    assert policy.ablation_level == 2, "no_network must set ablation_level=2"

    for agent in agents:
        agent.policy = policy

    world = _make_world()
    logger = EventLogger(tmp_path / "events.jsonl", overwrite=True)
    kernel = SimulationKernel(agents=agents, world=world, logger=logger)

    # run_round_batched must NOT use the batched code path for AblatedLLMPolicy —
    # it should fall back to run_round() since AblatedLLMPolicy is not LLMPolicy.
    assert not kernel._can_use_batched_mode(), (
        "AblatedLLMPolicy must not trigger batched mode — it uses sequential run_round()"
    )
    kernel.run_round()
    assert world.state.round_id == 1
