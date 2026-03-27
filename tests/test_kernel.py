from unittest.mock import MagicMock, patch

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from bgf_logging.event_logger import EventLogger
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


def test_batched_mode_respects_ablation_level(tmp_path):
    """run_round_batched must pass policy.ablation_level to build_prompt, not silently default to 5."""
    agents = build_test_agents()

    # Wire each agent with an LLMPolicy that has a non-default ablation_level.
    fake_backend = MagicMock()
    fake_backend.generate_batch.return_value = [
        ('{"action_type": "work", "amount": 8, "reasoning_summary": "test", "confidence": 0.9}', 0.1)
        for _ in agents
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
    original_build = __import__(
        "decision.prompt_builder", fromlist=["build_prompt"]
    ).build_prompt

    def spy_build_prompt(*args, **kwargs):
        captured_calls.append(kwargs.get("ablation_level"))
        return original_build(*args, **kwargs)

    with patch("decision.prompt_builder.build_prompt", side_effect=spy_build_prompt):
        kernel.run_round_batched()

    assert len(captured_calls) == len(agents), "build_prompt should be called once per agent"
    assert all(
        level == 2 for level in captured_calls
    ), f"Expected ablation_level=2 for all agents, got: {captured_calls}"
