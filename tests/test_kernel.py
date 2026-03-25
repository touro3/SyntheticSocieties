from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from bgf_logging.event_logger import EventLogger
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
