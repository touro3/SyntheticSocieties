import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.mock_policy import MockPolicy
from environment.institutions import InstitutionManager
from environment.world import World
from environment.world_state import WorldState
from bgf_logging.event_logger import EventLogger
from simulation.kernel import SimulationKernel


def build_agents(n: int) -> list[Agent]:
    agents = []

    for i in range(n):
        profile = AgentProfile(
            agent_id=f"agent_{i}",
            age=25 + i,
            income=1000 + i * 100,
            education="college",
            occupation="worker",
            location="italy",
            political_preference="center",
            risk_tolerance=0.5,
            social_class="middle",
        )

        state = AgentState(wealth=50.0 + i * 10)
        memory = MemoryBuffer(max_items=10)
        policy = MockPolicy()

        agents.append(
            Agent(
                profile=profile,
                state=state,
                memory=memory,
                policy=policy,
            )
        )

    return agents


def main() -> None:
    agents = build_agents(5)

    world = World(
        state=WorldState(
            public_signal={"economy": "stable"},
            prices={"food": 1.0},
            resources={"jobs": 100.0},
        ),
        institution_manager=InstitutionManager(),
    )

    logger = EventLogger("experiments/exp_0001/events.jsonl", overwrite=True)

    kernel = SimulationKernel(
        agents=agents,
        world=world,
        logger=logger,
    )

    kernel.run(num_rounds=3)

    for agent in agents:
        print(agent.profile.agent_id, agent.state)


if __name__ == "__main__":
    main()