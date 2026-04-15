import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState


def main():
    profile = AgentProfile(
        agent_id="agent_1",
        age=35,
        income=2000,
        education="college",
        occupation="teacher",
        location="italy",
        political_preference="center",
        risk_tolerance=0.5,
        social_class="middle",
    )

    state = AgentState(wealth=100)
    memory = MemoryBuffer()

    print(profile)
    print(state)
    print(memory)


if __name__ == "__main__":
    main()
