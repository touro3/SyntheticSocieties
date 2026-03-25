from __future__ import annotations

from agents.profile import AgentProfile
from agents.state import AgentState
from agents.memory import HierarchicalMemory
from decision.policy_protocol import PolicyProtocol
from decision.schemas import ProposedAction


class Agent:
    def __init__(
        self,
        profile: AgentProfile,
        state: AgentState,
        memory: HierarchicalMemory,
        policy: PolicyProtocol,
    ) -> None:
        self.profile = profile
        self.state = state
        self.memory = memory
        self.policy = policy

    def perceive(self, world_snapshot: dict, local_network: dict) -> dict:
        return {
            "world": world_snapshot,
            "network": local_network,
            "self_state": {
                "wealth": self.state.wealth,
                "stress": self.state.stress,
                "satisfaction": self.state.satisfaction,
            },
        }

    def decide(self, context: dict, round_id: int) -> ProposedAction:
        return self.policy.propose_action(
            profile=self.profile,
            state=self.state,
            memory=self.memory,
            context=context,
            round_id=round_id,
        )

    def apply_local_update(self, executed_event: dict) -> None:
        self.state.wealth += float(executed_event.get("wealth_delta", 0.0))
        self.state.stress += float(executed_event.get("stress_delta", 0.0))
        self.state.last_action = executed_event.get("action_type", self.state.last_action)
        self.state.clamp()
