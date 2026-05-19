"""RoundProcessor — validates, executes, and records a single agent action.

Extracted from SimulationKernel to eliminate the code duplication between
run_round() and run_round_batched(). The kernel retains only orchestration;
this class owns the validate → execute → update → log sequence.
"""

from __future__ import annotations

from agents.agent import Agent
from agents.memory import HierarchicalMemory, MemoryItem
from decision.schemas import ProposedAction


class RoundProcessor:
    def __init__(self, world, agent_lookup: dict[str, Agent], logger, trajectory_bank=None) -> None:
        self.world = world
        self.agent_lookup = agent_lookup
        self.logger = logger
        # Optional observational trajectory recorder (ruflo ReasoningBank
        # pattern).  Inert unless explicitly supplied — never feeds back
        # into decisions, so the controlled A/B design is unaffected.
        self.trajectory_bank = trajectory_bank

    def process_agent_action(
        self,
        agent: Agent,
        proposed_action: ProposedAction,
        round_id: int,
        perception: dict | None = None,
    ) -> dict:
        """Validate, execute, update state, record memory, and log one action.

        Returns the executed event dict (or rejection dict).
        """
        validation = self.world.validate_action(proposed_action, agent, self.agent_lookup)

        if validation.valid:
            executed_event = self.world.execute_action(proposed_action, agent, self.agent_lookup)
            agent.apply_local_update(executed_event)
            self._apply_target_delta(executed_event)
            self._record_memory(agent, proposed_action, executed_event, round_id)
            self._update_graph_rag(agent.policy, agent, proposed_action, round_id)
            if self.trajectory_bank is not None:
                self.trajectory_bank.record(agent, proposed_action.action_type, executed_event, round_id)
        else:
            executed_event = {
                "agent_id": agent.profile.agent_id,
                "action_type": "rejected",
                "reason": validation.reason,
                "round_id": round_id,
            }

        self._log_event(
            round_id=round_id,
            agent=agent,
            perception=perception or {},
            proposed_action=proposed_action,
            validation=validation,
            executed_event=executed_event,
        )

        return executed_event

    def _apply_target_delta(self, executed_event: dict) -> None:
        target_id = executed_event.get("target_agent_id")
        source_id = executed_event.get("agent_id")
        target_delta = executed_event.get("target_wealth_delta", 0.0)

        # Transactional boundary: this method mutates target wealth, source &
        # target trust, and the network graph in sequence. Snapshot the wealth
        # of every agent we may touch so a mid-sequence failure cannot leave a
        # half-applied economic state — restore and re-raise so the kernel
        # surfaces the error instead of silently corrupting the simulation.
        touched_ids = [aid for aid in (target_id, source_id) if aid and aid in self.agent_lookup]
        wealth_snapshot = {aid: self.agent_lookup[aid].state.wealth for aid in touched_ids}
        try:
            self._apply_target_delta_unsafe(executed_event, target_id, source_id, target_delta)
        except Exception:
            for aid, w in wealth_snapshot.items():
                self.agent_lookup[aid].state.wealth = w
            raise

    def _apply_target_delta_unsafe(
        self, executed_event: dict, target_id, source_id, target_delta: float
    ) -> None:
        if target_id and target_delta and target_id in self.agent_lookup:
            target = self.agent_lookup[target_id]
            target.state.wealth += target_delta
            target.state.clamp()

        # Update trust from cooperation events.
        # Asymmetric: the cooperator invested resources, so they build trust
        # toward the target (hoping for reciprocation). The target received
        # unsolicited help — they gain mild trust toward the source, but it
        # is NOT recorded as "reciprocated" since they didn't cooperate back.
        if executed_event.get("interaction_type") == "cooperation" and target_id:
            source_id = executed_event.get("agent_id")
            if source_id and source_id in self.agent_lookup:
                source = self.agent_lookup[source_id]
                # Check if target also cooperated with source this round
                # (true reciprocation). For now, unilateral cooperation is
                # NOT reciprocated — the trust gradient must be earned.
                source.state.update_trust_from_cooperation(target_id, was_reciprocated=False)
            if target_id in self.agent_lookup:
                target = self.agent_lookup[target_id]
                # Target received help — build mild trust toward the source.
                # was_reciprocated=True here means "the source demonstrated
                # goodwill toward me" (they invested in me).
                target.state.update_trust_from_cooperation(source_id, was_reciprocated=True)

            # Dynamic network evolution: cooperation creates/strengthens edges
            network = getattr(self.world, "network_manager", None)
            if network is not None and source_id:
                network.add_edge(source_id, target_id)

    def _record_memory(
        self,
        agent: Agent,
        proposed_action: ProposedAction,
        executed_event: dict,
        round_id: int,
    ) -> None:
        ttl = HierarchicalMemory.default_ttl(proposed_action.action_type)
        agent.memory.add(
            MemoryItem(
                round_id=round_id,
                partner_id=proposed_action.target_agent_id,
                event_type=proposed_action.action_type,
                content=proposed_action.reasoning_summary,
                outcome=executed_event,
                valid_at=round_id,
                expires_at_round=round_id + ttl if ttl is not None else None,
            )
        )

        # ── Communication injection (information contagion) ──────────────────
        # When an agent communicates, inject the message into the target's
        # memory as a 'received_message' item tagged with the source agent ID.
        if (
            proposed_action.action_type == "communicate"
            and proposed_action.target_agent_id
            and proposed_action.target_agent_id in self.agent_lookup
        ):
            target = self.agent_lookup[proposed_action.target_agent_id]
            source_id = agent.profile.agent_id
            msg_ttl = HierarchicalMemory.default_ttl("observation")  # short-lived
            target.memory.add(
                MemoryItem(
                    round_id=round_id,
                    partner_id=source_id,
                    event_type="received_message",
                    content=f"[from {source_id}] {proposed_action.reasoning_summary}",
                    outcome={},
                    importance=0.4,
                    valid_at=round_id,
                    expires_at_round=round_id + msg_ttl if msg_ttl is not None else None,
                )
            )

    def _update_graph_rag(
        self,
        policy,
        agent: Agent,
        proposed_action: ProposedAction,
        round_id: int,
    ) -> None:
        graph_rag = getattr(policy, "graph_rag", None)
        if graph_rag is None:
            return
        if proposed_action.action_type != "cooperate" or not proposed_action.target_agent_id:
            return
        graph_rag.add_event(
            {
                "round_id": round_id,
                "agent_id": agent.profile.agent_id,
                "action": proposed_action.model_dump(),
            }
        )

    def _log_event(
        self,
        round_id: int,
        agent: Agent,
        perception: dict,
        proposed_action: ProposedAction,
        validation,
        executed_event: dict,
    ) -> None:
        self.logger.log_event(
            {
                "round_id": round_id,
                "agent_id": agent.profile.agent_id,
                "perception": perception,
                "action": proposed_action.model_dump(),
                "validation": validation.model_dump(),
                # First-class audit field: non-empty only when the harness
                # parser rewrote the model's raw output. Lets analysis filter
                # model-authored decisions from harness-repaired ones.
                "harness_substitutions": list(getattr(proposed_action, "substitutions", []) or []),
                "result": executed_event,
                "state_after": agent.state.snapshot(),
            }
        )
