"""Event-driven simulation kernel.

Orchestrates round execution by gathering agent proposals (sequentially or
batched) and delegating processing to RoundProcessor. The kernel owns only
orchestration; RoundProcessor owns validate → execute → update → log.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import time
import warnings
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from agents.memory import HierarchicalMemory, MemoryItem, MemoryLevel
from decision.output_parser import get_parse_stats, reset_parse_stats
from metrics.inequality import gini_coefficient as _gini_canonical
from simulation.round_processor import RoundProcessor

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON to *path* atomically via a sibling .tmp file + os.rename().

    os.rename() is atomic on POSIX (Linux/macOS). A concurrent reader can
    never observe a partial write — it either sees the old file or the new
    one, never a half-written buffer.
    """
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)  # os.replace is POSIX-atomic


def _capture_rng_states() -> dict:
    """Snapshot every RNG the simulation may consume, in JSON-safe form.

    torch is optional — guarded so test/CI environments without it still
    checkpoint cleanly.
    """
    states: dict = {}

    py = random.getstate()
    # py = (version:int, tuple[int]*625, gauss_next:None|float)
    states["python"] = [py[0], list(py[1]), py[2]]

    try:
        import numpy as np

        np_state = np.random.get_state()
        states["numpy"] = [
            np_state[0],
            np_state[1].tolist(),
            int(np_state[2]),
            int(np_state[3]),
            float(np_state[4]),
        ]
    except Exception as exc:
        logger.debug("RNG state capture/restore skipped: %s", exc)

    try:
        import torch

        states["torch"] = torch.random.get_rng_state().tolist()
        if torch.cuda.is_available():
            states["torch_cuda"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
    except Exception as exc:
        logger.debug("RNG state capture/restore skipped: %s", exc)

    return states


def _restore_rng_states(states: dict) -> None:
    """Inverse of _capture_rng_states(). Missing keys are skipped silently."""
    if not states:
        return

    py = states.get("python")
    if py is not None:
        random.setstate((py[0], tuple(py[1]), py[2]))

    np_state = states.get("numpy")
    if np_state is not None:
        try:
            import numpy as np

            np.random.set_state(
                (
                    np_state[0],
                    np.array(np_state[1], dtype=np.uint32),
                    int(np_state[2]),
                    int(np_state[3]),
                    float(np_state[4]),
                )
            )
        except Exception:
            pass

    torch_state = states.get("torch")
    if torch_state is not None:
        try:
            import torch

            torch.random.set_rng_state(torch.tensor(torch_state, dtype=torch.uint8))
            cuda_states = states.get("torch_cuda")
            if cuda_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([torch.tensor(s, dtype=torch.uint8) for s in cuda_states])
        except Exception:
            pass


# Flush in-memory round_metrics to disk every N rounds to keep RAM bounded.
# At ~1 KB per dict and 100 agents per round, 50 rounds = ~50 KB in memory max.
_METRICS_FLUSH_INTERVAL = 50


class SimulationKernel:
    def __init__(
        self,
        agents: list,
        world,
        logger,
        heartbeat_path: Optional[Path] = None,
        collective_memory=None,
        social_env=None,
        trajectory_bank=None,
    ) -> None:
        self.agents = agents
        self.world = world
        self.logger = logger
        self.heartbeat_path = Path(heartbeat_path) if heartbeat_path else None
        self.collective_memory = collective_memory or getattr(world, "collective_memory", None)
        self.social_env = social_env
        self.agent_lookup = {agent.profile.agent_id: agent for agent in agents}
        if self.collective_memory is not None:
            for agent in agents:
                if hasattr(agent.policy, "collective_memory") and getattr(agent.policy, "collective_memory") is None:
                    agent.policy.collective_memory = self.collective_memory
        self._processor = RoundProcessor(
            world=world,
            agent_lookup=self.agent_lookup,
            logger=logger,
            trajectory_bank=trajectory_bank,
        )
        self.round_metrics: list[dict] = []

        # Derive metrics flush path from heartbeat directory so the kernel
        # needs no extra parameters from callers.
        if self.heartbeat_path is not None:
            self._metrics_flush_path: Optional[Path] = self.heartbeat_path.parent / "round_metrics.jsonl"
        else:
            self._metrics_flush_path = None

    def _can_use_batched_mode(self) -> bool:
        try:
            from decision.llm_policy import LLMPolicy
        except ImportError:
            return False

        if not self.agents or not hasattr(self.agents[0], "policy"):
            return False

        policy = self.agents[0].policy
        backend = getattr(policy, "backend", None)
        return isinstance(policy, LLMPolicy) and hasattr(backend, "generate_batch")

    def run_round(self) -> None:
        applied_injections = self.world.apply_exogenous_updates() or []
        round_id = self.world.state.round_id
        self._advance_collective_memory(round_id, applied_injections)

        if self.social_env is not None:
            self.social_env.apply_round_decay(round_id)

        # Advance each agent's memory clock — expires stale beliefs.
        for agent in self.agents:
            agent.memory.advance_round(round_id)

        for agent in self.agents:
            world_context = self.world.get_agent_context(agent.profile.agent_id)
            perception = agent.perceive(
                world_snapshot=world_context,
                local_network={"neighbors": world_context.get("neighbors", [])},
            )

            proposed_action = agent.decide(context=perception, round_id=round_id)
            self._processor.process_agent_action(
                agent,
                proposed_action,
                round_id,
                perception=perception,
            )

        if self.social_env is not None:
            self.social_env.step(self.agents, round_id)

        # Memory update on the sequential path — no cached neighbors available
        # so get_agent_context() is called per-agent as before.
        self._narrate_and_update_memory(self.world.state.round_id)

    def _prepare_agent_batch(
        self,
        policy,
        round_id: int,
    ) -> tuple[list[dict], list[list[dict]]]:
        """Build per-agent context dicts and prompt message lists for batched inference.

        Extracted from run_round_batched() to keep that method readable.  The
        heavy-lifting here is: perception, optional RAG context retrieval, prompt
        construction, and optional perturbation.

        Returns:
            agent_data: List of dicts with keys agent, perception, neighbors,
                        social_context, pop_context.
            messages_list: Parallel list of chat message lists ready for
                           backend.generate_batch().
        """
        from decision.prompt_builder import build_prompt

        agent_data: list[dict] = []
        messages_list: list[list[dict]] = []

        for agent in self.agents:
            world_context = self.world.get_agent_context(agent.profile.agent_id)
            perception = agent.perceive(
                world_snapshot=world_context,
                local_network={"neighbors": world_context.get("neighbors", [])},
            )

            social_context = policy.graph_rag_context(agent.profile.agent_id)

            pop_context = policy.sql_rag_context(
                age=agent.profile.age,
                gender=agent.profile.gender,
                country=agent.profile.country,
            )

            messages = build_prompt(
                profile=agent.profile,
                state=agent.state,
                memory=agent.memory,
                context=perception,
                round_id=round_id,
                memory_window=policy.memory_window,
                social_context=social_context,
                population_context=pop_context,
                ablation_level=getattr(policy, "ablation_level", 5),
                max_tokens=getattr(policy, "prompt_budget", None),
                collective_context=policy.collective_memory.get_context()
                if getattr(policy, "collective_memory", None) is not None
                else None,
            )

            if policy.perturbation_mode:
                from decision.prompt_perturbation import apply_perturbation

                seed = hash((round_id, agent.profile.agent_id)) % (2**31)
                messages = apply_perturbation(messages, mode=policy.perturbation_mode, seed=seed)

            agent_data.append(
                {
                    "agent": agent,
                    "perception": perception,
                    "neighbors": world_context.get("neighbors", []),
                    "social_context": social_context,
                    "pop_context": pop_context,
                    # Store the already-built messages so the logging step can
                    # reuse them directly — avoids rebuilding the full prompt
                    # (memory retrieval + token budget trimming) a second time.
                    "messages": messages,
                }
            )
            messages_list.append(messages)

        return agent_data, messages_list

    def run_round_batched(self) -> None:
        from decision.llm_policy import LLMPolicy
        from decision.output_parser import parse_llm_output

        sample_policy = self.agents[0].policy if self.agents and hasattr(self.agents[0], "policy") else None
        backend = getattr(sample_policy, "backend", None)

        if not isinstance(sample_policy, LLMPolicy) or not hasattr(backend, "generate_batch"):
            return self.run_round()

        policy = sample_policy

        applied_injections = self.world.apply_exogenous_updates() or []
        round_id = self.world.state.round_id
        self._advance_collective_memory(round_id, applied_injections)

        if self.social_env is not None:
            self.social_env.apply_round_decay(round_id)

        # Advance each agent's memory clock — expires stale beliefs.
        for agent in self.agents:
            agent.memory.advance_round(round_id)

        agent_data, messages_list = self._prepare_agent_batch(policy, round_id)

        batch_results = policy.backend.generate_batch(
            messages_list=messages_list,
            temperature=policy.temperature,
            max_batch_size=getattr(policy.backend, "_max_batch_size", 16),
        )

        for i, (raw_text, latency) in enumerate(batch_results):
            agent = agent_data[i]["agent"]
            perception = agent_data[i]["perception"]
            neighbors = agent_data[i]["neighbors"]

            action, parse_meta = parse_llm_output(raw_text, neighbors)

            if action is None:
                action = policy._fallback_action(agent.state, neighbors, profile=agent.profile)
                parse_meta["fallback"] = True

            if policy.prompt_logger:
                # Reuse the messages already built in _prepare_agent_batch —
                # avoids a full second prompt build (memory scoring + token
                # budget trimming) purely for logging.
                cached_msgs = agent_data[i].get("messages", [])
                prompt_text = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in cached_msgs)
                policy.prompt_logger.log(
                    round_id=round_id,
                    agent_id=agent.profile.agent_id,
                    prompt=prompt_text,
                    raw_output=raw_text,
                    parsed_action=action.model_dump() if action else None,
                    latency_ms=latency * 1000,
                    parse_metadata=parse_meta,
                    rag_context={
                        "sql_rag_present": bool(agent_data[i].get("pop_context")),
                        "graph_rag_present": bool(agent_data[i].get("social_context")),
                    },
                )

            self._processor.process_agent_action(
                agent,
                action,
                round_id,
                perception=perception,
            )

        if self.social_env is not None:
            self.social_env.step(self.agents, round_id)

        # Build neighbor cache from already-assembled agent_data before freeing
        # it — avoids a second get_agent_context() traversal in the memory step.
        cached_neighbors = {d["agent"].profile.agent_id: d["perception"].get("neighbors", []) for d in agent_data}

        # Release batch-local data structures immediately.  agent_data holds
        # references to every agent object plus their full perception dicts;
        # keeping them alive until Python's cyclic GC decides to collect them
        # can double peak RAM usage when population size is large (500+ agents).
        del agent_data, messages_list, batch_results
        # gc.collect() is expensive (~10–50 ms on 500-agent runs). Calling it
        # every round costs more CPU than it saves RAM in the common case;
        # run it every 10 rounds instead. Python's generational GC handles
        # the rest.
        if self.world.state.round_id % 10 == 0:
            gc.collect()

        # Memory update uses cached neighbors — no second network traversal.
        self._narrate_and_update_memory(self.world.state.round_id, cached_neighbors=cached_neighbors)

    # ── Checkpoint ───────────────────────────────────────────────────────────

    @staticmethod
    def _serialize_memory(memory: HierarchicalMemory) -> dict:
        """Serialize the full HierarchicalMemory so resume is not amnesiac."""
        return {
            "recent": [asdict(i) for i in memory.recent],
            "archive": [asdict(i) for i in memory.archive],
            "reflections": list(memory.reflections),
            "pending_buffer": [asdict(i) for i in memory._pending_buffer],
            "current_round": int(memory._current_round),
            "level": int(memory.level),
        }

    @staticmethod
    def _restore_memory(memory: HierarchicalMemory, snap: dict) -> None:
        def _items(rows):
            return [MemoryItem(**r) for r in rows]

        memory.recent = _items(snap.get("recent", []))
        memory.archive = _items(snap.get("archive", []))
        memory.reflections = list(snap.get("reflections", []))
        memory._pending_buffer = _items(snap.get("pending_buffer", []))
        memory._current_round = int(snap.get("current_round", 0))
        if "level" in snap:
            memory.level = MemoryLevel(int(snap["level"]))
        # Force reflection recomputation from the restored archive/recent.
        memory._cache_dirty = True
        memory._reflection_cache = None

    def _social_graph(self):
        """Return the live nx.Graph if a network manager is wired, else None."""
        nm = getattr(self.world, "network_manager", None)
        return getattr(nm, "graph", None) if nm is not None else None

    def save_checkpoint(self, path: Path) -> None:
        """Persist full simulation state to a JSON checkpoint.

        Captures everything needed for a scientifically-valid resume: agent
        state AND memory (C1), the social graph (C2), every RNG (C3),
        collective memory (C4), and the complete world state (C5).
        """
        data = {
            "round_id": self.world.state.round_id,
            "agents": {agent.profile.agent_id: agent.state.snapshot() for agent in self.agents},
            "agent_memory": {agent.profile.agent_id: self._serialize_memory(agent.memory) for agent in self.agents},
            "rng_states": _capture_rng_states(),
            "world_state": asdict(self.world.state),
        }

        # C2 — social graph (edge list with weights + nodes).
        graph = self._social_graph()
        if graph is not None:
            data["social_graph"] = {
                "nodes": list(graph.nodes()),
                "edges": [[u, v, float(d.get("weight", 1.0))] for u, v, d in graph.edges(data=True)],
            }

        # C4 — collective memory facts.
        if self.collective_memory is not None:
            data["collective_memory"] = [asdict(f) for f in self.collective_memory.snapshot()]

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, data)

    def load_checkpoint(self, path: Path) -> int:
        """Restore full simulation state from a checkpoint. Returns round_id."""
        data = json.loads(Path(path).read_text())
        saved = data.get("agents", {})
        saved_mem = data.get("agent_memory", {})
        for agent in self.agents:
            snap = saved.get(agent.profile.agent_id)
            if snap is not None:
                agent.state.wealth = float(snap.get("wealth", agent.state.wealth))
                agent.state.stress = float(snap.get("stress", agent.state.stress))
                agent.state.satisfaction = float(snap.get("satisfaction", agent.state.satisfaction))
                agent.state.last_action = snap.get("last_action", agent.state.last_action)
                agent.state.trust = {k: float(v) for k, v in snap.get("trust", {}).items()}
            mem_snap = saved_mem.get(agent.profile.agent_id)
            if mem_snap is not None:
                self._restore_memory(agent.memory, mem_snap)

        # C3 — RNG states.
        _restore_rng_states(data.get("rng_states", {}))

        # C5 — full world state (prices, signals, resources, injection queue).
        ws = data.get("world_state")
        if isinstance(ws, dict):
            for k, v in ws.items():
                if hasattr(self.world.state, k):
                    setattr(self.world.state, k, v)

        # C2 — social graph.
        graph = self._social_graph()
        sg = data.get("social_graph")
        if graph is not None and isinstance(sg, dict):
            graph.remove_edges_from(list(graph.edges()))
            graph.add_nodes_from(sg.get("nodes", []))
            for edge in sg.get("edges", []):
                u, v, w = edge[0], edge[1], float(edge[2])
                graph.add_edge(u, v, weight=w)

        # C4 — collective memory.
        cm = data.get("collective_memory")
        if self.collective_memory is not None and isinstance(cm, list):
            from agents.collective_memory import WorldFact

            with self.collective_memory._lock:
                self.collective_memory._facts = [WorldFact(**f) for f in cm]

        round_id = int(data.get("round_id", 0))
        self.world.state.round_id = round_id
        return round_id

    def run(self, num_rounds: int, start_round: int = 0, stop_flag: Optional[object] = None) -> int:
        """Run num_rounds simulation rounds, optionally skipping already-done ones.

        Args:
            num_rounds:  Total rounds in the experiment (from config).
            start_round: First round to execute (0 = fresh start; N = resume after N).
            stop_flag:   Optional ``GracefulShutdown`` instance.  When its
                         ``requested`` attribute becomes True (SIGTERM/SIGINT),
                         the loop exits cleanly after the current round finishes
                         and saves a checkpoint before returning.

        Returns:
            The number of completed rounds (< num_rounds if interrupted).
        """
        remaining = num_rounds - start_round
        if remaining <= 0:
            logger.info(
                "All %d rounds already complete — nothing to do.",
                num_rounds,
                extra={"num_rounds": num_rounds, "start_round": start_round, "n_agents": len(self.agents)},
            )
            return num_rounds - start_round
        if start_round > 0:
            logger.info(
                "Resuming from round %d (%d remaining).",
                start_round,
                remaining,
                extra={"start_round": start_round, "remaining": remaining, "n_agents": len(self.agents)},
            )
        # Evaluate once — the policy type and backend never change mid-run.
        use_batched = self._can_use_batched_mode()

        # Probe RAG handles once and dump a validation manifest so cloud
        # deploys missing data/ess_clean.parquet cannot silently collapse
        # Condition B → Condition A without leaving a paper trail. See
        # docs/AUDIT_DATA_METRICS_LOGGING.md A1.5.
        if start_round == 0:
            self._write_validation_manifest()

        completed = 0
        for _ in range(remaining):
            if stop_flag is not None and getattr(stop_flag, "requested", False):
                logger.info(
                    "GracefulShutdown requested after round %d — stopping loop.",
                    self.world.state.round_id,
                )
                break
            if use_batched:
                self.run_round_batched()
            else:
                self.run_round()
            self._log_round_metrics()
            self._write_heartbeat()
            completed += 1
            if self.heartbeat_path is not None:
                checkpoint_path = self.heartbeat_path.parent / "checkpoint.json"
                self.save_checkpoint(checkpoint_path)
        # Flush any remaining round_metrics to disk without clearing.
        # Callers and tests can still access the last batch via round_metrics.
        if self.round_metrics:
            self._flush_round_metrics()

        return completed

    # ── Validation manifest ──────────────────────────────────────────────────

    def _write_validation_manifest(self) -> None:
        """Emit validation.json alongside heartbeat so downstream tooling can
        detect a Condition B run that silently degraded to Condition A
        because data/ess_clean.parquet was missing or the parquet lacked
        the peer-group columns.
        """
        if self.heartbeat_path is None:
            return
        policy = getattr(self.agents[0], "policy", None) if self.agents else None
        sql_rag = getattr(policy, "sql_rag", None)
        graph_rag = getattr(policy, "graph_rag", None)

        sql_status = "absent"
        if sql_rag is not None:
            try:
                sql_rag._connect()  # idempotent; sets last_status
                sql_status = getattr(sql_rag, "last_status", "ok")
            except Exception as exc:  # pragma: no cover - defensive
                sql_status = f"probe_error:{type(exc).__name__}"

        graph_status = "absent" if graph_rag is None else "present"

        manifest = {
            "sql_rag_status": sql_status,
            "sql_rag_active": sql_status == "ok",
            "graph_rag_status": graph_status,
            "graph_rag_active": graph_rag is not None,
            "ablation_level": getattr(policy, "ablation_level", None),
            "policy_type": type(policy).__name__ if policy is not None else None,
            "n_agents": len(self.agents),
        }
        path = self.heartbeat_path.parent / "validation.json"
        try:
            _atomic_write_json(path, manifest)
        except Exception:  # pragma: no cover
            logger.exception("Failed to write validation.json at %s", path)

    # ── Heartbeat ────────────────────────────────────────────────────────────

    def _write_heartbeat(self) -> None:
        """Write a heartbeat file so external watchdogs can detect stalls."""
        if self.heartbeat_path is None:
            return
        try:
            payload: dict = {
                "round_id": self.world.state.round_id,
                "ts": time.time(),
                "n_agents": len(self.agents),
            }
            # Enrich with last round's metrics (available since _log_round_metrics
            # always runs before _write_heartbeat in the main loop).
            if self.round_metrics:
                last = self.round_metrics[-1]
                payload["mean_wealth"] = last.get("mean_wealth")
                payload["gini"] = last.get("gini")
                payload["action_distribution"] = last.get("action_distribution", {})
                payload["last_action"] = max(
                    last.get("action_distribution", {}),
                    key=last.get("action_distribution", {}).get,
                    default=None,
                )
            _atomic_write_json(self.heartbeat_path, payload)
        except Exception as exc:  # never let heartbeat I/O crash the sim
            logger.warning("Heartbeat write failed: %s", exc)

    # ── Real-time memory update loop ─────────────────────────────────────────

    def _narrate_and_update_memory(
        self,
        round_id: int,
        cached_neighbors: dict[str, list[str]] | None = None,
    ) -> None:
        """Convert the round's collective actions into NL observations per agent.

        Inspired by MiroFish's ZepGraphMemoryUpdater: after each round, each
        agent receives a natural-language summary of what its neighbors did,
        written as an 'observation' memory item with a short TTL so it fades
        naturally.  This closes the perception–action loop: agents not only
        act but also *observe* the social environment evolving around them.

        Args:
            cached_neighbors: Optional pre-built mapping of agent_id → neighbor
                list from the batch round.  When provided, skips the redundant
                get_agent_context() call (saves N network traversals per round).
        """
        # Collect last-action info for all agents
        action_registry: dict[str, str] = {}
        for agent in self.agents:
            if agent.state.last_action:
                action_registry[agent.profile.agent_id] = agent.state.last_action

        if not action_registry:
            return

        ttl = HierarchicalMemory.default_ttl("observation")

        for agent in self.agents:
            if cached_neighbors is not None:
                neighbors = cached_neighbors.get(agent.profile.agent_id, [])
            else:
                ctx = self.world.get_agent_context(agent.profile.agent_id)
                neighbors = ctx.get("neighbors", [])
            if not neighbors:
                continue

            # Build neighbor activity summary (cap at 5 to prevent prompt bloat,
            # mirrors MiroFish BATCH_SIZE = 5 pattern)
            # Sort neighbor IDs for deterministic observation strings
            # (adjacency-list order may vary after graph mutations).
            neighbor_actions: list[str] = []
            for nid in sorted(neighbors)[:5]:
                action = action_registry.get(nid)
                if action:
                    neighbor_actions.append(f"{nid} chose to {action}")

            if not neighbor_actions:
                continue

            narration = f"Round {round_id} observations: {'; '.join(neighbor_actions)}."

            if self.social_env is not None:
                sctx = self.social_env.get_agent_context(agent.profile.agent_id)
                feed = sctx.get("feed", [])[:2]
                if feed:
                    feed_note = "; ".join(p["content"][:60] for p in feed if p.get("content"))
                    if feed_note:
                        narration += f" Social feed: [{feed_note}]"

            obs_item = MemoryItem(
                round_id=round_id,
                partner_id=None,
                event_type="observation",
                content=narration,
                outcome={},
                importance=0.3,  # Lower than actions, higher than noise
                valid_at=round_id,
                expires_at_round=(round_id + ttl if ttl is not None else None),
            )
            agent.memory.add(obs_item)

    def _advance_collective_memory(self, round_id: int, applied_injections: list[dict]) -> None:
        if self.collective_memory is None:
            return
        self.collective_memory.advance_round(round_id)
        for injection in applied_injections:
            if not isinstance(injection, dict):
                continue
            event_type = str(injection.get("event_type", injection.get("type", "narrative")))
            payload = injection.get("payload", {})
            if not isinstance(payload, dict):
                payload = {"content": payload}

            content = payload.get("content") or payload.get("message")
            if not content and event_type == "signal_update":
                signal = payload.get("signal", payload)
                if isinstance(signal, dict):
                    content = ", ".join(f"{key}={value}" for key, value in signal.items())
            if not content and event_type == "wealth_shock":
                content = f"Wealth shock magnitude {payload.get('magnitude', payload.get('amount', 0.0))}"

            if content:
                importance = {"wealth_shock": 0.9, "narrative": 0.75, "signal_update": 0.6}.get(event_type, 0.5)
                fact_type = "shock" if event_type == "wealth_shock" else event_type
                self.collective_memory.record(round_id, fact_type, str(content), importance=importance)

    # ── Per-round metrics ────────────────────────────────────────────────────

    def _log_round_metrics(self) -> None:
        """Compute and store aggregate metrics for the current round.

        Tracks action distribution, wealth Gini, mean wealth/stress/satisfaction.
        Emits a warning if action collapse is detected (>90% same action).
        """
        round_id = self.world.state.round_id

        # Action distribution from last_action
        actions = [a.state.last_action for a in self.agents if a.state.last_action]
        action_counts = dict(Counter(actions))
        total_actions = sum(action_counts.values())
        action_dist = {k: round(v / total_actions, 3) if total_actions > 0 else 0 for k, v in action_counts.items()}
        if self.collective_memory is not None and action_dist.get("cooperate", 0.0) >= 0.6:
            self.collective_memory.record(
                round_id,
                "norm_shift",
                f"Cooperation reached {action_dist['cooperate']:.0%} of recorded actions.",
                importance=0.7,
            )

        # Wealth stats
        wealths = [a.state.wealth for a in self.agents]
        mean_wealth = sum(wealths) / len(wealths) if wealths else 0.0
        gini = self._compute_gini(wealths)

        # Stress and satisfaction stats
        stresses = [a.state.stress for a in self.agents]
        satisfactions = [a.state.satisfaction for a in self.agents]
        mean_stress = sum(stresses) / len(stresses) if stresses else 0.0
        mean_satisfaction = sum(satisfactions) / len(satisfactions) if satisfactions else 0.0

        # LLM output quality stats for this round (MiroFish get_stats() pattern).
        # Captures how often each parse strategy was used so drift and
        # JSON-degradation trends are visible in experiment metrics.
        parse_stats = get_parse_stats()
        reset_parse_stats()
        llm_quality = {
            "direct_json": parse_stats.get("direct_json", 0),
            "regex_json": parse_stats.get("regex_json", 0),
            "keyword_fallback": parse_stats.get("keyword_fallback", 0),
            "field_extract": parse_stats.get("field_extract", 0),
            "retry_success": parse_stats.get("retry_success", 0),
            "retry_exhausted": parse_stats.get("retry_exhausted", 0),
            "failed": parse_stats.get("failed", 0),
        }
        degraded = (
            llm_quality["keyword_fallback"]
            + llm_quality["field_extract"]
            + llm_quality["retry_exhausted"]
            + llm_quality["failed"]
        )
        if degraded > 0:
            logger.info(
                "Round %d LLM quality: %d degraded parse(s) (keyword=%d field_extract=%d exhausted=%d failed=%d)",
                round_id,
                degraded,
                llm_quality["keyword_fallback"],
                llm_quality["field_extract"],
                llm_quality["retry_exhausted"],
                llm_quality["failed"],
            )

        metrics = {
            "round_id": round_id,
            "action_distribution": action_dist,
            "action_counts": action_counts,
            "gini": round(gini, 4),
            "mean_wealth": round(mean_wealth, 2),
            "mean_stress": round(mean_stress, 4),
            "mean_satisfaction": round(mean_satisfaction, 4),
            "n_agents": len(self.agents),
            "llm_quality": llm_quality,
        }

        if self.social_env is not None:
            metrics["social_actions"] = self.social_env.get_stats()

        self.round_metrics.append(metrics)

        # Flush to disk every N rounds to keep the in-memory list bounded.
        # round_metrics.jsonl accumulates all historical data; round_metrics
        # in memory only ever holds _METRICS_FLUSH_INTERVAL entries at a time.
        # When no flush path is configured (e.g. unit tests, kernels created
        # without heartbeat_path), _flush_round_metrics is a no-op but we
        # still clear the list so it cannot grow unbounded over long runs.
        if len(self.round_metrics) >= _METRICS_FLUSH_INTERVAL:
            self._flush_round_metrics()
            self.round_metrics.clear()
        elif self._metrics_flush_path is None and len(self.round_metrics) > _METRICS_FLUSH_INTERVAL * 4:
            # Safety net: even without a flush path, cap in-memory growth
            # at 4× the flush interval so the list cannot leak in tests.
            self.round_metrics = self.round_metrics[-_METRICS_FLUSH_INTERVAL:]

        # Early warning: action collapse detection
        if action_dist:
            max_action_pct = max(action_dist.values())
            if max_action_pct > 0.90 and total_actions >= 3:
                dominant_action = max(action_dist, key=action_dist.get)
                warnings.warn(
                    f"Round {round_id}: Action collapse detected — "
                    f"{dominant_action} accounts for {max_action_pct:.0%} of actions.",
                    stacklevel=2,
                )

    def _flush_round_metrics(self) -> None:
        """Append the current in-memory round_metrics batch to disk.

        Called automatically every _METRICS_FLUSH_INTERVAL rounds.  If no
        heartbeat path is configured (e.g., in unit tests) the flush is skipped
        and the in-memory list is simply cleared by the caller.
        """
        if not self._metrics_flush_path or not self.round_metrics:
            return
        self._metrics_flush_path.parent.mkdir(parents=True, exist_ok=True)
        with self._metrics_flush_path.open("a", encoding="utf-8") as f:
            for m in self.round_metrics:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @staticmethod
    def _compute_gini(values: list[float]) -> float:
        """Thin wrapper around the canonical Gini implementation."""
        if not values or len(values) < 2:
            return 0.0
        return float(_gini_canonical(values))
