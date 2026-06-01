"""
File-system IPC for post-hoc agent interviews.

Enables interactive querying of a running (or completed) simulation without
modifying kernel.py.  Mirrors SimulationIPC pattern:

  ┌─────────────────────┐        ipc_commands/       ┌──────────────────────┐
  │  SimulationIPCClient │ ──── cmd_<uuid>.json ────► │  SimulationIPCServer │
  │  (external process) │ ◄── resp_<uuid>.json ─────  │  (inside simulation) │
  └─────────────────────┘       ipc_responses/        └──────────────────────┘

Supported commands
──────────────────
  interview_agent   Query a single agent about its reasoning history.
  interview_batch   Query multiple agents at once.
  inject_event      Queue an exogenous world-state event for the next round.
  get_status        Return current round, agent count, alive agent list.
  list_agents       List all agent IDs registered with the server.

Usage (server side — inside simulation kernel or a long-lived process)
──────────────────────────────────────────────────────────────────────
    server = SimulationIPCServer(agents_registry, base_dir="experiments/run_001")
    server.start()          # spawns background polling thread
    ...                     # run simulation
    server.stop()

Usage (client side — interactive script or notebook)
────────────────────────────────────────────────────
    client = SimulationIPCClient(base_dir="experiments/run_001")
    reply  = client.interview_agent("agent_042", "Why did you cooperate in round 5?")
    print(reply["answer"])
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# How often the server polls for new command files (seconds).
_POLL_INTERVAL = 0.5

# How long the client waits for a response before timing out (seconds).
_CLIENT_TIMEOUT = 30.0

# Directory names relative to base_dir.
_CMD_DIR = "ipc_commands"
_RESP_DIR = "ipc_responses"


# ── Server ────────────────────────────────────────────────────────────────────


class SimulationIPCServer:
    """Background server that processes IPC commands from external processes.

    Maintains a reference to the live agents registry so it can answer
    interview queries without disrupting the simulation loop.

    Args:
        agents:      Dict mapping agent_id → agent object. The server reads
                     agent.memory and agent.profile from each entry.
        base_dir:    Experiment directory where ipc_commands/ and ipc_responses/
                     subdirectories are created.
        current_round_fn: Optional callable returning the current round number.
                     Useful when the kernel updates a shared integer.
    """

    def __init__(
        self,
        agents: dict[str, Any],
        base_dir: str | Path = ".",
        current_round_fn: Optional[Callable[[], int]] = None,
        world_state: Any | None = None,
    ):
        self._agents = agents
        self._world_state = world_state
        self._base = Path(base_dir)
        self._cmd_dir = self._base / _CMD_DIR
        self._resp_dir = self._base / _RESP_DIR
        self._current_round_fn = current_round_fn or (lambda: -1)

        self._cmd_dir.mkdir(parents=True, exist_ok=True)
        self._resp_dir.mkdir(parents=True, exist_ok=True)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, name="SimulationIPCServer", daemon=True)
        self._thread.start()
        logger.info("SimulationIPCServer started (base=%s)", self._base)

    def stop(self) -> None:
        """Stop the background polling thread gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("SimulationIPCServer stopped.")

    # ── Poll loop ─────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while self._running:
            try:
                for cmd_file in sorted(self._cmd_dir.glob("cmd_*.json")):
                    self._handle_command_file(cmd_file)
            except Exception as exc:
                logger.error("IPC poll error: %s", exc)
            time.sleep(_POLL_INTERVAL)

    def _handle_command_file(self, path: Path) -> None:
        try:
            raw = path.read_text(encoding="utf-8")
            cmd = json.loads(raw)
        except Exception as exc:
            logger.warning("Could not parse IPC command %s: %s", path.name, exc)
            path.unlink(missing_ok=True)
            return

        request_id = cmd.get("request_id", path.stem)
        command = cmd.get("command", "")
        payload = cmd.get("payload", {})

        try:
            result = self._dispatch(command, payload)
        except Exception as exc:
            result = {"error": str(exc)}

        response = {
            "request_id": request_id,
            "command": command,
            "result": result,
        }
        resp_path = self._resp_dir / f"resp_{request_id}.json"
        tmp_path = resp_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
        os.replace(tmp_path, resp_path)

        # Remove the command file after processing.
        path.unlink(missing_ok=True)

    # ── Command dispatch ──────────────────────────────────────────────────────

    def _dispatch(self, command: str, payload: dict) -> Any:
        handlers: dict[str, Callable] = {
            "interview_agent": self._cmd_interview_agent,
            "interview_batch": self._cmd_interview_batch,
            "inject_event": self._cmd_inject_event,
            "get_status": self._cmd_get_status,
            "list_agents": self._cmd_list_agents,
        }
        fn = handlers.get(command)
        if fn is None:
            return {"error": f"Unknown command '{command}'. Available: {list(handlers)}"}
        return fn(payload)

    def _cmd_interview_agent(self, payload: dict) -> dict:
        agent_id = payload.get("agent_id")
        question = payload.get("question", "Describe your recent decisions.")

        if not agent_id:
            return {"error": "Missing 'agent_id' in payload."}

        agent = self._agents.get(agent_id)
        if agent is None:
            return {"error": f"Agent '{agent_id}' not found."}

        # Static fallback so existing clients still receive `answer`. Newer
        # clients (api/app.py) prefer the structured context fields and do
        # LLM synthesis on their side — the subprocess hosting this server
        # has no OpenAI key, so we cannot synthesize here.
        answer = self._build_agent_answer(agent, question)
        return {
            "agent_id": agent_id,
            "question": question,
            "answer": answer,
            "live_context": self._collect_live_context(agent),
        }

    def _cmd_interview_batch(self, payload: dict) -> dict:
        agent_ids = payload.get("agent_ids", [])
        question = payload.get("question", "Describe your recent decisions.")

        results = {}
        for aid in agent_ids:
            agent = self._agents.get(aid)
            if agent is None:
                results[aid] = {"error": f"Agent '{aid}' not found."}
            else:
                results[aid] = self._build_agent_answer(agent, question)
        return {"question": question, "responses": results}

    def _cmd_inject_event(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {"error": "Injection payload must be a JSON object."}
        if not payload.get("event_type"):
            return {"error": "Missing 'event_type' in injection payload."}
        error = self._apply_injection(payload)
        if error:
            return {"error": error}
        return {"status": "ok", "round": self._current_round_fn()}

    def _cmd_get_status(self, payload: dict) -> dict:  # noqa: ARG002
        return {
            "current_round": self._current_round_fn(),
            "n_agents": len(self._agents),
            "agent_ids": list(self._agents),
        }

    def _cmd_list_agents(self, payload: dict) -> dict:  # noqa: ARG002
        agents_info = []
        for aid, agent in self._agents.items():
            entry: dict[str, Any] = {"agent_id": aid}
            if hasattr(agent, "state"):
                state = agent.state
                entry["wealth"] = round(getattr(state, "wealth", 0), 2)
                entry["stress"] = round(getattr(state, "stress", 0), 4)
            if hasattr(agent, "profile"):
                profile = agent.profile
                entry["country"] = getattr(profile, "country", None)
                entry["age"] = getattr(profile, "age", None)
            agents_info.append(entry)
        return {"agents": agents_info}

    def _apply_injection(self, event: dict) -> str | None:
        if self._world_state is None:
            return "IPC server was not configured with a world_state."
        with self._lock:
            pending = getattr(self._world_state, "pending_injections", None)
            if pending is None:
                return "world_state does not expose pending_injections."
            pending.append(event)
        return None

    # ── Live context collector ────────────────────────────────────────────────

    @staticmethod
    def _collect_live_context(agent: Any) -> dict:
        """Return a JSON-safe snapshot of the live agent's state + memory.

        The API consumes this to drive an LLM synthesis pass on its own
        side (the subprocess hosting this IPC server has no OpenAI key).
        """
        ctx: dict[str, Any] = {}

        state = getattr(agent, "state", None)
        if state is not None:
            ctx["state"] = {
                "wealth": getattr(state, "wealth", None),
                "stress": getattr(state, "stress", None),
                "satisfaction": getattr(state, "satisfaction", None),
            }

        profile = getattr(agent, "profile", None)
        if profile is not None:
            ctx["profile"] = {
                k: getattr(profile, k, None)
                for k in (
                    "agent_id",
                    "age",
                    "gender",
                    "country",
                    "education_level",
                    "income_decile",
                    "trust_people",
                    "trust_institutions",
                    "risk_tolerance",
                    "competitiveness",
                    "left_right",
                    "political_orientation",
                    "life_satisfaction",
                    "is_adversarial",
                )
                if getattr(profile, k, None) is not None
            }

        mem = getattr(agent, "memory", None)
        if mem is not None:
            if hasattr(mem, "generate_reflection"):
                try:
                    ctx["memory_reflection"] = mem.generate_reflection() or ""
                except Exception:
                    ctx["memory_reflection"] = ""
            if hasattr(mem, "get_recent"):
                try:
                    recent = mem.get_recent(8) or []
                    ctx["recent_events"] = [
                        {
                            "round_id": getattr(item, "round_id", None),
                            "event_type": getattr(item, "event_type", None),
                            "partner_id": getattr(item, "partner_id", None),
                        }
                        for item in recent
                    ]
                except Exception:
                    ctx["recent_events"] = []
        return ctx

    # ── Answer builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_agent_answer(agent: Any, question: str) -> str:
        """Construct a natural-language answer from agent memory and state."""
        parts: list[str] = []

        # State snapshot
        if hasattr(agent, "state"):
            s = agent.state
            parts.append(
                f"Current state — wealth: {getattr(s, 'wealth', '?'):.1f}, stress: {getattr(s, 'stress', '?'):.2f}."
            )

        # Memory reflection
        if hasattr(agent, "memory"):
            mem = agent.memory
            if hasattr(mem, "generate_reflection"):
                ref = mem.generate_reflection()
                if ref:
                    parts.append(f"Memory summary: {ref}")
            if hasattr(mem, "get_recent"):
                recent = mem.get_recent(5)
                if recent:
                    lines = []
                    for item in recent:
                        line = f"  Round {item.round_id}: {item.event_type}"
                        if item.partner_id:
                            line += f" with {item.partner_id}"
                        lines.append(line)
                    parts.append("Recent actions:\n" + "\n".join(lines))

        if not parts:
            return f"[No state or memory data available for question: {question!r}]"

        return "\n".join(parts)


# ── Client ────────────────────────────────────────────────────────────────────


class SimulationIPCClient:
    """Client for sending commands to a running SimulationIPCServer.

    Args:
        base_dir: Same experiment directory used by the server.
        timeout:  Seconds to wait for a response before raising TimeoutError.
    """

    def __init__(
        self,
        base_dir: str | Path = ".",
        timeout: float = _CLIENT_TIMEOUT,
    ):
        self._base = Path(base_dir)
        self._cmd_dir = self._base / _CMD_DIR
        self._resp_dir = self._base / _RESP_DIR
        self._timeout = timeout

    # ── Generic send/receive ──────────────────────────────────────────────────

    def send(self, command: str, payload: dict | None = None) -> dict:
        """Send a command and block until a response arrives.

        Args:
            command: Command name (e.g. 'interview_agent').
            payload: Command-specific arguments.

        Returns:
            The 'result' dict from the server response.

        Raises:
            TimeoutError: If no response arrives within ``self.timeout`` seconds.
        """
        request_id = str(uuid.uuid4()).replace("-", "")[:16]
        cmd_obj = {
            "request_id": request_id,
            "command": command,
            "payload": payload or {},
        }
        self._cmd_dir.mkdir(parents=True, exist_ok=True)
        cmd_path = self._cmd_dir / f"cmd_{request_id}.json"
        tmp_path = cmd_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(cmd_obj, indent=2), encoding="utf-8")
        os.replace(tmp_path, cmd_path)

        # Poll for response
        resp_path = self._resp_dir / f"resp_{request_id}.json"
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if resp_path.exists():
                raw = resp_path.read_text(encoding="utf-8")
                # Guard against reading a partially-written / empty file
                # (race condition where the file exists but the server
                # hasn't finished flushing the JSON content yet).
                if raw.strip():
                    try:
                        data = json.loads(raw)
                        resp_path.unlink(missing_ok=True)
                        return data.get("result", data)
                    except json.JSONDecodeError:
                        # File exists but contains partial JSON — retry.
                        pass
            time.sleep(_POLL_INTERVAL)

        # Timed out — clean up the command file if still present
        cmd_path.unlink(missing_ok=True)
        raise TimeoutError(f"IPC command '{command}' (id={request_id}) timed out after {self._timeout}s.")

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def interview_agent(self, agent_id: str, question: str) -> dict:
        """Ask a single agent a natural-language question."""
        return self.send("interview_agent", {"agent_id": agent_id, "question": question})

    def interview_batch(self, agent_ids: list[str], question: str) -> dict:
        """Ask multiple agents the same question in one round-trip."""
        return self.send("interview_batch", {"agent_ids": agent_ids, "question": question})

    def inject_event(self, event_type: str | dict, payload: dict | None = None) -> dict:
        """Queue an exogenous event in the running simulation."""
        if isinstance(event_type, dict) and payload is None:
            event = event_type
        else:
            event = {"event_type": event_type, "payload": payload or {}}
        return self.send("inject_event", event)

    def get_status(self) -> dict:
        """Return current round, agent count, and agent ID list."""
        return self.send("get_status")

    def list_agents(self) -> list[dict]:
        """Return a list of agent info dicts (id, wealth, stress, country, age)."""
        result = self.send("list_agents")
        return result.get("agents", [])
