"""
File-system IPC for post-hoc agent interviews.

Enables interactive querying of a running (or completed) simulation without
modifying kernel.py.  Mirrors MiroFish's SimulationIPC pattern:

  ┌─────────────────────┐        ipc_commands/       ┌──────────────────────┐
  │  SimulationIPCClient │ ──── cmd_<uuid>.json ────► │  SimulationIPCServer │
  │  (external process) │ ◄── resp_<uuid>.json ─────  │  (inside simulation) │
  └─────────────────────┘       ipc_responses/        └──────────────────────┘

Supported commands
──────────────────
  interview_agent   Query a single agent about its reasoning history.
  interview_batch   Query multiple agents at once.
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
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# How often the server polls for new command files (seconds).
_POLL_INTERVAL = 0.5

# How long the client waits for a response before timing out (seconds).
_CLIENT_TIMEOUT = 30.0

# Directory names relative to base_dir.
_CMD_DIR  = "ipc_commands"
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
    ):
        self._agents = agents
        self._base = Path(base_dir)
        self._cmd_dir  = self._base / _CMD_DIR
        self._resp_dir = self._base / _RESP_DIR
        self._current_round_fn = current_round_fn or (lambda: -1)

        self._cmd_dir.mkdir(parents=True, exist_ok=True)
        self._resp_dir.mkdir(parents=True, exist_ok=True)

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, name="SimulationIPCServer", daemon=True
        )
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
        command    = cmd.get("command", "")
        payload    = cmd.get("payload", {})

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
        resp_path.write_text(json.dumps(response, indent=2), encoding="utf-8")

        # Remove the command file after processing.
        path.unlink(missing_ok=True)

    # ── Command dispatch ──────────────────────────────────────────────────────

    def _dispatch(self, command: str, payload: dict) -> Any:
        handlers: dict[str, Callable] = {
            "interview_agent": self._cmd_interview_agent,
            "interview_batch":  self._cmd_interview_batch,
            "get_status":       self._cmd_get_status,
            "list_agents":      self._cmd_list_agents,
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

        answer = self._build_agent_answer(agent, question)
        return {"agent_id": agent_id, "question": question, "answer": answer}

    def _cmd_interview_batch(self, payload: dict) -> dict:
        agent_ids = payload.get("agent_ids", [])
        question  = payload.get("question", "Describe your recent decisions.")

        results = {}
        for aid in agent_ids:
            agent = self._agents.get(aid)
            if agent is None:
                results[aid] = {"error": f"Agent '{aid}' not found."}
            else:
                results[aid] = self._build_agent_answer(agent, question)
        return {"question": question, "responses": results}

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
                entry["wealth"]  = round(getattr(state, "wealth", 0), 2)
                entry["stress"]  = round(getattr(state, "stress", 0), 4)
            if hasattr(agent, "profile"):
                profile = agent.profile
                entry["country"] = getattr(profile, "country", None)
                entry["age"]     = getattr(profile, "age", None)
            agents_info.append(entry)
        return {"agents": agents_info}

    # ── Answer builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_agent_answer(agent: Any, question: str) -> str:
        """Construct a natural-language answer from agent memory and state."""
        parts: list[str] = []

        # State snapshot
        if hasattr(agent, "state"):
            s = agent.state
            parts.append(
                f"Current state — wealth: {getattr(s, 'wealth', '?'):.1f}, "
                f"stress: {getattr(s, 'stress', '?'):.2f}."
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
        self._cmd_dir  = self._base / _CMD_DIR
        self._resp_dir = self._base / _RESP_DIR
        self._timeout  = timeout

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
            "command":    command,
            "payload":    payload or {},
        }
        self._cmd_dir.mkdir(parents=True, exist_ok=True)
        cmd_path = self._cmd_dir / f"cmd_{request_id}.json"
        cmd_path.write_text(json.dumps(cmd_obj, indent=2), encoding="utf-8")

        # Poll for response
        resp_path = self._resp_dir / f"resp_{request_id}.json"
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if resp_path.exists():
                raw = resp_path.read_text(encoding="utf-8")
                resp_path.unlink(missing_ok=True)
                data = json.loads(raw)
                return data.get("result", data)
            time.sleep(_POLL_INTERVAL)

        # Timed out — clean up the command file if still present
        cmd_path.unlink(missing_ok=True)
        raise TimeoutError(
            f"IPC command '{command}' (id={request_id}) timed out after {self._timeout}s."
        )

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def interview_agent(self, agent_id: str, question: str) -> dict:
        """Ask a single agent a natural-language question."""
        return self.send("interview_agent", {"agent_id": agent_id, "question": question})

    def interview_batch(self, agent_ids: list[str], question: str) -> dict:
        """Ask multiple agents the same question in one round-trip."""
        return self.send("interview_batch", {"agent_ids": agent_ids, "question": question})

    def get_status(self) -> dict:
        """Return current round, agent count, and agent ID list."""
        return self.send("get_status")

    def list_agents(self) -> list[dict]:
        """Return a list of agent info dicts (id, wealth, stress, country, age)."""
        result = self.send("list_agents")
        return result.get("agents", [])
