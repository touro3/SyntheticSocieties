"""
Human Baseline Experiment Server (Phase 29.2).

Flask backend that replicates the BGF economic game for human participants.
Reuses canonical payoff constants from environment/payoffs.py.

Endpoints:
    POST /session              Create session, get 3 random neighbors + initial state
    POST /action               Submit action, receive updated state
    GET  /status/<session_id>  Check current round and state
    POST /complete             Save participant data to CSV
    GET  /health               Liveness probe

Usage:
    cd human_experiment && python server/server.py
    # Opens at http://localhost:5050
"""

from __future__ import annotations

import csv
import logging
import random
import sys
import threading
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
except ImportError:
    raise ImportError("Flask and flask-cors are required. Install with: pip install flask flask-cors")

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    _LIMITER_AVAILABLE = True
except ImportError:
    _LIMITER_AVAILABLE = False

from environment.payoffs import DEFAULT_PAYOFFS  # noqa: E402 — requires sys.path patch above

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_ROUNDS = 10
INITIAL_WEALTH = 50.0
INITIAL_STRESS = 0.3
COOPERATE_AMOUNT = 5.0  # fixed donation amount (matches DEFAULT_COOPERATE_AMOUNT)
NEIGHBOR_POOL = [f"neighbor_{chr(65 + i)}" for i in range(6)]  # A-F

DATA_DIR = REPO_ROOT / "data" / "human"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESPONSES_CSV = DATA_DIR / "responses.csv"

_CSV_HEADERS = [
    "participant_id",
    "round_id",
    "action",
    "target",
    "wealth_after",
    "stress_after",
    "pre_trust",
    "pre_risk",
    "cooperation_count",
    "total_rounds",
]

# ── In-memory session store ───────────────────────────────────────────────────

_sessions: dict[str, dict] = {}
_session_lock = threading.Lock()


def _new_session(pre_trust: float, pre_risk: float) -> dict:
    """Initialise a fresh participant session."""
    neighbors = random.sample(NEIGHBOR_POOL, 3)
    return {
        "session_id": str(uuid.uuid4()),
        "pre_trust": float(pre_trust),
        "pre_risk": float(pre_risk),
        "wealth": INITIAL_WEALTH,
        "stress": INITIAL_STRESS,
        "round_id": 0,
        "neighbors": neighbors,
        "actions": [],  # list of {round_id, action, target}
        "cooperation_count": 0,
        "complete": False,
    }


def _apply_action(session: dict, action: str, target: str | None) -> dict:
    """Apply game payoffs and return updated state dict."""
    p = DEFAULT_PAYOFFS
    wealth = session["wealth"]
    stress = session["stress"]
    wealth_delta = 0.0
    stress_delta = 0.0
    target_used = None

    if action == "work":
        wealth_delta = p.work_income
        stress_delta = p.work_stress_increase

    elif action == "save":
        wealth_delta = p.save_wealth_delta
        stress_delta = p.save_stress_relief

    elif action == "cooperate":
        if target not in session["neighbors"]:
            # Invalid target — fall back to work
            action = "work"
            wealth_delta = p.work_income
            stress_delta = p.work_stress_increase
        else:
            wealth_delta = -COOPERATE_AMOUNT
            stress_delta = p.cooperate_stress_relief
            target_used = target
            session["cooperation_count"] += 1

    # Clamp
    new_wealth = max(0.0, wealth + wealth_delta)
    new_stress = max(0.0, min(1.0, stress + stress_delta))

    session["wealth"] = round(new_wealth, 2)
    session["stress"] = round(new_stress, 3)
    session["round_id"] += 1
    session["actions"].append(
        {
            "round_id": session["round_id"],
            "action": action,
            "target": target_used,
        }
    )

    return {
        "round_id": session["round_id"],
        "wealth": session["wealth"],
        "stress": session["stress"],
        "action": action,
        "wealth_delta": round(wealth_delta, 2),
        "done": session["round_id"] >= NUM_ROUNDS,
    }


def _append_csv(session: dict) -> None:
    """Append one row per round to the responses CSV."""
    file_exists = RESPONSES_CSV.exists()
    with open(RESPONSES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        for act in session["actions"]:
            writer.writerow(
                {
                    "participant_id": session["session_id"],
                    "round_id": act["round_id"],
                    "action": act["action"],
                    "target": act.get("target", ""),
                    "wealth_after": session["wealth"],
                    "stress_after": session["stress"],
                    "pre_trust": session["pre_trust"],
                    "pre_risk": session["pre_risk"],
                    "cooperation_count": session["cooperation_count"],
                    "total_rounds": NUM_ROUNDS,
                }
            )


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(REPO_ROOT / "human_experiment" / "app" / "static"))
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # 64 KB — survey answers need no more

if _LIMITER_AVAILABLE:
    _limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per hour", "60 per minute"],
        storage_uri="memory://",
    )
    _session_limit = _limiter.limit("10 per minute")
    _action_limit = _limiter.limit("60 per minute")
    _complete_limit = _limiter.limit("5 per minute")
    _status_limit = _limiter.limit("30 per minute")
else:

    def _noop(f):
        return f

    _session_limit = _action_limit = _complete_limit = _status_limit = _noop
    logger.warning("flask-limiter not installed — rate limiting disabled.")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "human-experiment"})


@app.post("/session")
@_session_limit
def create_session():
    """Create a new participant session.

    Body (JSON):
        pre_trust  float  1-10 trust score from pre-survey
        pre_risk   float  1-10 risk tolerance score from pre-survey
    """
    body = request.get_json(silent=True) or {}
    try:
        pre_trust = float(body.get("pre_trust", 5))
        pre_risk = float(body.get("pre_risk", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "pre_trust and pre_risk must be numbers"}), 400
    # Clamp to survey scale [1, 10] before normalising to [0, 1].
    pre_trust = max(1.0, min(10.0, pre_trust))
    pre_risk = max(1.0, min(10.0, pre_risk))

    # Normalise from 1-10 to 0-1
    trust_norm = (pre_trust - 1) / 9.0
    risk_norm = (pre_risk - 1) / 9.0

    session = _new_session(trust_norm, risk_norm)
    with _session_lock:
        _sessions[session["session_id"]] = session

    return jsonify(
        {
            "session_id": session["session_id"],
            "neighbors": session["neighbors"],
            "initial_wealth": session["wealth"],
            "initial_stress": session["stress"],
            "total_rounds": NUM_ROUNDS,
            "payoffs": {
                "work": {
                    "wealth": f"+{DEFAULT_PAYOFFS.work_income:.0f}",
                    "stress": f"+{DEFAULT_PAYOFFS.work_stress_increase:.0%}",
                },
                "save": {"wealth": "+0", "stress": f"{DEFAULT_PAYOFFS.save_stress_relief:.0%}"},
                "cooperate": {
                    "wealth": f"-{COOPERATE_AMOUNT:.0f}",
                    "stress": f"{DEFAULT_PAYOFFS.cooperate_stress_relief:.0%}",
                    "note": f"Neighbor receives +{COOPERATE_AMOUNT * DEFAULT_PAYOFFS.cooperation_multiplier:.0f}",
                },
            },
        }
    ), 201


@app.post("/action")
@_action_limit
def take_action():
    """Submit an action for the current round.

    Body (JSON):
        session_id  str   Session ID from /session
        action      str   "work" | "save" | "cooperate"
        target      str   Neighbor ID (required for cooperate)
    """
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id")
    action = body.get("action", "work").lower()
    target = body.get("target")

    with _session_lock:
        session = _sessions.get(session_id)

    if session is None:
        return jsonify({"error": "Session not found"}), 404
    if session["complete"]:
        return jsonify({"error": "Session already complete"}), 400
    if session["round_id"] >= NUM_ROUNDS:
        return jsonify({"error": "All rounds completed — call /complete"}), 400

    if action not in {"work", "save", "cooperate"}:
        return jsonify({"error": f"Invalid action: {action}"}), 400

    result = _apply_action(session, action, target)
    return jsonify(result)


@app.get("/status/<session_id>")
@_status_limit
def get_status(session_id: str):
    with _session_lock:
        session = _sessions.get(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(
        {
            "session_id": session_id,
            "round_id": session["round_id"],
            "wealth": session["wealth"],
            "stress": session["stress"],
            "cooperation_count": session["cooperation_count"],
            "complete": session["complete"],
        }
    )


@app.post("/complete")
@_complete_limit
def complete_session():
    """Mark the session as complete and persist the data.

    Body (JSON):
        session_id  str  Session ID
    """
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id")

    with _session_lock:
        session = _sessions.get(session_id)

    if session is None:
        return jsonify({"error": "Session not found"}), 404

    session["complete"] = True

    try:
        _append_csv(session)
    except Exception as exc:
        return jsonify({"error": f"Failed to save data: {exc}"}), 500

    coop_rate = session["cooperation_count"] / max(session["round_id"], 1)
    completion_code = f"BGF-{session['session_id'][:8].upper()}"

    return jsonify(
        {
            "completion_code": completion_code,
            "final_wealth": session["wealth"],
            "final_stress": session["stress"],
            "cooperation_rate": round(coop_rate, 3),
            "saved_to": str(RESPONSES_CSV),
        }
    )


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Human baseline experiment server")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1 — pass 0.0.0.0 to expose externally)"
    )
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO)
    print(f"Human experiment server starting on http://localhost:{args.port}")
    print(f"Data will be saved to: {RESPONSES_CSV}")
    app.run(host=args.host, port=args.port, debug=args.debug)
