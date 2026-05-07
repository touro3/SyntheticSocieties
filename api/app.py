"""
BGF REST API — remote simulation control and inspection.

Mirrors MiroFish's Flask blueprint architecture, adapted for BGF's
research workflow on a remote GPU server. Enables:
  - Triggering simulations from a laptop while the GPU runs remotely
  - Polling run status without SSH
  - Querying results and metrics without downloading experiment dirs
  - Interviewing agents via the IPC bridge

Authentication
──────────────
Set BGF_API_TOKEN in the environment to enable bearer-token auth.
All requests must include:  Authorization: Bearer <token>
Leave BGF_API_TOKEN unset to run in open mode (local / trusted networks only).

Run with:
    python api/app.py                          # dev server, localhost:5050
    gunicorn 'api.app:create_app()'            # production

Endpoints
─────────
  POST   /simulate                 Trigger a new simulation run (async)
  GET    /status/<exp_id>          Poll run_state.json for a run
  GET    /results/<exp_id>         Return summary.json + key metrics
  GET    /experiments              List all experiments in tracker
  POST   /interview/<exp_id>/<agent_id>  Interview a live agent via IPC
  POST   /inject/<exp_id>          Inject a live exogenous event via IPC
  GET    /report                   Run ReACT agent on a query (sync, slow)
  GET    /health                   Liveness probe
  GET    /incomplete               List resumable (non-complete) runs
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Flask dependency check ─────────────────────────────────────────────────────

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS

    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    _LIMITER_AVAILABLE = True
except ImportError:
    _LIMITER_AVAILABLE = False

_EXPERIMENTS_ROOT = Path("experiments")
_CONFIGS_ROOT = Path("configs")
_TRACKER_INDEX = Path("tracker/experiment_index.parquet")
_STATIC_DIR = Path(__file__).parent / "static"
_UPLOADS_DIR = Path("uploads") / "ess_data"

# Bearer-token auth.  Set BGF_API_TOKEN env var to enable.
# If unset, auth is disabled — intended for local / trusted-network use only.
_AUTH_TOKEN: str | None = os.environ.get("BGF_API_TOKEN")

# Valid experiment ID pattern: alphanumeric, underscores, hyphens, dots only.
# Blocks path-traversal sequences like "../", "%2e%2e", etc.
_EXP_ID_RE = re.compile(r"^[A-Za-z0-9_\-\.]{1,128}$")

# Valid LLM model name: HF model IDs (org/name) and OpenAI slugs.
# Allows letters, digits, dots, hyphens, forward-slashes, colons — nothing else.
_MODEL_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/:-]{0,100}$")
_INJECTION_TYPES = {"wealth_shock", "signal_update", "narrative"}

# UUID v4 pattern for uploaded file IDs.
_FILE_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")

# Maximum uploaded file size: 50 MB.
_UPLOAD_MAX_BYTES = 50 * 1024 * 1024

# BGF dimensions: (name, primary_col, computed_from, description)
# Mirrors the mapping in population/generator.py.  computed_from lists
# the source columns used to derive the dimension when primary_col is absent.
_BGF_DIMENSIONS: list[tuple[str, str, list[str], str]] = [
    ("age", "age", [], "Agent age → wealth init & risk defaults"),
    ("income", "income_decile", [], "Income decile → initial wealth"),
    (
        "trust_institutions",
        "trust_institutions",
        ["trust_parliament", "trust_legal", "trust_police"],
        "Institutional trust → cooperation bias",
    ),
    ("trust_people", "trust_people", [], "Interpersonal trust"),
    ("education", "education_level", [], "Education level"),
    ("location", "urbanization", [], "Urban / rural → network density"),
    ("political_preference", "left_right", [], "Political orientation"),
    ("risk_tolerance", "risk_taking", [], "Risk tolerance → decision variance"),
    ("social_activity", "social_meeting_freq", [], "Social interaction frequency"),
    ("competitiveness", "competitiveness", [], "Competitive drive"),
    ("leadership_preference", "leadership_preference", [], "Leadership tendency"),
    ("life_satisfaction", "life_satisfaction", [], "Wellbeing → cooperation tendency"),
    ("health_status", "self_rated_health", [], "Self-rated health"),
    ("religiosity", "religious_belonging", [], "Religious identity"),
    ("country", "country", [], "Country-level context"),
    ("gender", "gender", [], "Demographic attribute"),
]

_MAX_QUERY_LEN = 2000  # characters — cap for /report ?q= parameter

# Active IPC clients keyed by experiment_id (populated on demand)
_ipc_clients: dict[str, Any] = {}
_ipc_lock = threading.Lock()


# ── Security helpers ──────────────────────────────────────────────────────────


def _check_auth() -> tuple[bool, Any]:
    """Verify bearer token.  Returns (ok, error_response_or_None)."""
    if not _AUTH_TOKEN:
        return True, None  # Auth disabled — open mode

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False, (jsonify({"error": "Authorization header required"}), 401)

    provided = auth_header[len("Bearer ") :]
    # hmac.compare_digest prevents timing-oracle attacks.
    if not hmac.compare_digest(provided.encode(), _AUTH_TOKEN.encode()):
        logger.warning("Failed auth attempt from %s", request.remote_addr)
        return False, (jsonify({"error": "Invalid token"}), 403)

    return True, None


def _resolve_exp_dir(exp_id: str) -> Path:
    """Return the experiment directory path, raising ValueError on bad input.

    Rejects IDs that contain path-traversal sequences or non-alphanumeric
    characters before performing a resolve()-based containment check.
    """
    if not _EXP_ID_RE.match(exp_id):
        raise ValueError(f"Invalid experiment ID: {exp_id!r}")

    exp_dir = (_EXPERIMENTS_ROOT / exp_id).resolve()
    root = _EXPERIMENTS_ROOT.resolve()
    try:
        exp_dir.relative_to(root)
    except ValueError:
        raise ValueError(f"Path traversal blocked for exp_id: {exp_id!r}")
    return exp_dir


def _validate_config_path(config_path: str) -> Path:
    """Ensure config_path stays inside the configs/ directory and is a YAML file.

    Prevents an API caller from pointing a simulation at an arbitrary file
    anywhere on the filesystem (e.g., /etc/passwd, ~/.ssh/config).
    """
    p = Path(config_path).resolve()
    if p.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"config_path must be a YAML file (.yaml or .yml): {config_path!r}")
    root = _CONFIGS_ROOT.resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise ValueError(f"config_path must be within '{_CONFIGS_ROOT}/': {config_path!r}")
    return p


def _safe_json_file(path: Path) -> Any:
    """Read and parse a JSON file, returning None on any error."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ── App factory ───────────────────────────────────────────────────────────────


def create_app(
    experiments_root: str = "experiments",
    configs_root: str = "configs",
    allowed_origins: list[str] | str | None = None,
) -> Any:
    """Flask application factory.

    Args:
        experiments_root: Root directory for experiment outputs.
        configs_root:     Root directory for config YAMLs (used to sandbox
                          the /simulate config_path parameter).
        allowed_origins:  CORS origins.  Pass a list of URLs for production,
                          or None to allow all (fine for localhost / research).
    """
    if not _FLASK_AVAILABLE:
        raise ImportError("Flask and flask-cors are required for the API. Install with: pip install flask flask-cors")

    global _EXPERIMENTS_ROOT, _CONFIGS_ROOT
    _EXPERIMENTS_ROOT = Path(experiments_root)
    _CONFIGS_ROOT = Path(configs_root)

    app = Flask(__name__, template_folder=Path(__file__).parent / "templates", static_folder=None)

    # Allow up to 52 MB to accommodate ESS data file uploads (max 50 MB).
    # JSON-only endpoints send small payloads; the upload endpoint enforces
    # its own 50 MB cap after reading.
    app.config["MAX_CONTENT_LENGTH"] = 52 * 1024 * 1024  # 52 MB

    # CORS — restrict to caller-specified origins (default: open for research).
    CORS(app, origins=allowed_origins or "*")

    # ── Rate limiting (optional — requires flask-limiter) ────────────────────
    if _LIMITER_AVAILABLE:
        limiter = Limiter(
            get_remote_address,
            app=app,
            default_limits=["300 per hour", "60 per minute"],
            storage_uri="memory://",
        )
        _simulate_limit = limiter.limit("10 per minute")
        _report_limit = limiter.limit("5 per minute")
        _write_limit = limiter.limit("30 per minute")
        _read_limit = limiter.limit("60 per minute")
        _experiments_limit = limiter.limit("30 per minute")
        _incomplete_limit = limiter.limit("10 per minute")
    else:
        # No-op decorators if flask-limiter is not installed.
        def _noop(f):
            return f

        _simulate_limit = _report_limit = _write_limit = _noop
        _read_limit = _experiments_limit = _incomplete_limit = _noop
        logger.warning("flask-limiter not installed — rate limiting disabled. Install with: pip install flask-limiter")

    # ── Static assets for Vue SPA (built to api/static/) ─────────────────────

    @app.get("/assets/<path:filename>")
    def vue_assets(filename: str):
        from flask import send_from_directory

        assets_dir = _STATIC_DIR / "assets"
        if not assets_dir.exists():
            return ("", 404)
        return send_from_directory(str(assets_dir), filename)

    # ── Landing page / Vue SPA ────────────────────────────────────────────────

    @app.get("/")
    def index():
        from flask import render_template, send_from_directory

        idx = _STATIC_DIR / "index.html"
        if idx.exists():
            return send_from_directory(str(_STATIC_DIR), "index.html")
        # Fallback: serve old static template
        base_url = request.host_url.rstrip("/")
        return render_template("index.html", base_url=base_url)

    # ── List available config files ───────────────────────────────────────────

    @app.get("/configs")
    def list_configs():
        configs = []
        if _CONFIGS_ROOT.exists():
            for p in sorted(_CONFIGS_ROOT.rglob("*.yaml")):
                rel = str(p.relative_to(_CONFIGS_ROOT.parent))
                if "__pycache__" not in rel:
                    configs.append(rel)
        return jsonify({"configs": configs})

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "bgf-api"})

    # ── Simulate ──────────────────────────────────────────────────────────────

    @app.post("/simulate")
    @_simulate_limit
    def simulate():
        """
        Trigger a simulation run asynchronously.

        Body (JSON):
          config_path  str   Path to config YAML under configs/ (required)
          resume       str   experiment_id to resume (optional)

        Returns the experiment_id immediately; poll /status/<exp_id> for progress.
        """
        ok, err = _check_auth()
        if not ok:
            return err

        body = request.get_json(silent=True) or {}
        config_path_raw = body.get("config_path")
        resume = body.get("resume")

        if not config_path_raw:
            return jsonify({"error": "config_path is required"}), 400

        # Validate config_path is sandboxed within configs/
        try:
            config_path = _validate_config_path(str(config_path_raw))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if not config_path.exists():
            return jsonify({"error": "Config not found"}), 404

        # Validate resume ID if provided
        if resume is not None:
            if not _EXP_ID_RE.match(str(resume)):
                return jsonify({"error": "Invalid resume experiment ID"}), 400

        # Derive experiment_id from config
        try:
            import yaml

            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            exp_id = cfg.get("project", {}).get("experiment_id", "api_run")
        except Exception:
            exp_id = "api_run"

        cmd = [
            sys.executable,
            "scripts/run_config_simulation.py",
            "--config",
            str(config_path),
        ]
        if resume:
            cmd += ["--resume", str(resume)]

        def _run():
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                logger.error("Simulation subprocess failed (exp=%s): %s", exp_id, exc)

        thread = threading.Thread(target=_run, name=f"sim-{exp_id}", daemon=True)
        thread.start()

        return jsonify({"experiment_id": exp_id, "status": "started"}), 202

    # ── Status ────────────────────────────────────────────────────────────────

    @app.get("/status/<exp_id>")
    @_read_limit
    def status(exp_id: str):
        """Poll the run_state.json for a specific experiment."""
        ok, err = _check_auth()
        if not ok:
            return err

        try:
            exp_dir = _resolve_exp_dir(exp_id)
        except ValueError:
            return jsonify({"error": "Invalid experiment ID"}), 400

        state_path = exp_dir / "run_state.json"
        heartbeat_path = exp_dir / "heartbeat.json"

        if not state_path.exists():
            return jsonify({"error": "Experiment not found"}), 404

        state = _safe_json_file(state_path)
        if state is None:
            return jsonify({"error": "Could not read run state"}), 500

        heartbeat = _safe_json_file(heartbeat_path)
        if heartbeat is not None:
            state["heartbeat"] = heartbeat

        # Enrich with metadata fields so the UI has agent count and policy even before heartbeat
        meta = _safe_json_file(exp_dir / "metadata.json")
        if meta:
            if "policy_type" not in state:
                state["policy_type"] = meta.get("policy_type")
            if "total_agents" not in state:
                state["total_agents"] = meta.get("population_size")

        # While running, heartbeat.round_id is the authoritative per-round counter.
        # run_state.completed_rounds is only updated once at the end by run_mgr.tick(),
        # so use the heartbeat value during the run to keep the UI progress bar live.
        if state.get("status") == "running" and heartbeat:
            hb_round = heartbeat.get("round_id") or heartbeat.get("round") or 0
            if hb_round > state.get("completed_rounds", 0):
                state["completed_rounds"] = hb_round

        # Detect stale LLM run: running, 0 rounds completed, updated > 90s ago
        import time as _time

        if state.get("status") == "running" and state.get("completed_rounds", 0) == 0:
            updated_at = state.get("updated_at") or state.get("started_at") or 0
            if _time.time() - float(updated_at) > 90:
                state["stale"] = True
                if state.get("policy_type") in ("llm", "generative_agents"):
                    state["gpu_wait"] = True

        return jsonify(state)

    # ── Results ───────────────────────────────────────────────────────────────

    @app.get("/results/<exp_id>")
    @_read_limit
    def results(exp_id: str):
        """Return summary.json and metrics.json for a completed experiment."""
        ok, err = _check_auth()
        if not ok:
            return err

        try:
            exp_dir = _resolve_exp_dir(exp_id)
        except ValueError:
            return jsonify({"error": "Invalid experiment ID"}), 400

        if not exp_dir.exists():
            return jsonify({"error": "Experiment not found"}), 404

        payload: dict[str, Any] = {"experiment_id": exp_id}

        for fname in ("summary.json", "metrics.json", "metadata.json"):
            data = _safe_json_file(exp_dir / fname)
            if data is not None:
                payload[fname.replace(".json", "")] = data

        return jsonify(payload)

    # ── List experiments ──────────────────────────────────────────────────────

    @app.get("/experiments")
    @_experiments_limit
    def experiments_list():
        """List all experiments from the DuckDB tracker."""
        ok, err = _check_auth()
        if not ok:
            return err

        if not _TRACKER_INDEX.exists():
            # Fallback: scan experiments/ directory directly
            runs = []
            if _EXPERIMENTS_ROOT.exists():
                for exp_dir in sorted(_EXPERIMENTS_ROOT.iterdir(), reverse=True):
                    if not exp_dir.is_dir():
                        continue
                    state = _safe_json_file(exp_dir / "run_state.json")
                    meta = _safe_json_file(exp_dir / "metadata.json")
                    summ = _safe_json_file(exp_dir / "summary.json")
                    if state is None and meta is None:
                        continue
                    wealth = summ.get("wealth", {}).get("values", []) if summ else []
                    wealth_mean = sum(wealth) / len(wealth) if wealth else None
                    runs.append(
                        {
                            "experiment_id": exp_dir.name,
                            "status": (state or {}).get("status"),
                            "policy_type": (meta or {}).get("policy_type"),
                            "seed": (meta or {}).get("seed"),
                            "wealth_mean": round(wealth_mean, 3) if wealth_mean is not None else None,
                            "gini": None,
                        }
                    )
                    if len(runs) >= 500:
                        break
            return jsonify({"experiments": runs, "note": "Tracker index not available — using filesystem scan"})

        try:
            import duckdb

            # Enforce a hard cap on limit regardless of caller input.
            try:
                limit = min(int(request.args.get("limit", 50)), 500)
            except (ValueError, TypeError):
                limit = 50

            policy = request.args.get("policy") or None

            conn = duckdb.connect()
            # _TRACKER_INDEX is server-controlled (not user input), safe to format.
            parquet_path = str(_TRACKER_INDEX).replace("'", "''")
            conn.execute(
                f"CREATE VIEW exp AS SELECT * FROM read_parquet('{parquet_path}')",
            )

            # Use parameterized queries to prevent SQL injection.
            if policy:
                df = conn.execute(
                    """
                    SELECT experiment_id, policy_type, seed,
                           ROUND(wealth_mean, 3) AS wealth_mean,
                           ROUND(wealth_gini, 3) AS gini
                    FROM exp
                    WHERE policy_type = ?
                    ORDER BY experiment_id DESC
                    LIMIT ?
                    """,
                    [policy, limit],
                ).fetchdf()
            else:
                df = conn.execute(
                    """
                    SELECT experiment_id, policy_type, seed,
                           ROUND(wealth_mean, 3) AS wealth_mean,
                           ROUND(wealth_gini, 3) AS gini
                    FROM exp
                    ORDER BY experiment_id DESC
                    LIMIT ?
                    """,
                    [limit],
                ).fetchdf()

            return jsonify({"experiments": df.to_dict(orient="records")})

        except Exception as exc:
            logger.error("experiments_list error: %s", exc, exc_info=True)
            return jsonify({"error": "Failed to query experiments"}), 500

    # ── Agent interview ───────────────────────────────────────────────────────

    @app.post("/interview/<exp_id>/<agent_id>")
    @_write_limit
    def interview(exp_id: str, agent_id: str):
        """
        Interview a live agent via the file-system IPC bridge.

        Body (JSON):
          question  str   Question to ask the agent (required)

        The simulation must be running with SimulationIPCServer active.
        """
        ok, err = _check_auth()
        if not ok:
            return err

        try:
            exp_dir = _resolve_exp_dir(exp_id)
        except ValueError:
            return jsonify({"error": "Invalid experiment ID"}), 400

        # Validate agent_id with same pattern as exp_id
        if not _EXP_ID_RE.match(agent_id):
            return jsonify({"error": "Invalid agent ID"}), 400

        if not exp_dir.exists():
            return jsonify({"error": "Experiment not found"}), 404

        body = request.get_json(silent=True) or {}
        question = str(body.get("question", "Describe your recent decisions."))[:1000]

        # ── Try live IPC first (simulation must be running) ───────────────────
        state_data = _safe_json_file(exp_dir / "run_state.json") or {}
        run_status = state_data.get("status", "unknown")

        if run_status == "running":
            try:
                from simulation.ipc import SimulationIPCClient

                with _ipc_lock:
                    if exp_id not in _ipc_clients:
                        _ipc_clients[exp_id] = SimulationIPCClient(base_dir=str(exp_dir), timeout=15.0)
                    client = _ipc_clients[exp_id]

                reply = client.interview_agent(agent_id, question)
                return jsonify(reply)

            except TimeoutError:
                return jsonify({"error": "IPC timeout — simulation may have finished"}), 504
            except Exception as exc:
                logger.error("interview IPC error (exp=%s agent=%s): %s", exp_id, agent_id, exc)
                # Fall through to data-based replay

        # ── Data-based replay for completed / failed runs ─────────────────────
        events_path = exp_dir / "events.jsonl"
        if not events_path.exists():
            return jsonify({"error": "No events log found — interview requires a completed run with saved events"}), 404

        import json as _json

        # Load all events for this agent
        agent_events = []
        try:
            with events_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = _json.loads(line)
                        if ev.get("agent_id") == agent_id:
                            agent_events.append(ev)
                    except _json.JSONDecodeError:
                        continue
        except OSError as exc:
            return jsonify({"error": f"Could not read events: {exc}"}), 500

        if not agent_events:
            return jsonify({"error": f"No events found for {agent_id} in this experiment"}), 404

        # ── Derive statistics from events ─────────────────────────────────────
        total_rounds = len(agent_events)
        action_counts: dict = {}
        for ev in agent_events:
            a = ev.get("action", {}).get("action_type", "unknown")
            action_counts[a] = action_counts.get(a, 0) + 1
        dominant = max(action_counts, key=action_counts.get) if action_counts else "unknown"
        final_state = agent_events[-1].get("state_after", {})
        last_ev = agent_events[-1]
        last_action = last_ev.get("action", {}).get("action_type", "unknown")
        last_round = last_ev.get("round_id", total_rounds)
        last_reason = last_ev.get("action", {}).get("reasoning_summary", "")
        final_wealth = final_state.get("wealth", 0)
        first_wealth = agent_events[0].get("state_after", {}).get("wealth", final_wealth)
        final_stress = final_state.get("stress", 0)
        wealth_delta = final_wealth - first_wealth
        coop_count = action_counts.get("cooperate", 0)

        # ── Build per-agent social/interaction maps ───────────────────────────
        # neighbors seen across all rounds (union of perception.network.neighbors)
        all_neighbors: set = set()
        # agents cooperated WITH (target_agent_id when action=cooperate)
        coop_targets: dict = {}  # agent_id → count
        # agents who stole from us or we never cooperated with
        steal_targets: dict = {}  # agent_id → count

        for ev in agent_events:
            nb_list = ev.get("perception", {}).get("network", {}).get("neighbors", [])
            all_neighbors.update(nb_list)
            act_type = ev.get("action", {}).get("action_type", "")
            target = ev.get("action", {}).get("target_agent_id")
            if act_type == "cooperate" and target:
                coop_targets[target] = coop_targets.get(target, 0) + 1
            if act_type == "steal" and target:
                steal_targets[target] = steal_targets.get(target, 0) + 1

        # Neighbors never cooperated with = potential "don't trust" candidates
        never_coop_neighbors = sorted(all_neighbors - set(coop_targets.keys()))
        most_coop_neighbor = max(coop_targets, key=coop_targets.get) if coop_targets else None
        most_stolen_from = max(steal_targets, key=steal_targets.get) if steal_targets else None

        # Build a history block for the OpenAI prompt (last 10 rounds)
        history_lines = []
        for ev in agent_events[-10:]:
            r = ev.get("round_id", "?")
            act = ev.get("action", {}).get("action_type", "?")
            w = ev.get("state_after", {}).get("wealth")
            s = ev.get("state_after", {}).get("stress")
            rsn = ev.get("action", {}).get("reasoning_summary", "")
            # Strip LLM-fallback markers to present clean data
            rsn = rsn.replace("[LLM fallback: ", "").rstrip("]")
            entry = f"Round {r}: {act}"
            if w is not None:
                entry += f", wealth {w:.0f}"
            if s is not None:
                entry += f", stress {s:.2f}"
            if rsn:
                entry += f" — {rsn}"
            history_lines.append(entry)
        history_text = "\n".join(history_lines)

        # ── Try OpenAI for natural-language answer ────────────────────────────
        oai_key = os.environ.get("OPENAI_API_KEY", "")
        if oai_key:
            try:
                from openai import OpenAI as _OAI

                oai = _OAI(api_key=oai_key)
                system_prompt = (
                    f"You are {agent_id}, a synthetic economic agent in a completed simulation ({exp_id}). "
                    f"You played {total_rounds} rounds. Your dominant strategy was '{dominant}' "
                    f"({', '.join(f'{v}× {k}' for k, v in sorted(action_counts.items(), key=lambda x: -x[1]))}). "
                    f"Final wealth: {final_wealth:.1f} (started at {first_wealth:.1f}, delta {wealth_delta:+.1f}). "
                    f"Final stress: {final_stress:.2f}. Cooperation rounds: {coop_count}.\n\n"
                    f"Recent round history:\n{history_text}\n\n"
                    "Answer the user's question in first-person as this agent, "
                    "drawing only from the data above. Be natural and concise (2-4 sentences). "
                    "Do not mention 'LLM fallback' or simulation internals."
                )
                resp = oai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    max_tokens=220,
                    temperature=0.7,
                )
                answer = resp.choices[0].message.content.strip()
                return jsonify({"response": answer, "source": "replay_llm"})
            except Exception as exc:
                logger.warning("Replay LLM failed, using rule-based fallback: %s", exc)

        # ── Rule-based natural-language fallback (no LLM key needed) ─────────
        q_low = question.lower()

        def _wealth_feel(w: float) -> str:
            if w >= 120:
                return "quite wealthy and secure"
            if w >= 80:
                return "comfortable, though not rich"
            if w >= 50:
                return "adequate but cautious"
            return "stretched and concerned"

        def _stress_desc(s: float) -> str:
            if s > 0.4:
                return "quite stressed"
            if s > 0.1:
                return "moderately stressed"
            if s > -0.1:
                return "fairly relaxed"
            return "very calm and settled"

        last_reason_clean = last_reason.replace("[LLM fallback: ", "").rstrip("]")

        # Detect "which agent" / "who" specificity first — most specific branch
        _who_keywords = (
            "which agent",
            "who do",
            "who don't",
            "who don",
            "name an agent",
            "any agent",
            "specific agent",
            "which one",
            "who would",
        )
        _distrust_words = (
            "don't trust",
            "dont trust",
            "not trust",
            "least trust",
            "avoid",
            "suspicious",
            "who to avoid",
            "who not to",
            "don't you trust",
            "dont you trust",
            "you not trust",
            "distrust",
            "no trust",
            "wouldn't trust",
        )
        _trust_words = ("trust most", "most trust", "rely on", "cooperate with most", "best ally", "favorite")

        is_who_question = any(w in q_low for w in _who_keywords)
        # Also detect: "which agent ... trust ... not/don't/at all/never"
        _negations = ("not", "don't", "dont", "never", "at all", "least", "no")
        is_distrust_q = any(w in q_low for w in _distrust_words) or (
            is_who_question and "trust" in q_low and any(n in q_low for n in _negations)
        )
        is_trust_q = any(w in q_low for w in _trust_words)

        if is_who_question and is_distrust_q:
            # "Which agent don't you trust / would you avoid?"
            nb_list = sorted(all_neighbors)
            if most_stolen_from:
                answer = (
                    f"If I had to name one agent I'd be wary of, it would be {most_stolen_from} — "
                    f"they were on the receiving end of my steal actions {steal_targets[most_stolen_from]}× "
                    f"which tells you how I read their reliability."
                )
            elif never_coop_neighbors:
                target = never_coop_neighbors[0]
                answer = (
                    f"I never cooperated with {target} across all {total_rounds} rounds. "
                    f"That says something — I didn't see enough shared incentive to invest in that relationship."
                )
            elif nb_list:
                answer = (
                    f"I interacted with {', '.join(nb_list[:3])} as my main neighbors. "
                    f"Honestly, trust was low across the board — I defaulted to {dominant} most of the time."
                )
            else:
                answer = (
                    f"I didn't observe strong enough signals to single out one agent. "
                    f"I kept to myself, sticking with {dominant} for most of the {total_rounds} rounds."
                )

        elif is_who_question and is_trust_q:
            # "Which agent do you trust most / cooperated with most?"
            if most_coop_neighbor:
                answer = (
                    f"I cooperated with {most_coop_neighbor} the most — {coop_targets[most_coop_neighbor]}× across "
                    f"{total_rounds} rounds. That repeated cooperation signals genuine mutual benefit."
                )
            elif all_neighbors:
                nb_list = sorted(all_neighbors)
                answer = (
                    f"I mostly interacted with {', '.join(nb_list[:2])}. "
                    f"I didn't cooperate much overall ({coop_count}× total), so trust was generally low."
                )
            else:
                answer = (
                    f"I didn't form strong cooperative bonds — with {coop_count} cooperative rounds "
                    f"out of {total_rounds}, there's no clear standout partner."
                )

        elif any(w in q_low for w in ("last decision", "last round", "most recent", "round")):
            answer = (
                f"In round {last_round} I chose to {last_action}. "
                + (f"My reasoning: {last_reason_clean}. " if last_reason_clean else "")
                + f"At that point my wealth stood at {final_wealth:.0f} and I was {_stress_desc(final_stress)}."
            )

        elif any(w in q_low for w in ("feel", "wealth", "rich", "money", "current")):
            direction = "grown" if wealth_delta > 0 else ("stayed flat" if wealth_delta == 0 else "declined")
            answer = (
                f"My wealth has {direction} to {final_wealth:.0f} over {total_rounds} rounds "
                f"(started at {first_wealth:.0f}, delta {wealth_delta:+.0f}). "
                f"I feel {_wealth_feel(final_wealth)}, and I finished {_stress_desc(final_stress)}."
            )

        elif any(w in q_low for w in ("strategy", "change", "different", "would you", "approach")):
            action_str = ", ".join(f"{k} ({v}×)" for k, v in sorted(action_counts.items(), key=lambda x: -x[1]))
            alt = "cooperation" if dominant not in ("cooperate",) else "saving"
            answer = (
                f"My dominant strategy was {dominant} — actions: {action_str}. "
                + (f"I cooperated {coop_count} times, contributing to the shared pool. " if coop_count else "")
                + f"If I were to replay this, I'd probably experiment more with {alt} "
                f"given my final wealth of {final_wealth:.0f}."
            )

        elif any(w in q_low for w in ("cooperat", "social", "pool", "shared")):
            # General cooperation question (not "which agent")
            if coop_count > 0:
                partner_note = f" My main cooperation partner was {most_coop_neighbor}." if most_coop_neighbor else ""
                answer = (
                    f"I cooperated {coop_count} out of {total_rounds} rounds, "
                    f"investing in the shared pool when incentives aligned.{partner_note} "
                    f"Overall {dominant} stayed my primary action though."
                )
            else:
                answer = (
                    f"I didn't cooperate in any of my {total_rounds} rounds. "
                    f"I focused on {dominant} to manage my own resources — "
                    f"the marginal return from cooperation didn't outweigh the risk for me."
                )

        elif any(w in q_low for w in ("trust", "neighbor", "relationship")):
            # General trust / neighborhood question (not "which agent")
            nb_list = sorted(all_neighbors)
            if nb_list:
                coop_note = (
                    f" I cooperated with {most_coop_neighbor} {coop_targets[most_coop_neighbor]}× most."
                    if most_coop_neighbor
                    else " I didn't build cooperative ties with any of them."
                )
                answer = (
                    f"My network included {', '.join(nb_list[:4])}{'...' if len(nb_list) > 4 else ''}.{coop_note} "
                    f"Trust in this simulation was built through repeated cooperation, "
                    f"and I {'invested in it' if coop_count else 'kept my distance'}."
                )
            else:
                answer = (
                    f"I had limited social data from this run — "
                    f"my {total_rounds} rounds were dominated by {dominant}. "
                    f"Network neighbors weren't logged in detail for this experiment."
                )

        else:
            # Generic behavioral summary
            action_str = ", ".join(f"{k} {v}×" for k, v in sorted(action_counts.items(), key=lambda x: -x[1]))
            answer = (
                f"Across {total_rounds} rounds my main action was {dominant}. "
                f"Full breakdown: {action_str}. "
                f"I ended with wealth {final_wealth:.0f} and stress {final_stress:.2f} — "
                f"feeling {_wealth_feel(final_wealth)} overall."
            )

        return jsonify({"response": answer, "source": "replay_data"})

    # ── Live exogenous injection ─────────────────────────────────────────────

    @app.post("/inject/<exp_id>")
    @_write_limit
    def inject(exp_id: str):
        """
        Inject a variable or event into a running simulation.

        Body (JSON):
          event_type   str   "wealth_shock" | "signal_update" | "narrative"
          payload      dict  Event-type-specific parameters
        """
        ok, err = _check_auth()
        if not ok:
            return err

        try:
            exp_dir = _resolve_exp_dir(exp_id)
        except ValueError:
            return jsonify({"error": "Invalid experiment ID"}), 400

        if not exp_dir.exists():
            return jsonify({"error": "Experiment not found"}), 404

        body = request.get_json(silent=True) or {}
        event_type = str(body.get("event_type", "")).strip()
        payload = body.get("payload", {})

        if event_type not in _INJECTION_TYPES:
            return jsonify({"error": f"Invalid event_type. Expected one of: {sorted(_INJECTION_TYPES)}"}), 400
        if not isinstance(payload, dict):
            return jsonify({"error": "payload must be a JSON object"}), 400

        # ── Pre-check: sim must be running to accept injections ───────────────
        run_state_path = exp_dir / "run_state.json"
        if run_state_path.exists():
            try:
                import json as _json

                with run_state_path.open() as _f:
                    _rs = _json.load(_f)
                _status = _rs.get("status", "unknown")
                if _status in ("complete", "completed", "failed"):
                    return jsonify(
                        {
                            "error": f"Simulation is {_status} — inject only works on actively running simulations.",
                            "status": _status,
                        }
                    ), 409
            except Exception:
                pass

        try:
            from simulation.ipc import SimulationIPCClient

            with _ipc_lock:
                if exp_id not in _ipc_clients:
                    _ipc_clients[exp_id] = SimulationIPCClient(base_dir=str(exp_dir), timeout=8.0)
                client = _ipc_clients[exp_id]

            reply = client.inject_event(event_type, payload)
            return jsonify(reply)

        except TimeoutError:
            return jsonify(
                {"error": "IPC timeout — simulation may not have IPC enabled. Run a new simulation to inject events."}
            ), 504
        except Exception as exc:
            logger.error("inject error (exp=%s event=%s): %s", exp_id, event_type, exc)
            return jsonify({"error": f"Injection failed: {exc}"}), 500

    # ── ReACT report (synchronous, can be slow) ───────────────────────────────

    @app.get("/report")
    @_report_limit
    def report():
        """
        Run the ReACT report agent on a query.

        Query params:
          q       str   Research question (required)
          model   str   LLM model name (default: gpt-4o-mini)

        The LLM API key is read from the BGF_REPORT_API_KEY environment variable
        or falls back to OPENAI_API_KEY.  It must NOT be passed as a query param
        (doing so would expose it in server logs, browser history, and Referer headers).
        """
        ok, err = _check_auth()
        if not ok:
            return err

        query = request.args.get("q")
        model = request.args.get("model", "gpt-4o-mini")

        if not query:
            return jsonify({"error": "'q' query parameter is required"}), 400
        if len(query) > _MAX_QUERY_LEN:
            return jsonify({"error": f"'q' exceeds maximum length of {_MAX_QUERY_LEN} characters"}), 400
        if not _MODEL_RE.match(model):
            return jsonify({"error": "Invalid model name"}), 400

        # API key comes from the server environment only — never from the caller.
        api_key = os.environ.get("BGF_REPORT_API_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY")
        base_url = None  # Configurable via env if needed: os.environ.get("BGF_REPORT_BASE_URL")

        try:
            from analysis.react_report_agent import ReportAgent

            agent = ReportAgent(
                api_key=api_key,
                base_url=base_url,
                model=model,
                index_path=str(_TRACKER_INDEX),
            )
            text = agent.generate_report(query)
            return jsonify({"query": query, "report": text})

        except Exception as exc:
            logger.error("report error (q=%r): %s", query[:100], exc, exc_info=True)
            return jsonify({"error": "Report generation failed"}), 500

    # ── ESS data file upload ──────────────────────────────────────────────

    @app.post("/upload-ess-data")
    @_write_limit
    def upload_ess_data():
        """Upload a population data file (.csv or .parquet) for empirical grounding.

        No required columns — accepts any tabular file.  Returns a structured
        analysis of which BGF simulation dimensions the file covers (directly,
        via derivation, or via config fallback), mirroring the column mapping
        in population/generator.py.

        The returned file_id can be passed to /simulate-wizard as ess_data_file_id.
        """
        import io
        import uuid as _uuid

        import numpy as np
        import pandas as pd

        ok, err = _check_auth()
        if not ok:
            return err

        if "file" not in request.files:
            return jsonify({"error": "No file provided — send multipart/form-data with field 'file'"}), 400

        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400

        fname_lower = f.filename.lower()
        if not (fname_lower.endswith(".csv") or fname_lower.endswith(".parquet")):
            return jsonify({"error": "Only .csv and .parquet files are accepted"}), 400

        raw = f.read()
        if len(raw) > _UPLOAD_MAX_BYTES:
            return jsonify({"error": "File too large — maximum is 50 MB"}), 413

        try:
            if fname_lower.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(raw))
            else:
                df = pd.read_parquet(io.BytesIO(raw))
        except Exception as exc:
            return jsonify({"error": f"Could not parse file: {exc}"}), 422

        if df.empty:
            return jsonify({"error": "File contains no rows"}), 422

        # Derive trust_institutions if absent (mirrors ESSGrounder.load)
        cols = set(df.columns)
        if "trust_institutions" not in cols:
            src = [c for c in ("trust_parliament", "trust_legal", "trust_police") if c in cols]
            if src:
                df["trust_institutions"] = df[src].mean(axis=1)
                cols.add("trust_institutions")

        # ── Dimension analysis ────────────────────────────────────────────
        def _col_stats(col: str) -> dict:
            s = df[col].dropna()
            if s.empty:
                return {}
            out: dict = {"non_null_pct": round(len(s) / len(df) * 100, 1)}
            if pd.api.types.is_numeric_dtype(s):
                out["mean"] = round(float(s.mean()), 3)
                out["std"] = round(float(s.std()), 3)
                out["min"] = round(float(s.min()), 3)
                out["max"] = round(float(s.max()), 3)
            else:
                top = s.value_counts().head(3).to_dict()
                out["top_values"] = {str(k): int(v) for k, v in top.items()}
            return out

        dimensions = []
        direct = computed = fallback_count = 0

        for dim_name, primary_col, computed_from, description in _BGF_DIMENSIONS:
            if primary_col in cols:
                status = "direct"
                direct += 1
                source = primary_col
                stats = _col_stats(primary_col)
            elif any(c in cols for c in computed_from):
                status = "computed"
                computed += 1
                available = [c for c in computed_from if c in cols]
                source = ", ".join(available)
                derived = df[available].mean(axis=1)
                stats = {
                    "non_null_pct": round(derived.notna().sum() / len(df) * 100, 1),
                    "mean": round(float(derived.mean()), 3),
                    "std": round(float(derived.std()), 3),
                }
            else:
                status = "fallback"
                fallback_count += 1
                source = "config default"
                stats = {}

            dimensions.append(
                {
                    "name": dim_name,
                    "status": status,
                    "source": source,
                    "description": description,
                    "stats": stats,
                }
            )

        total_dims = len(_BGF_DIMENSIONS)
        covered = direct + computed
        coverage_pct = round(covered / total_dims * 100)

        # ── Per-column stats (numeric, capped at 30) ──────────────────────
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col_stats = []
        for col in numeric_cols[:30]:
            s = df[col].dropna()
            if s.empty:
                continue
            col_stats.append(
                {
                    "col": col,
                    "non_null_pct": round(len(s) / len(df) * 100, 1),
                    "mean": round(float(s.mean()), 3),
                    "std": round(float(s.std()), 3),
                    "min": round(float(s.min()), 3),
                    "max": round(float(s.max()), 3),
                }
            )

        overall_completeness = round(df.notna().sum().sum() / (len(df) * len(df.columns)) * 100, 1)

        analysis = {
            "dimensions": dimensions,
            "coverage": {
                "direct": direct,
                "computed": computed,
                "fallback": fallback_count,
                "total": total_dims,
                "pct": coverage_pct,
            },
            "column_stats": col_stats,
            "quality": {
                "completeness_pct": overall_completeness,
                "total_rows": len(df),
                "total_columns": len(df.columns),
            },
            "summary": (
                f"{len(df):,} rows · {len(df.columns)} columns · "
                f"{covered}/{total_dims} BGF dimensions covered ({coverage_pct}%)"
            ),
        }

        # ── Persist ───────────────────────────────────────────────────────
        _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        file_id = str(_uuid.uuid4())
        out_path = _UPLOADS_DIR / f"{file_id}.parquet"
        try:
            df.to_parquet(out_path, index=False)
        except Exception as exc:
            logger.error("ESS upload write failed: %s", exc)
            return jsonify({"error": "Failed to save uploaded file"}), 500

        logger.info(
            "Data uploaded: file_id=%s rows=%d cols=%d coverage=%d%%",
            file_id,
            len(df),
            len(df.columns),
            coverage_pct,
        )
        return jsonify(
            {
                "file_id": file_id,
                "rows": len(df),
                "columns": df.columns.tolist(),
                "analysis": analysis,
            }
        ), 200

    # ── No-code wizard simulate ──────────────────────────────────────────

    @app.post("/simulate-wizard")
    @_simulate_limit
    def simulate_wizard():
        """
        Launch a simulation from wizard parameters (no YAML knowledge needed).

        Body (JSON):
          policy            str   "mock"|"random"|"rule_based"|"template"|"llm"  (default: rule_based)
          agents            int   Population size 5-500                           (default: 20)
          rounds            int   Simulation rounds 5-100                        (default: 10)
          network_type      str   "random"|"small_world"                         (default: random)
          population_source str   "synthetic"|"empirical"                        (default: synthetic)
          seed              int   Random seed                                    (default: 42)
          bad_apple_frac    float Fraction of adversarial agents 0.0-0.3        (default: 0.0)
          ess_data_file_id  str   UUID returned by POST /upload-ess-data        (optional)
                                  When set with population_source=empirical,
                                  uses the uploaded file instead of the default
                                  ESS Round 11 dataset.
        """
        ok, err = _check_auth()
        if not ok:
            return err

        import time

        import yaml as _yaml

        body = request.get_json(silent=True) or {}

        # ── Validate and clamp wizard params ─────────────────────────────
        _VALID_POLICIES = {"mock", "random", "rule_based", "template", "llm", "generative_agents"}
        _VALID_NETWORKS = {"random", "small_world"}
        _VALID_SOURCES = {"synthetic", "empirical"}
        _VALID_BACKENDS = {"huggingface", "openai"}
        _VALID_OAI_MODELS = {"gpt-4o-mini", "gpt-4o", "gpt-4o-2024-11-20", "gpt-4-turbo", "o1-mini"}

        policy = str(body.get("policy", "rule_based"))
        if policy not in _VALID_POLICIES:
            return jsonify({"error": f"Invalid policy. Choose from: {sorted(_VALID_POLICIES)}"}), 400

        try:
            agents = max(5, min(500, int(body.get("agents", 20))))
            rounds = max(5, min(100, int(body.get("rounds", 10))))
            seed = int(body.get("seed", 42))
        except (TypeError, ValueError):
            return jsonify({"error": "agents, rounds, seed must be integers"}), 400

        network_type = str(body.get("network_type", "random"))
        if network_type not in _VALID_NETWORKS:
            return jsonify({"error": f"Invalid network_type. Choose from: {sorted(_VALID_NETWORKS)}"}), 400

        pop_source = str(body.get("population_source", "synthetic"))
        if pop_source not in _VALID_SOURCES:
            return jsonify({"error": f"Invalid population_source. Choose from: {sorted(_VALID_SOURCES)}"}), 400

        # Optional custom ESS data file (only meaningful when pop_source == "empirical")
        ess_data_path: str | None = None
        raw_file_id = str(body.get("ess_data_file_id", "")).strip()
        if raw_file_id:
            if not _FILE_ID_RE.match(raw_file_id):
                return jsonify({"error": "Invalid ess_data_file_id format"}), 400
            candidate = (_UPLOADS_DIR / f"{raw_file_id}.parquet").resolve()
            uploads_root = _UPLOADS_DIR.resolve()
            try:
                candidate.relative_to(uploads_root)
            except ValueError:
                return jsonify({"error": "Invalid ess_data_file_id"}), 400
            if not candidate.exists():
                return jsonify({"error": "Uploaded ESS data file not found — please re-upload"}), 404
            ess_data_path = str(candidate)

        try:
            bad_apple_frac = float(body.get("bad_apple_frac", 0.0))
            bad_apple_frac = max(0.0, min(0.3, bad_apple_frac))
        except (TypeError, ValueError):
            bad_apple_frac = 0.0

        # ── LLM backend / provider params ────────────────────────────────────
        _VALID_BACKENDS = {"huggingface", "openai", "groq", "ollama"}

        llm_backend = str(body.get("llm_backend", "openai"))
        if llm_backend not in _VALID_BACKENDS:
            llm_backend = "openai"

        llm_model_id = str(body.get("llm_model_id", "gpt-4o-mini"))

        # Accept key under either name
        llm_api_key = str(body.get("openai_api_key", "")).strip() or str(body.get("llm_api_key", "")).strip() or None

        # ── Generate experiment ID and YAML config ────────────────────────
        ts = int(time.time())
        exp_id = f"wizard_{policy}_{agents}a_{rounds}r_{ts}"

        wizard_dir = _CONFIGS_ROOT / "wizard"
        wizard_dir.mkdir(parents=True, exist_ok=True)
        config_path = wizard_dir / f"{exp_id}.yaml"

        cfg_dict = {
            "project": {
                "name": "bgf-wizard",
                "experiment_id": exp_id,
                "seed": seed,
            },
            "simulation": {
                "rounds": rounds,
                "population_size": agents,
            },
            "policy": {"type": policy},
            "population": {"source": pop_source},
            "network": {"type": network_type},
        }
        if ess_data_path:
            cfg_dict["data"] = {"ess_clean_path": ess_data_path}
        if bad_apple_frac > 0:
            cfg_dict["bad_apple"] = {"fraction": bad_apple_frac}
        if policy in ("llm", "generative_agents"):
            cfg_dict["llm"] = {
                "backend_type": llm_backend,
                "model_id": llm_model_id,
            }

        try:
            config_path.write_text(_yaml.dump(cfg_dict, default_flow_style=False))
        except Exception as exc:
            logger.error("Wizard config write failed: %s", exc)
            return jsonify({"error": "Failed to write wizard config"}), 500

        # ── Inherit base config values via --config + overrides ───────────
        base_cfg = _CONFIGS_ROOT / "base_config.yaml"
        cmd = [
            sys.executable,
            "scripts/run_config_simulation.py",
            "--config",
            str(base_cfg if base_cfg.exists() else config_path),
            f"project.experiment_id={exp_id}",
            f"project.seed={seed}",
            f"simulation.rounds={rounds}",
            f"simulation.population_size={agents}",
            f"policy.type={policy}",
            f"population.source={pop_source}",
            f"network.type={network_type}",
        ]
        if ess_data_path:
            cmd.append(f"data.ess_clean_path={ess_data_path}")

        # Append LLM-specific overrides for GPU/API policies
        run_env = None
        if policy in ("llm", "generative_agents"):
            cmd += [
                f"llm.backend_type={llm_backend}",
                f"llm.model_id={llm_model_id}",
            ]
            if llm_api_key:
                import os as _os

                env_var = "GROQ_API_KEY" if llm_backend == "groq" else "OPENAI_API_KEY"
                run_env = {**_os.environ, env_var: llm_api_key}

        def _run():
            try:
                subprocess.run(cmd, check=True, env=run_env)
            except subprocess.CalledProcessError as exc:
                logger.error("Wizard simulation failed (exp=%s): %s", exp_id, exc)

        thread = threading.Thread(target=_run, name=f"wiz-{exp_id}", daemon=True)
        thread.start()

        resp_params: dict = {
            "policy": policy,
            "agents": agents,
            "rounds": rounds,
            "network_type": network_type,
            "population_source": pop_source,
            "seed": seed,
            "bad_apple_frac": bad_apple_frac,
        }
        if raw_file_id:
            resp_params["ess_data_file_id"] = raw_file_id
        if policy in ("llm", "generative_agents"):
            resp_params["llm_backend"] = llm_backend
            resp_params["llm_model_id"] = llm_model_id

        return jsonify(
            {
                "experiment_id": exp_id,
                "status": "started",
                "params": resp_params,
            }
        ), 202

    # ── Scan incomplete runs ──────────────────────────────────────────────────

    @app.get("/incomplete")
    @_incomplete_limit
    def incomplete():
        """List all experiments with status != complete (resumable runs)."""
        ok, err = _check_auth()
        if not ok:
            return err

        try:
            from simulation.crash_recovery import scan_incomplete_runs

            runs = scan_incomplete_runs(str(_EXPERIMENTS_ROOT))
            return jsonify({"incomplete_runs": runs})
        except Exception as exc:
            logger.error("incomplete scan error: %s", exc, exc_info=True)
            return jsonify({"error": "Failed to scan incomplete runs"}), 500

    # ── 413 handler — body too large ─────────────────────────────────────────

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return jsonify({"error": "Request body too large (max 1 MB)"}), 413

    return app


# ── Dev server entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BGF REST API server")
    # Default to localhost — callers must explicitly pass 0.0.0.0 to expose externally.
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1 — set 0.0.0.0 to expose externally)"
    )
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--configs-root", default="configs")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if not _AUTH_TOKEN:
        logger.warning(
            "BGF_API_TOKEN is not set — running in open mode. Set this env var before exposing the API on a network."
        )

    app = create_app(
        experiments_root=args.experiments_root,
        configs_root=args.configs_root,
    )
    port = int(os.environ.get("PORT", args.port))
    app.run(host=args.host, port=port, debug=args.debug)
