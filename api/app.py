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

    app = Flask(__name__, template_folder=Path(__file__).parent / "templates")

    # Limit incoming request bodies to 1 MB to prevent memory exhaustion.
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB

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

    # ── Landing page ──────────────────────────────────────────────────────────

    @app.get("/")
    def index():
        from flask import render_template

        base_url = request.host_url.rstrip("/")
        return render_template("index.html", base_url=base_url)

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
            return jsonify({"experiments": [], "note": "Tracker index not found"})

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

        try:
            from simulation.ipc import SimulationIPCClient

            with _ipc_lock:
                if exp_id not in _ipc_clients:
                    _ipc_clients[exp_id] = SimulationIPCClient(base_dir=str(exp_dir), timeout=15.0)
                client = _ipc_clients[exp_id]

            reply = client.interview_agent(agent_id, question)
            return jsonify(reply)

        except TimeoutError:
            return jsonify({"error": "IPC timeout — is the simulation running with IPC enabled?"}), 504
        except Exception as exc:
            logger.error("interview error (exp=%s agent=%s): %s", exp_id, agent_id, exc)
            return jsonify({"error": "Interview request failed"}), 500

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

        try:
            from simulation.ipc import SimulationIPCClient

            with _ipc_lock:
                if exp_id not in _ipc_clients:
                    _ipc_clients[exp_id] = SimulationIPCClient(base_dir=str(exp_dir), timeout=15.0)
                client = _ipc_clients[exp_id]

            reply = client.inject_event(event_type, payload)
            return jsonify(reply)

        except TimeoutError:
            return jsonify({"error": "IPC timeout — is the simulation running with IPC enabled?"}), 504
        except Exception as exc:
            logger.error("inject error (exp=%s event=%s): %s", exp_id, event_type, exc)
            return jsonify({"error": "Injection request failed"}), 500

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
