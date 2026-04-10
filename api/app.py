"""
BGF REST API — remote simulation control and inspection.

Mirrors MiroFish's Flask blueprint architecture, adapted for BGF's
research workflow on a remote GPU server. Enables:
  - Triggering simulations from a laptop while the GPU runs remotely
  - Polling run status without SSH
  - Querying results and metrics without downloading experiment dirs
  - Interviewing agents via the IPC bridge

Run with:
    python api/app.py                       # dev server, port 5050
    gunicorn api.app:create_app()           # production

Endpoints
─────────
  POST   /simulate                 Trigger a new simulation run (async)
  GET    /status/<exp_id>          Poll run_state.json for a run
  GET    /results/<exp_id>         Return summary.json + key metrics
  GET    /experiments              List all experiments in tracker
  POST   /interview/<exp_id>/<agent_id>  Interview a live agent via IPC
  GET    /report                   Run ReACT agent on a query (sync, slow)
  GET    /health                   Liveness probe
"""

from __future__ import annotations

import json
import logging
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

_EXPERIMENTS_ROOT = Path("experiments")
_TRACKER_INDEX    = Path("tracker/experiment_index.parquet")

# Active IPC clients keyed by experiment_id (populated on demand)
_ipc_clients: dict[str, Any] = {}
_ipc_lock = threading.Lock()


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(experiments_root: str = "experiments") -> Any:
    """Flask application factory."""
    if not _FLASK_AVAILABLE:
        raise ImportError(
            "Flask and flask-cors are required for the API. "
            "Install with: pip install flask flask-cors"
        )

    global _EXPERIMENTS_ROOT
    _EXPERIMENTS_ROOT = Path(experiments_root)

    app = Flask(__name__)
    CORS(app)

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "bgf-api"})

    # ── Simulate ──────────────────────────────────────────────────────────────

    @app.post("/simulate")
    def simulate():
        """
        Trigger a simulation run asynchronously.

        Body (JSON):
          config_path  str   Path to config YAML (required)
          resume       str   experiment_id to resume (optional)

        Returns the experiment_id immediately; poll /status/<exp_id> for progress.
        """
        body = request.get_json(silent=True) or {}
        config_path = body.get("config_path")
        resume      = body.get("resume")

        if not config_path:
            return jsonify({"error": "config_path is required"}), 400
        if not Path(config_path).exists():
            return jsonify({"error": f"Config not found: {config_path}"}), 404

        # Derive experiment_id from config
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            exp_id = cfg.get("project", {}).get("experiment_id", "api_run")
        except Exception:
            exp_id = "api_run"

        cmd = [
            sys.executable, "scripts/run_config_simulation.py",
            "--config", config_path,
        ]
        if resume:
            cmd += ["--resume", resume]

        def _run():
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                logger.error("Simulation subprocess failed: %s", exc)

        thread = threading.Thread(target=_run, name=f"sim-{exp_id}", daemon=True)
        thread.start()

        return jsonify({"experiment_id": exp_id, "status": "started"}), 202

    # ── Status ────────────────────────────────────────────────────────────────

    @app.get("/status/<exp_id>")
    def status(exp_id: str):
        """Poll the run_state.json for a specific experiment."""
        state_path = _EXPERIMENTS_ROOT / exp_id / "run_state.json"
        heartbeat_path = _EXPERIMENTS_ROOT / exp_id / "heartbeat.json"

        if not state_path.exists():
            return jsonify({"error": f"No run_state.json for {exp_id}"}), 404

        state = json.loads(state_path.read_text())
        # Attach heartbeat if available
        if heartbeat_path.exists():
            try:
                state["heartbeat"] = json.loads(heartbeat_path.read_text())
            except Exception:
                pass

        return jsonify(state)

    # ── Results ───────────────────────────────────────────────────────────────

    @app.get("/results/<exp_id>")
    def results(exp_id: str):
        """Return summary.json and metrics.json for a completed experiment."""
        exp_dir = _EXPERIMENTS_ROOT / exp_id
        if not exp_dir.exists():
            return jsonify({"error": f"Experiment '{exp_id}' not found"}), 404

        payload: dict[str, Any] = {"experiment_id": exp_id}

        for fname in ("summary.json", "metrics.json", "metadata.json"):
            p = exp_dir / fname
            if p.exists():
                try:
                    payload[fname.replace(".json", "")] = json.loads(p.read_text())
                except Exception:
                    pass

        return jsonify(payload)

    # ── List experiments ──────────────────────────────────────────────────────

    @app.get("/experiments")
    def experiments_list():
        """List all experiments from the DuckDB tracker."""
        if not _TRACKER_INDEX.exists():
            return jsonify({"experiments": [], "note": "Tracker index not found"})
        try:
            import duckdb
            conn = duckdb.connect()
            conn.execute(f"CREATE VIEW exp AS SELECT * FROM read_parquet('{_TRACKER_INDEX}')")
            limit = int(request.args.get("limit", 50))
            policy = request.args.get("policy")
            where = f"WHERE policy_type = '{policy}'" if policy else ""
            df = conn.execute(f"""
                SELECT experiment_id, policy_type, seed,
                       ROUND(wealth_mean, 3) AS wealth_mean,
                       ROUND(wealth_gini, 3) AS gini
                FROM exp {where}
                ORDER BY experiment_id DESC
                LIMIT {limit}
            """).fetchdf()
            return jsonify({"experiments": df.to_dict(orient="records")})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # ── Agent interview ───────────────────────────────────────────────────────

    @app.post("/interview/<exp_id>/<agent_id>")
    def interview(exp_id: str, agent_id: str):
        """
        Interview a live agent via the file-system IPC bridge.

        Body (JSON):
          question  str   Question to ask the agent (required)

        The simulation must be running with SimulationIPCServer active.
        """
        body = request.get_json(silent=True) or {}
        question = body.get("question", "Describe your recent decisions.")

        exp_dir = _EXPERIMENTS_ROOT / exp_id
        if not exp_dir.exists():
            return jsonify({"error": f"Experiment dir not found: {exp_id}"}), 404

        try:
            from simulation.ipc import SimulationIPCClient
            with _ipc_lock:
                if exp_id not in _ipc_clients:
                    _ipc_clients[exp_id] = SimulationIPCClient(
                        base_dir=str(exp_dir), timeout=15.0
                    )
                client = _ipc_clients[exp_id]

            reply = client.interview_agent(agent_id, question)
            return jsonify(reply)
        except TimeoutError:
            return jsonify({"error": "IPC timeout — is the simulation running with IPC enabled?"}), 504
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # ── ReACT report (synchronous, can be slow) ───────────────────────────────

    @app.get("/report")
    def report():
        """
        Run the ReACT report agent on a query.

        Query params:
          q          str   Research question (required)
          model      str   LLM model name (default: gpt-4o-mini)
          api_key    str   LLM API key
          base_url   str   LLM base URL
        """
        query    = request.args.get("q")
        model    = request.args.get("model", "gpt-4o-mini")
        api_key  = request.args.get("api_key", "EMPTY")
        base_url = request.args.get("base_url") or None

        if not query:
            return jsonify({"error": "'q' query parameter is required"}), 400

        try:
            from analysis.react_report_agent import ReportAgent
            agent = ReportAgent(
                api_key=api_key, base_url=base_url, model=model,
                index_path=str(_TRACKER_INDEX),
            )
            text = agent.generate_report(query)
            return jsonify({"query": query, "report": text})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # ── Scan incomplete runs ──────────────────────────────────────────────────

    @app.get("/incomplete")
    def incomplete():
        """List all experiments with status != complete (resumable runs)."""
        try:
            from simulation.crash_recovery import scan_incomplete_runs
            runs = scan_incomplete_runs(str(_EXPERIMENTS_ROOT))
            return jsonify({"incomplete_runs": runs})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


# ── Dev server entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BGF REST API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    app = create_app(experiments_root=args.experiments_root)
    app.run(host=args.host, port=args.port, debug=args.debug)
