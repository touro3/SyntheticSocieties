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
  POST   /anchor/<exp_id>          Query the omniscient anchor agent (macro view)
  POST   /inject/<exp_id>          Inject a live exogenous event via IPC
  GET    /report                   Run ReACT agent on a query (sync, slow)
  GET    /health                   Liveness probe
  GET    /incomplete               List resumable (non-complete) runs
  GET    /human-eval/scenarios     Return vignette pairs for Prolific study
  POST   /human-eval/rating        Save a participant rating
  GET    /human-eval/results       Aggregate results (admin, requires auth)
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

# Load .env from repo root before anything else reads os.environ
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

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

_CONFIGS_ROOT = Path("configs")
_STATIC_DIR = Path(__file__).parent / "static"

# ── Writable filesystem roots ────────────────────────────────────────────────
# When BGF_DATA_ROOT is set (e.g. HF Space with persistent storage mounted
# at /data) all four user-generated content paths are rerooted under it so a
# container restart does not wipe experiments, uploads, the tracker index, or
# human-eval responses. When unset, paths match the historical layout under
# the repository root, so local development is unaffected.
_BGF_DATA_ROOT_ENV = os.environ.get("BGF_DATA_ROOT", "").strip()
if _BGF_DATA_ROOT_ENV:
    _DATA_ROOT = Path(_BGF_DATA_ROOT_ENV).resolve()
    _EXPERIMENTS_ROOT = _DATA_ROOT / "experiments"
    _TRACKER_INDEX = _DATA_ROOT / "tracker" / "experiment_index.parquet"
    _UPLOADS_DIR = _DATA_ROOT / "uploads" / "ess_data"
    _HUMAN_OUTPUTS_DIR = _DATA_ROOT / "human_outputs"
else:
    _DATA_ROOT = Path(".").resolve()
    _EXPERIMENTS_ROOT = Path("experiments")
    _TRACKER_INDEX = Path("tracker/experiment_index.parquet")
    _UPLOADS_DIR = Path("uploads") / "ess_data"
    _HUMAN_OUTPUTS_DIR = Path("data/human")

# Bearer-token auth.  Set BGF_API_TOKEN env var to enable.
# If unset, auth is disabled — intended for local / trusted-network use only.
_AUTH_TOKEN: str | None = os.environ.get("BGF_API_TOKEN")

# Public POST endpoints: write/compute routes that stay open even when a token
# is configured, because they are part of the public demo UX and would
# otherwise require shipping the token in client-side JS (where it is trivially
# readable, defeating the purpose). Cost/DoS exposure on these is bounded
# instead by the per-IP rate limiter (see `_simulate_limit`), not by the token.
#
# /human-eval/rating and /human-game/* are public participant/demo flows (Prolific
# study + interactive baseline game): visitors have no token, so token-gating
# would break the study. They are rate-limited and resource-bounded instead.
#
# /simulate-wizard is the public demo's "Launch Simulation" path. It spawns a
# subprocess simulation, so it is the heaviest public endpoint — its cost/DoS
# exposure is bounded by a dedicated stricter limiter (`_wizard_limit`) plus the
# per-run size caps in configs/schema.py, not by the token.
_PUBLIC_POST_PATHS: frozenset[str] = frozenset(
    {
        "/design-simulation",
        "/simulate-wizard",
        "/human-eval/rating",
        "/human-game/session",
        "/human-game/action",
        "/human-game/complete",
    }
)

# Public POST routes with dynamic path segments (exp_id / agent_id), so they
# cannot be matched exactly: the Interact page's "Send" (agent interview),
# "Broadcast" (anchor query), and "Inject Event" buttons. All are rate-limited
# and operate only on existing experiments. The trailing slash is required so
# the prefix cannot match an unintended sibling route.
_PUBLIC_POST_PREFIXES: tuple[str, ...] = ("/interview/", "/anchor/", "/inject/")

# Provider API-key token shapes (OpenAI sk-/sk-proj-, Groq gsk_, HF hf_).
_SECRET_TOKEN_RE = re.compile(r"\b(sk-[A-Za-z0-9_-]{8,}|gsk_[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{8,})")


def _scrub_secrets(text: str | None) -> str:
    """Mask credentials before they reach logs / status files.

    Replaces the live values of known secret env vars and any provider
    token-shaped substrings, so a chatty/erroring child process cannot leak
    a key into server logs or the publicly-readable run status file.
    """
    if not text:
        return text or ""
    out = text
    for _var in ("OPENAI_API_KEY", "GROQ_API_KEY", "BGF_API_TOKEN"):
        _val = os.environ.get(_var, "")
        if _val and len(_val) >= 8:
            out = out.replace(_val, "***REDACTED***")
    return _SECRET_TOKEN_RE.sub("***REDACTED***", out)


# Valid experiment ID pattern: alphanumeric, underscores, hyphens, dots only.
# Blocks path-traversal sequences like "../", "%2e%2e", etc.
_EXP_ID_RE = re.compile(r"^[A-Za-z0-9_\-\.]{1,128}$")

# ── Design-LLM response cache ─────────────────────────────────────────────
# Keyed by sha256(provider + user_msg).  Identical prompts (same scenario
# description) return the cached design without burning an API credit.
_DESIGN_CACHE: dict[str, tuple] = {}
_DESIGN_CACHE_MAX = 128  # ~128 scenario designs in RAM before FIFO eviction
_DESIGN_CACHE_LOCK = threading.Lock()  # guards the check-evict-insert sequence

# ── OpenAI client singleton ───────────────────────────────────────────────
# Re-using one client per api-key prefix avoids TCP handshake overhead on
# every /interview call (Render's free tier has limited open-file descriptors).
# FIFO-bounded so a long-running server seeing many distinct API keys (multi-
# tenant deploys) cannot leak client objects.
_OAI_CLIENTS: dict[str, Any] = {}
_OAI_CLIENTS_MAX = 16
_OAI_CLIENTS_LOCK = threading.Lock()  # guards check-evict-insert sequence


def _oai_clients_put(prefix: str, client: Any) -> None:
    """Insert into _OAI_CLIENTS, evicting oldest if over cap. Caller holds lock."""
    _OAI_CLIENTS[prefix] = client
    while len(_OAI_CLIENTS) > _OAI_CLIENTS_MAX:
        _OAI_CLIENTS.pop(next(iter(_OAI_CLIENTS)))


def _oai_clients_get_or_create(prefix: str, factory):
    """Atomic check-then-insert. factory() is called at most once per prefix
    even under concurrent access."""
    with _OAI_CLIENTS_LOCK:
        client = _OAI_CLIENTS.get(prefix)
        if client is None:
            client = factory()
            _oai_clients_put(prefix, client)
        return client


# ── Ollama warmup throttle ────────────────────────────────────────────────
# Tracks the last time we kicked off a background model-load so we don't
# spam Ollama on every /api/capabilities poll.
_OLLAMA_WARMUP_TS: float = 0.0
_OLLAMA_WARMUP_LOCK = threading.Lock()

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

# ── events.jsonl byte-offset cache ────────────────────────────────────────────
# Avoids O(n) full-file scans on every /interview and /anchor request.
# Each entry: (byte_offset, events_list). On the next read for the same exp_id,
# we seek to byte_offset and only parse new lines appended since last read.
# Capped at 32 experiments; FIFO eviction prevents unbounded growth.
# A threading.Lock serializes concurrent reads on the same exp_id so two
# simultaneous requests don't both parse the same new lines.
_EVENTS_CACHE: dict[str, tuple[int, list[dict]]] = {}
_EVENTS_CACHE_MAX = 32
_EVENTS_CACHE_LOCK = threading.Lock()


def _append_interview_response(
    exp_dir: Path,
    agent_id: str,
    question: str,
    response: str,
    extra: dict | None = None,
) -> None:
    """Append a Q&A pair to interview_responses.jsonl so the anchor can tally opinions.

    Extra fields (persona summary, memory provenance, history window) get
    folded into the record so downstream consumers can filter or re-weight
    answers without re-deriving the context.
    """
    import json as _json
    import time as _time

    record = {
        "agent_id": agent_id,
        "question": question,
        "response": response,
        "ts": _time.time(),
    }
    if extra:
        record.update(extra)
    try:
        with (exp_dir / "interview_responses.jsonl").open("a") as fh:
            fh.write(_json.dumps(record) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass  # non-critical; anchor works without it


def _load_population_snapshot(exp_dir: Path) -> dict[str, dict]:
    """Load population_snapshot.jsonl into an {agent_id → profile} mapping.

    Returns an empty dict when the snapshot is missing (older runs predating
    the snapshot write or runs where it failed to serialize).
    """
    import json as _json

    snap = exp_dir / "population_snapshot.jsonl"
    if not snap.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        with snap.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except Exception:
                    continue
                aid = rec.get("agent_id")
                if aid:
                    out[str(aid)] = rec
    except OSError:
        return {}
    return out


# ── Anchor stance extraction (semantic clustering) ─────────────────────────
# Cache keyed by sha256(text+options) so re-running the same anchor question
# does not re-burn OpenAI tokens. Bounded LRU to keep memory predictable.
import hashlib as _hashlib  # noqa: E402 — keeps the import next to the cache it owns
from collections import OrderedDict as _OrderedDict  # noqa: E402

_STANCE_CACHE: _OrderedDict[str, dict] = _OrderedDict()
_STANCE_CACHE_MAX = 1024


def _extract_stance_via_llm(
    text: str,
    question: str,
    options: list[str],
    *,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> dict | None:
    """Use gpt-4o-mini to extract a structured stance from a single response.

    Returns ``{"stance", "rationale", "topic_tags"}`` or None on failure.
    Replaces the historical substring-tally that double-counted agents whose
    response mentioned both option words and missed nuance ("paper" appearing
    in a negative clause still counted as a vote for paper).
    """
    if not text or not options:
        return None

    cache_key = _hashlib.sha256((text + "||" + question + "||" + "|".join(options)).encode("utf-8")).hexdigest()
    cached = _STANCE_CACHE.get(cache_key)
    if cached is not None:
        _STANCE_CACHE.move_to_end(cache_key)
        return cached

    try:
        from openai import OpenAI as _OAI

        oai = _oai_clients_get_or_create(api_key[:8], lambda: _OAI(api_key=api_key))
        system_prompt = (
            "You are extracting a structured stance from a single agent's text. "
            "Return JSON with exactly these keys: "
            "`stance` (one of the provided options or null if the text expresses "
            "no preference), `rationale` (one-sentence summary of why), "
            "`topic_tags` (array of 1-3 short topic tags). "
            "Do not infer beyond what is in the text."
        )
        user_prompt = (
            f"Question put to the agent: {question}\nAllowed stance options: {options}\nAgent text:\n{text[:1200]}"
        )
        resp = oai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=180,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return None
        # Normalize: stance must be one of options (case-insensitive) or null.
        stance = parsed.get("stance")
        if stance is not None and isinstance(stance, str):
            stance_low = stance.strip().lower()
            normalized = next((o for o in options if o.lower() == stance_low), None)
            parsed["stance"] = normalized
        else:
            parsed["stance"] = None
        _STANCE_CACHE[cache_key] = parsed
        if len(_STANCE_CACHE) > _STANCE_CACHE_MAX:
            _STANCE_CACHE.popitem(last=False)
        return parsed
    except Exception as exc:  # noqa: BLE001 — graceful degradation
        logger.warning("Stance extraction failed: %s", exc)
        return None


def _semantic_stance_tally(
    texts_by_agent: dict[str, str],
    question: str,
    options: list[str],
) -> tuple[dict, dict, list[dict]] | None:
    """Aggregate per-agent stances via LLM extraction.

    Returns ``(counts, by_agent, per_agent_details)`` or None if no API key
    is available (caller falls back to substring counting).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    counts: dict[str, int] = {}
    by_agent: dict[str, list[str]] = {}
    details: list[dict] = []
    for ag, txt in texts_by_agent.items():
        result = _extract_stance_via_llm(txt, question, options, api_key=api_key)
        if not result:
            continue
        stance = result.get("stance")
        details.append({"agent_id": ag, **result})
        if stance:
            counts[stance] = counts.get(stance, 0) + 1
            by_agent.setdefault(stance, []).append(ag)
    return counts, by_agent, details


def _synthesize_live_interview_answer(
    *,
    exp_id: str,
    exp_dir: Path,
    agent_id: str,
    question: str,
    live_context: dict,
) -> dict | None:
    """LLM-synthesize an interview answer for a still-running agent.

    Combines the live state/memory reflection delivered by the IPC server
    with the same persona/scenario context the replay path uses. Returns
    None when OpenAI is unreachable so the caller can fall back to the
    static IPC answer.
    """
    import time as _time

    profile = live_context.get("profile") or {}
    state = live_context.get("state") or {}
    memory_reflection = (live_context.get("memory_reflection") or "").strip()
    recent_events = live_context.get("recent_events") or []

    # Build a faux events list so we can reuse the replay-path helper.
    faux_events = []
    for item in recent_events:
        faux_events.append(
            {
                "round_id": item.get("round_id"),
                "action": {
                    "action_type": item.get("event_type"),
                    "target_agent_id": item.get("partner_id"),
                    "reasoning_summary": "",
                },
                "state_after": state,
                "perception": {"network": {"neighbors": []}},
            }
        )

    # If we know the agent profile but no population snapshot exists yet
    # (snapshot is written at run start, so it should exist), seed one in
    # memory by writing a stub to a temp dict and constructing the context
    # manually.
    snapshot = _load_population_snapshot(exp_dir)
    if profile and not snapshot.get(agent_id):
        # Write the live profile snapshot opportunistically so subsequent
        # replays inherit the persona block too.
        try:
            snap_path = exp_dir / "population_snapshot.jsonl"
            with snap_path.open("a", encoding="utf-8") as fh:
                rec = {"agent_id": agent_id, **{k: v for k, v in profile.items() if k != "agent_id"}}
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass

    ctx = _interview_prompt_context(
        exp_dir,
        agent_id,
        faux_events,
        live_memory_reflection=memory_reflection or None,
    )

    oai_key = os.environ.get("OPENAI_API_KEY", "")
    if not oai_key:
        return None

    try:
        from openai import OpenAI as _OAI
        from openai import RateLimitError as _RLE

        oai = _oai_clients_get_or_create(oai_key[:8], lambda: _OAI(api_key=oai_key))

        blocks = [
            (
                f"You are {agent_id}, a synthetic agent partway through a live BGF "
                f"simulation ({exp_id}). The simulation is still running; describe "
                f"your perspective in the present tense.\n{ctx['persona_block']}"
            ),
            ctx["scenario_block"],
            (
                f"Live state: wealth {state.get('wealth')}, stress {state.get('stress')}, "
                f"satisfaction {state.get('satisfaction')}."
            ),
            ctx["memory_block"],
            ctx["social_block"],
            (
                "Answer the user's question in first-person, drawing on your "
                "demographics, dispositions, and recent memory. Be natural and "
                "concise (2-4 sentences). Do not mention simulation internals."
            ),
        ]
        system_prompt = "\n\n".join(b for b in blocks if b)

        resp = None
        for att in range(3):
            try:
                resp = oai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    max_tokens=220,
                    temperature=0.7,
                )
                break
            except _RLE:
                _time.sleep(2**att)
        if resp is None:
            return None
        answer = resp.choices[0].message.content.strip()
        return {
            "response": answer,
            "persona_used": bool(ctx["persona_block"]),
            "history_window": len(faux_events),
            "memory_source": "live",
            "model": "gpt-4o-mini",
            "source": "ipc_live_llm",
        }
    except Exception as exc:  # noqa: BLE001 — graceful degradation
        logger.warning("Live IPC LLM synthesis failed (exp=%s agent=%s): %s", exp_id, agent_id, exc)
        return None


def _interview_prompt_context(
    exp_dir: Path,
    agent_id: str,
    agent_events: list[dict],
    *,
    live_memory_reflection: str | None = None,
) -> dict:
    """Build the structured context dict for an /interview LLM call.

    The helper is used by both the data-replay path (events on disk) and the
    live IPC path (running simulation) so live and replayed interviews share
    the same prompt structure and answer quality, rather than the live path
    being a thin memory dump as it was historically.
    """

    # ── Persona block ─────────────────────────────────────────────────────
    snapshot = _load_population_snapshot(exp_dir)
    profile = snapshot.get(agent_id, {})

    def _fmt(value, fmt: str = "{}") -> str:
        if value is None or value == "":
            return ""
        try:
            return fmt.format(value)
        except Exception:
            return str(value)

    persona_lines: list[str] = []
    demo_bits: list[str] = []
    if profile.get("age") is not None:
        demo_bits.append(_fmt(profile.get("age"), "{:.0f}-year-old"))
    if profile.get("gender") is not None:
        _g = {1: "man", 2: "woman"}.get(int(profile["gender"]) if str(profile["gender"]).isdigit() else -1)
        if _g:
            demo_bits.append(_g)
    if profile.get("country"):
        demo_bits.append(f"from {profile['country']}")
    if demo_bits:
        persona_lines.append("Demographics: " + ", ".join(demo_bits) + ".")

    def _band(name: str, val) -> str | None:
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        if v >= 0.65:
            return f"high {name}"
        if v <= 0.35:
            return f"low {name}"
        return f"moderate {name}"

    trait_bands = [
        _band("interpersonal trust", profile.get("trust_people")),
        _band("institutional trust", profile.get("trust_institutions")),
        _band("risk tolerance", profile.get("risk_tolerance")),
        _band("competitiveness", profile.get("competitiveness")),
        _band("life satisfaction", profile.get("life_satisfaction")),
    ]
    trait_bands = [t for t in trait_bands if t]
    if trait_bands:
        persona_lines.append("Dispositions: " + "; ".join(trait_bands) + ".")
    if profile.get("left_right") is not None or profile.get("political_orientation") is not None:
        pol = profile.get("political_orientation") or profile.get("left_right")
        try:
            pol_v = float(pol)
            pol_label = "left-leaning" if pol_v < 0.4 else "right-leaning" if pol_v > 0.6 else "centrist"
            persona_lines.append(f"Political orientation: {pol_label} ({pol_v:.2f}/1.0).")
        except (TypeError, ValueError):
            pass
    if profile.get("is_adversarial"):
        persona_lines.append("Behavioral type: adversarial (constrained to predatory actions).")

    persona_block = "\n".join(persona_lines) if persona_lines else ""

    # ── Scenario block (always rendered, falls back to config-derived) ────
    scenario_json = _safe_json_file(exp_dir / "scenario.json") or {}
    config_yaml: dict = {}
    try:
        import yaml as _yaml  # noqa: WPS433 — used opportunistically

        config_path = exp_dir / "config.yaml"
        if config_path.exists():
            config_yaml = _yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        config_yaml = {}

    if scenario_json.get("scenario_title"):
        scenario_block = (
            f"Scenario: {scenario_json['scenario_title']}\n"
            f"Context: {scenario_json.get('scenario_description', '')}\n"
            f"Population backdrop: {scenario_json.get('population_narrative', '')}"
        ).strip()
    else:
        sim = config_yaml.get("simulation") or {}
        net = config_yaml.get("network") or {}
        pop = config_yaml.get("population") or {}
        data = config_yaml.get("data") or {}
        scenario_block = (
            f"Scenario: a {sim.get('rounds', '?')}-round economic simulation with "
            f"{sim.get('population_size', '?')} agents, "
            f"policy={config_yaml.get('policy', {}).get('type', '?')}, "
            f"network={net.get('type', '?')}, "
            f"population_source={pop.get('source', '?')}."
        )
        if data.get("population_context"):
            scenario_block += f"\nPopulation backdrop: {data['population_context']}"

    # ── Adaptive history window ──────────────────────────────────────────
    total_rounds = len(agent_events)
    window = min(30, max(10, total_rounds // 3 or 10))
    history_events = agent_events[-window:]

    history_lines: list[str] = []
    for ev in history_events:
        r = ev.get("round_id", "?")
        act = ev.get("action", {}).get("action_type", "?")
        w = ev.get("state_after", {}).get("wealth")
        s = ev.get("state_after", {}).get("stress")
        rsn = ev.get("action", {}).get("reasoning_summary", "") or ""
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

    # ── Full-run social graph (not just last 10) ──────────────────────────
    all_neighbors: set[str] = set()
    coop_targets: dict[str, int] = {}
    steal_targets: dict[str, int] = {}
    for ev in agent_events:
        for nb in ev.get("perception", {}).get("network", {}).get("neighbors", []) or []:
            all_neighbors.add(str(nb))
        act_type = ev.get("action", {}).get("action_type", "")
        tgt = ev.get("action", {}).get("target_agent_id")
        if tgt:
            if act_type == "cooperate":
                coop_targets[str(tgt)] = coop_targets.get(str(tgt), 0) + 1
            elif act_type == "steal":
                steal_targets[str(tgt)] = steal_targets.get(str(tgt), 0) + 1

    social_lines: list[str] = []
    if coop_targets:
        top_coop = sorted(coop_targets.items(), key=lambda kv: -kv[1])[:3]
        social_lines.append("Repeated cooperation partners: " + ", ".join(f"{aid} (×{n})" for aid, n in top_coop))
    if steal_targets:
        top_steal = sorted(steal_targets.items(), key=lambda kv: -kv[1])[:3]
        social_lines.append("Stole from: " + ", ".join(f"{aid} (×{n})" for aid, n in top_steal))
    if all_neighbors and not coop_targets:
        social_lines.append("Neighbors never cooperated with: " + ", ".join(sorted(all_neighbors)[:4]))
    social_block = "\n".join(social_lines)

    # ── Memory block ──────────────────────────────────────────────────────
    memory_block = ""
    memory_source = "events_only"
    if live_memory_reflection:
        memory_block = f"Recent memory reflection: {live_memory_reflection.strip()}"
        memory_source = "live"

    return {
        "persona": profile,
        "persona_block": persona_block,
        "scenario_block": scenario_block,
        "history_text": history_text,
        "history_window": window,
        "social_block": social_block,
        "memory_block": memory_block,
        "memory_source": memory_source,
        "all_neighbors": all_neighbors,
        "coop_targets": coop_targets,
        "steal_targets": steal_targets,
    }


def _read_events_cached(events_path: Path) -> list[dict]:
    """Return all parsed events from events_path using a byte-offset cache.

    On first call: reads the whole file, caches (offset, events).
    On subsequent calls: seeks to the cached offset, parses only new lines,
    appends to the cached list. O(new_lines) instead of O(total_lines).

    Thread-safe: a single lock serializes all readers so two concurrent
    requests don't parse the same new lines twice.
    """
    import json as _json

    cache_key = str(events_path)
    with _EVENTS_CACHE_LOCK:
        offset, cached_events = _EVENTS_CACHE.get(cache_key, (0, []))
        try:
            new_events: list[dict] = []
            with events_path.open("rb") as fh:
                fh.seek(offset)
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        new_events.append(_json.loads(line))
                    except _json.JSONDecodeError:
                        continue
                new_offset = fh.tell()
        except OSError:
            return list(cached_events)

        all_events = cached_events + new_events

        # Evict oldest entry when cache is full
        if cache_key not in _EVENTS_CACHE and len(_EVENTS_CACHE) >= _EVENTS_CACHE_MAX:
            _EVENTS_CACHE.pop(next(iter(_EVENTS_CACHE)))
        _EVENTS_CACHE[cache_key] = (new_offset, all_events)
        return all_events


# ── AI simulation design ──────────────────────────────────────────────────────

_DESIGN_SYSTEM_PROMPT = """You are a population designer for BGF (Behavioral Grounding Framework), \
an agent-based economic simulation where agents choose work/save/cooperate/steal each round.

Analyze the user's simulation description and return a JSON object with EXACTLY these fields:
{
  "scenario_title": "short descriptive title (≤8 words)",
  "scenario_description": "2-3 sentence context for what this simulation models",
  "config": {
    "agents": <integer 5-200>,
    "rounds": <integer 5-100>,
    "policy": <"rule_based" | "llm" | "random">,
    "network_type": <"random" | "small_world">,
    "bad_apple_frac": <float 0.0-0.3>
  },
  "population_traits": {
    "trust_institutions": <float 0.0-1.0>,
    "trust_people": <float 0.0-1.0>,
    "risk_taking": <float 0.0-1.0>,
    "life_satisfaction": <float 0.0-1.0>,
    "competitiveness": <float 0.0-1.0>,
    "income_decile_avg": <float 1.0-10.0>,
    "age_mean": <float 18-80>,
    "urbanization": <float 0.0-1.0>
  },
  "population_narrative": "2-4 sentences injected into agent decision prompts. Describe trust levels, economic tendencies, social structure, and cultural context so agents behave realistically.",
  "reasoning": "1-2 sentences explaining key parameter choices"
}

Guidelines:
- small_world network when community/tribe/neighborhood is implied; random otherwise
- rule_based policy is fastest and realistic; use llm only if deep reasoning is requested
- increase bad_apple_frac for scenarios with corruption/fraud/adversarial dynamics
- high income → income_decile_avg 7-9; working class → 3-5; mixed → 4-6
- population_narrative is the most important field — make it specific and behaviorally rich

Return ONLY valid JSON, no markdown fences."""


def _ollama_available() -> bool:
    """Return True if a local Ollama server is reachable."""
    try:
        import urllib.request as _ur

        with _ur.urlopen("http://localhost:11434/api/tags", timeout=1) as _r:
            return _r.status == 200
    except Exception:
        return False


def _maybe_warmup_ollama(model: str = "llama3.2") -> None:
    """Fire a background thread to pre-load the Ollama model if it isn't already
    warm.  Throttled to at most once every 4 minutes so repeated /api/capabilities
    polls don't pile up.  The warmup completes silently; any error is ignored."""
    global _OLLAMA_WARMUP_TS
    import time as _time

    with _OLLAMA_WARMUP_LOCK:
        now = _time.time()
        if now - _OLLAMA_WARMUP_TS < 240:
            return
        _OLLAMA_WARMUP_TS = now

    def _run():
        try:
            import json as _j
            import urllib.request as _ur2

            payload = _j.dumps(
                {"model": model, "messages": [{"role": "user", "content": ""}], "max_tokens": 1}
            ).encode()
            req = _ur2.Request(
                "http://localhost:11434/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with _ur2.urlopen(req, timeout=120):
                pass
            logger.debug("Ollama model '%s' warm-up complete", model)
        except Exception as exc:
            logger.debug("Ollama warm-up skipped: %s", exc)

    threading.Thread(target=_run, daemon=True).start()


def _call_design_llm(provider: str, api_key: str, user_msg: str, ollama_model: str = "llama3.2") -> tuple[dict, dict]:
    """Call OpenAI, Groq, or Ollama to generate a simulation design from a prompt.

    Returns:
        (design_dict, meta) where meta contains usage, resolved_model, prompt_hash, temperature.
    """
    import hashlib
    import time as _time

    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("openai package not installed — pip install openai") from exc

    # Cache check — skip API call for identical prompts (same scenario wording)
    cache_key = hashlib.sha256(f"{provider}:{user_msg}".encode()).hexdigest()
    with _DESIGN_CACHE_LOCK:
        if cache_key in _DESIGN_CACHE:
            logger.info("design LLM: cache hit %s", cache_key)
            cached_design, cached_meta = _DESIGN_CACHE[cache_key]
            return cached_design, {**cached_meta, "cache_hit": True}

    _TEMPERATURE = 0.4
    use_json_format = True
    if provider == "groq":
        client = openai.OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        model = "llama-3.3-70b-versatile"
    elif provider == "ollama":
        client = openai.OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
        model = ollama_model
        # Older Ollama builds don't support response_format — skip it
        use_json_format = False
    else:
        # Reuse singleton to avoid TCP reconnect on every call
        key_prefix = api_key[:8]
        client = _oai_clients_get_or_create(key_prefix, lambda: openai.OpenAI(api_key=api_key))
        model = "gpt-4o-mini"

    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": _DESIGN_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=_TEMPERATURE,
        max_tokens=1000,
    )
    if use_json_format:
        kwargs["response_format"] = {"type": "json_object"}

    # Retry on 429 rate-limit with exponential backoff (max 3 attempts)
    _last_exc: Exception | None = None
    for _attempt in range(3):
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except openai.RateLimitError as exc:
            _last_exc = exc
            _wait = 2**_attempt
            logger.warning("design LLM 429 (attempt %d) — retrying in %ds", _attempt + 1, _wait)
            _time.sleep(_wait)
    else:
        raise _last_exc  # type: ignore[misc]

    content = resp.choices[0].message.content
    # Strip markdown fences Ollama sometimes adds
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    usage: dict = {}
    if resp.usage:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }

    meta = {
        "provider": provider,
        "resolved_model": resp.model,
        "temperature": _TEMPERATURE,
        "prompt_hash": hashlib.sha256(_DESIGN_SYSTEM_PROMPT.encode()).hexdigest()[:8],
        "usage": usage,
        "cache_hit": False,
    }
    logger.info(
        "design LLM call: provider=%s model=%s tokens=%s",
        provider,
        resp.model,
        usage.get("total_tokens", "?"),
    )
    result = json.loads(content.strip())

    # Store in cache; FIFO eviction when full
    with _DESIGN_CACHE_LOCK:
        if len(_DESIGN_CACHE) >= _DESIGN_CACHE_MAX:
            _DESIGN_CACHE.pop(next(iter(_DESIGN_CACHE)))
        _DESIGN_CACHE[cache_key] = (result, meta)

    return result, meta


def _generate_population_parquet(design: dict, n_agents: int) -> str | None:
    """Synthesise a population parquet from AI-designed trait distributions."""
    try:
        import numpy as np
        import pandas as pd

        traits = design.get("population_traits", {})
        rng = np.random.default_rng(42)

        def _s(mean_val: float, std: float = 0.15, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
            return np.clip(rng.normal(mean_val, std, n_agents), lo, hi).astype(np.float32)

        df = pd.DataFrame(
            {
                "trust_institutions": _s(traits.get("trust_institutions", 0.5)),
                "trust_people": _s(traits.get("trust_people", 0.5)),
                "risk_taking": _s(traits.get("risk_taking", 0.5)),
                "life_satisfaction": _s(traits.get("life_satisfaction", 0.6)),
                "competitiveness": _s(traits.get("competitiveness", 0.5)),
                "income_decile": _s(traits.get("income_decile_avg", 5.0), std=2.0, lo=1.0, hi=10.0),
                "age": _s(traits.get("age_mean", 40.0), std=12.0, lo=18.0, hi=85.0).astype(np.int32),
                "urbanization": _s(traits.get("urbanization", 0.6)),
                "gender": rng.choice([1, 2], size=n_agents).astype(np.int32),
            }
        )

        _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        import uuid as _uuid2

        file_id = str(_uuid2.uuid4())
        out_path = _UPLOADS_DIR / f"{file_id}.parquet"
        df.to_parquet(out_path, index=False)

        # Analysis sidecar so simulate_wizard can load the narrative
        title = design.get("scenario_title", "AI-generated population")
        analysis = {
            "filename": f"ai_{title.lower().replace(' ', '_')}.parquet",
            "narrative": design.get("population_narrative", ""),
            "generated": True,
        }
        (_UPLOADS_DIR / f"{file_id}_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False))
        return file_id
    except Exception as exc:
        logger.error("Population parquet generation failed: %s", exc)
        return None


# ── Security helpers ──────────────────────────────────────────────────────────


def _check_auth() -> tuple[bool, Any]:
    """Verify bearer token.  Returns (ok, error_response_or_None)."""
    if not _AUTH_TOKEN:
        return True, None  # Auth disabled — open mode (no token configured)

    # Public reads: when a token IS configured, safe/side-effect-free methods
    # (GET/HEAD/OPTIONS) stay open so the Space works as a public demo. Only
    # write/compute methods (POST/PUT/PATCH/DELETE) — /simulate, /inject,
    # /upload-ess-data, /design-simulation, /interview, /anchor, etc. —
    # require the bearer token, which is what closes the DoS/cost exposure.
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return True, None

    # Public demo POSTs (e.g. /design-simulation): rate-limited, not token-gated.
    if request.path in _PUBLIC_POST_PATHS or request.path.startswith(_PUBLIC_POST_PREFIXES):
        return True, None

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False, (jsonify({"error": "Authorization header required"}), 401)

    provided = auth_header[len("Bearer ") :]
    # hmac.compare_digest prevents timing-oracle attacks.
    if not hmac.compare_digest(provided.encode(), _AUTH_TOKEN.encode()):
        logger.warning("Failed auth attempt from %s", request.remote_addr)
        return False, (jsonify({"error": "Invalid token"}), 403)

    return True, None


def _is_trusted_caller() -> bool:
    """True if the caller may supply privileged inputs (e.g. an LLM API key).

    Open mode (no token configured) is trusted local/LAN use. When a token IS
    configured, only a request presenting the valid bearer token is trusted —
    a public, tokenless caller hitting a `_PUBLIC_POST_PATHS` route is NOT, so
    its client-supplied API keys are ignored in favour of server-env keys
    only. This prevents anonymous users from injecting arbitrary provider
    credentials through the public demo endpoints.
    """
    if not _AUTH_TOKEN:
        return True
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    provided = auth_header[len("Bearer ") :]
    return hmac.compare_digest(provided.encode(), _AUTH_TOKEN.encode())


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


def _load_uploaded_dataframe(filename: str, raw: bytes):
    """Decode an uploaded population file into a pandas DataFrame.

    Handles CSV (with delimiter sniffing), Parquet, SPSS ``.sav`` (ESS-native),
    and Stata ``.dta``. Returns ``(df, error_message)`` — exactly one is None.
    Reading is best-effort; any decoder exception is converted into a clear
    user-facing message rather than a 500.
    """
    import csv as _csv
    import io as _io

    import pandas as _pd

    fname = filename.lower()
    buf = _io.BytesIO(raw)

    try:
        if fname.endswith(".csv"):
            # Sniff delimiter from a head sample — European exports often use
            # ';' or tab, and silently misparsing them is worse than failing.
            sample = raw[:8192].decode("utf-8", errors="replace")
            sep = ","
            try:
                dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
                sep = dialect.delimiter
            except _csv.Error:
                pass
            df = _pd.read_csv(buf, sep=sep)
        elif fname.endswith(".parquet"):
            df = _pd.read_parquet(buf)
        elif fname.endswith(".sav"):
            try:
                import pyreadstat  # noqa: WPS433 — optional dep, surfaced below
            except ImportError:
                return None, (
                    "SPSS .sav support requires the 'pyreadstat' package. "
                    "On the server, install it via `pip install pyreadstat` and redeploy."
                )
            # pyreadstat needs a filesystem path — write to a temp file.
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            try:
                df, _meta = pyreadstat.read_sav(tmp_path, apply_value_formats=False)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        elif fname.endswith(".dta"):
            df = _pd.read_stata(buf, convert_categoricals=False)
        else:
            return None, f"Unsupported file extension: {fname}"
    except Exception as exc:  # noqa: BLE001 — surface the parser error verbatim
        return None, f"Could not parse file: {exc}"

    return df, None


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
    # Honour the `experiments_root` arg only when it differs from the default,
    # so that BGF_DATA_ROOT-rerouted paths win over the historical default.
    if experiments_root != "experiments":
        _EXPERIMENTS_ROOT = Path(experiments_root)
    _CONFIGS_ROOT = Path(configs_root)
    _EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    _TRACKER_INDEX.parent.mkdir(parents=True, exist_ok=True)
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    _HUMAN_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__, template_folder=Path(__file__).parent / "templates", static_folder=None)

    # Allow up to 52 MB to accommodate ESS data file uploads (max 50 MB).
    # JSON-only endpoints send small payloads; the upload endpoint enforces
    # its own 50 MB cap after reading.
    app.config["MAX_CONTENT_LENGTH"] = 52 * 1024 * 1024  # 52 MB

    # CORS origin resolution, in priority order:
    #   1. `allowed_origins` arg (programmatic override, e.g. tests).
    #   2. BGF_CORS_ORIGINS env var (comma-separated list).
    #   3. Auto-derived HF Space origin from SPACE_ID + localhost dev origins.
    # This avoids hardcoding a specific username so the repo is portable; HF
    # Spaces always sets SPACE_ID=<user>/<space>, which maps to
    # https://<user>-<space>.hf.space.
    _LOCALHOST_ORIGINS = [
        "http://localhost:5050",
        "http://localhost:5173",
        "http://127.0.0.1:5050",
        "http://127.0.0.1:5173",
    ]
    _hf_space_id = os.environ.get("SPACE_ID", "").strip()
    _hf_origin: list[str] = []
    if _hf_space_id and "/" in _hf_space_id:
        _hf_user, _hf_name = _hf_space_id.split("/", 1)
        _hf_origin = [f"https://{_hf_user}-{_hf_name}.hf.space".lower()]
    _env_origins = os.environ.get("BGF_CORS_ORIGINS", "").strip()
    _resolved_origins = (
        allowed_origins
        or ([o.strip() for o in _env_origins.split(",") if o.strip()] if _env_origins else None)
        or (_hf_origin + _LOCALHOST_ORIGINS)
    )
    CORS(app, origins=_resolved_origins)

    # ── Rate limiting (optional — requires flask-limiter) ────────────────────
    if _LIMITER_AVAILABLE:
        limiter = Limiter(
            get_remote_address,
            app=app,
            default_limits=["300 per hour", "60 per minute"],
            storage_uri="memory://",
        )
        _simulate_limit = limiter.limit("10 per minute")
        # Stricter cap for the public, unauthenticated, paid-LLM design endpoint:
        # bounds Groq/OpenAI cost exposure since it is no longer token-gated.
        _design_limit = limiter.limit("5 per minute; 40 per hour; 150 per day")
        # Heaviest public endpoint: each call spawns a subprocess simulation.
        # Tighter than _simulate_limit since it is no longer token-gated.
        _wizard_limit = limiter.limit("3 per minute; 20 per hour; 80 per day")
        _report_limit = limiter.limit("5 per minute")
        _write_limit = limiter.limit("30 per minute")
        _read_limit = limiter.limit("60 per minute")
        _experiments_limit = limiter.limit("30 per minute")
        _incomplete_limit = limiter.limit("10 per minute")
        # Public, unauthenticated write/compute demo endpoints — bound disk-write
        # and memory-growth abuse since they are not token-gated.
        _human_eval_write_limit = limiter.limit("10 per minute; 100 per hour")
        _human_game_limit = limiter.limit("30 per minute; 300 per hour")
    else:
        # No-op decorators if flask-limiter is not installed.
        def _noop(f):
            return f

        _simulate_limit = _report_limit = _write_limit = _noop
        _design_limit = _read_limit = _experiments_limit = _incomplete_limit = _noop
        _wizard_limit = _human_eval_write_limit = _human_game_limit = _noop
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

    # ── SPA fallback — serve index.html for all non-API routes (web history mode) ──

    @app.get("/<path:path>")
    def spa_fallback(path: str):
        from flask import send_from_directory

        _api_prefixes = (
            "health",
            "ping",
            "simulate",
            "status",
            "results",
            "experiments",
            "interview",
            "inject",
            "report",
            "incomplete",
            "human-eval",
            "human-game",
            "configs",
            "assets",
            "upload-ess-data",
            "design-simulation",
            "api",
            "benchmark",
        )
        if any(path.startswith(p) for p in _api_prefixes):
            return jsonify({"error": "Not found"}), 404
        idx = _STATIC_DIR / "index.html"
        if idx.exists():
            return send_from_directory(str(_STATIC_DIR), "index.html")
        return jsonify({"error": "Not found"}), 404

    # ── List available config files ───────────────────────────────────────────

    @app.get("/configs")
    def list_configs():
        # Only expose curated, static config files. Wizard/run-generated configs
        # live under configs/wizard/ (or carry an exp_-style id) and would leak
        # experiment identifiers and run layout, so they are excluded.
        configs = []
        if _CONFIGS_ROOT.exists():
            for p in sorted(_CONFIGS_ROOT.rglob("*.yaml")):
                rel = str(p.relative_to(_CONFIGS_ROOT.parent))
                parts = p.relative_to(_CONFIGS_ROOT).parts
                if "__pycache__" in rel:
                    continue
                if parts and parts[0] == "wizard":
                    continue
                if p.stem.startswith("exp_"):
                    continue
                configs.append(rel)
        return jsonify({"configs": configs})

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/ping")
    def ping():
        """Ultra-cheap keep-alive target for UptimeRobot / cron-job.org pingers.
        No auth, no filesystem I/O, no imports — returns instantly.
        Configure your external pinger to hit GET /ping every 10-14 minutes to
        prevent Render's free tier from spinning down the dyno."""
        return ("pong", 200, {"Content-Type": "text/plain", "Cache-Control": "no-store"})

    @app.get("/health")
    def health():
        # Public liveness probe — never report credential presence here.
        # For provider availability, use /api/capabilities. For full
        # diagnostics (including key presence), use /admin/health with
        # the BGF_API_TOKEN bearer.
        checks = {
            "experiments_dir": _EXPERIMENTS_ROOT.exists(),
            "tracker_index": _TRACKER_INDEX.exists(),
            "ollama": _ollama_available(),
        }
        status = "ok" if checks["experiments_dir"] else "degraded"
        return jsonify({"status": status, "service": "bgf-api", "checks": checks})

    @app.get("/admin/health")
    def admin_health():
        """Authenticated diagnostics, including provider key presence.

        Requires BGF_API_TOKEN to be set and a matching bearer token. When
        BGF_API_TOKEN is unset (open dev mode), this endpoint is disabled
        rather than open — so we never leak key presence in open mode.
        Bypasses the GET-method auth exemption because this is the one read
        path that does expose credential metadata.
        """
        configured_token = os.environ.get("BGF_API_TOKEN", "").strip()
        if not configured_token:
            return jsonify({"error": "Admin endpoint disabled (BGF_API_TOKEN not configured)"}), 404
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authorization header required"}), 401
        if not hmac.compare_digest(auth_header[len("Bearer ") :].encode(), configured_token.encode()):
            logger.warning("Failed /admin/health auth from %s", request.remote_addr)
            return jsonify({"error": "Invalid token"}), 401
        checks = {
            "experiments_dir": _EXPERIMENTS_ROOT.exists(),
            "tracker_index": _TRACKER_INDEX.exists(),
            "ollama": _ollama_available(),
            "groq_key": bool(os.environ.get("GROQ_API_KEY", "").strip()),
            "openai_key": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        }
        status = "ok" if checks["experiments_dir"] else "degraded"
        return jsonify({"status": status, "service": "bgf-api", "checks": checks})

    # ── Capabilities (what's available server-side, no secrets exposed) ───────

    @app.get("/api/capabilities")
    def api_capabilities():
        groq_key = bool(os.environ.get("GROQ_API_KEY", "").strip())
        openai_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())

        ollama_ok = False
        try:
            import urllib.request as _ur

            with _ur.urlopen("http://localhost:11434/api/tags", timeout=1) as _r:
                ollama_ok = _r.status == 200
        except Exception:
            pass

        if ollama_ok:
            preferred = "ollama"
            _maybe_warmup_ollama()
        elif groq_key:
            preferred = "groq"
        elif openai_key:
            preferred = "openai"
        else:
            preferred = None

        return jsonify(
            {
                "design": {
                    "groq": groq_key,
                    "openai": openai_key,
                    "ollama": ollama_ok,
                    "preferred": preferred,
                    "server_configured": preferred is not None,
                },
                "simulation": {
                    "groq": groq_key,
                    "openai": openai_key,
                    "ollama": ollama_ok,
                },
            }
        )

    # ── B_RLHF Benchmark (spec, leaderboard, submissions) ────────────────────

    _BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"
    _BENCHMARK_SUBMISSIONS_DIR = _BENCHMARK_DIR / "submissions"

    @app.get("/benchmark/spec")
    def benchmark_spec():
        """Return SPECIFICATION.md as plain markdown for the web UI."""
        spec_path = _BENCHMARK_DIR / "SPECIFICATION.md"
        if not spec_path.exists():
            return jsonify({"error": "SPECIFICATION.md not found"}), 404
        return spec_path.read_text(), 200, {"Content-Type": "text/markdown"}

    @app.get("/benchmark/leaderboard")
    def benchmark_leaderboard():
        """Return LEADERBOARD.md (auto-generated) as markdown."""
        lb_path = _BENCHMARK_DIR / "LEADERBOARD.md"
        if not lb_path.exists():
            return (
                "# B_RLHF Benchmark Leaderboard\n\n_No submissions yet — run "
                "`python benchmark/build_leaderboard.py`._\n",
                200,
                {"Content-Type": "text/markdown"},
            )
        return lb_path.read_text(), 200, {"Content-Type": "text/markdown"}

    @app.get("/benchmark/submissions")
    def benchmark_submissions():
        """List every score card under benchmark/submissions/.

        Returns a JSON array of compact summaries (omits the heavy
        per-seed audit blocks). Use /benchmark/submissions/<name> to
        fetch a full card.
        """
        if not _BENCHMARK_SUBMISSIONS_DIR.exists():
            return jsonify([])
        out: list[dict] = []
        for path in sorted(_BENCHMARK_SUBMISSIONS_DIR.glob("*.json")):
            try:
                card = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            sub = card.get("submission", {})
            scores = card.get("scores", {})
            out.append(
                {
                    "id": path.stem,
                    "name": sub.get("name"),
                    "team": sub.get("team"),
                    "tier": sub.get("tier"),
                    "date": sub.get("date"),
                    "model": (sub.get("llm") or {}).get("model_id"),
                    "memory": (sub.get("grounding_config") or {}).get("memory"),
                    "rag": (sub.get("grounding_config") or {}).get("rag"),
                    "schema_version": card.get("schema_version"),
                    "scores": {
                        "condition_A": scores.get("condition_A"),
                        "condition_B": scores.get("condition_B"),
                        "delta_B_RLHF_relative": scores.get("delta_B_RLHF_relative"),
                        "delta_BRM_composite": scores.get("delta_BRM_composite"),
                    },
                }
            )
        return jsonify(out)

    @app.get("/benchmark/submissions/<name>")
    def benchmark_submission_detail(name: str):
        """Return the full score-card JSON for a single submission."""
        # Sanitize: alphanumerics, dash, underscore only.
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
            return jsonify({"error": "invalid submission name"}), 400
        path = _BENCHMARK_SUBMISSIONS_DIR / f"{name}.json"
        if not path.exists():
            return jsonify({"error": "submission not found"}), 404
        return jsonify(json.loads(path.read_text()))

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

        try:
            body = request.get_json(force=True, silent=False) or {}
        except Exception:
            return jsonify({"error": "Request body must be valid JSON"}), 400
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

        # Include AI-design scenario context if present
        scenario = _safe_json_file(exp_dir / "scenario.json")
        if scenario:
            payload["scenario"] = scenario

        return jsonify(payload)

    # ── Network replay (real per-round trust/cooperation graph) ───────────────

    @app.get("/replay/<exp_id>")
    @_read_limit
    def replay(exp_id: str):
        """Reconstruct the real per-round interaction network from events.jsonl.

        Unlike the decorative flock animation, every node action, speech
        bubble, and edge here comes from the recorded simulation: each round
        carries each agent's actual action/target/wealth/stress plus the
        directed cooperate/steal edges that actually fired that round.

        Agents are capped at _REPLAY_AGENT_CAP (most interaction-active first)
        to keep the payload and the SVG readable for large populations.
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

        events_path = exp_dir / "events.jsonl"
        if not events_path.exists():
            return jsonify({"error": "No events log — replay requires a run with saved events"}), 404

        all_events = _read_events_cached(events_path)
        if not all_events:
            return jsonify({"error": "Events log is empty"}), 404

        _REPLAY_AGENT_CAP = 60

        def _agent_key(aid: str):
            # Natural sort: "agent_2" < "agent_10".
            digits = "".join(ch for ch in aid if ch.isdigit())
            return (int(digits) if digits else 0, aid)

        # First pass: interaction degree (to pick which agents to show) and
        # which agents ever steal (adversarial / "bad apple").
        degree: dict[str, int] = {}
        adversarial: set[str] = set()
        all_agents: set[str] = set()
        for ev in all_events:
            aid = ev.get("agent_id")
            if not aid:
                continue
            all_agents.add(aid)
            act = ev.get("action", {}) or {}
            atype = act.get("action_type", "")
            tgt = act.get("target_agent_id")
            if atype == "steal":
                adversarial.add(aid)
            if atype in ("cooperate", "steal") and tgt:
                degree[aid] = degree.get(aid, 0) + 1
                degree[tgt] = degree.get(tgt, 0) + 1

        shown = sorted(all_agents, key=lambda a: (-degree.get(a, 0), _agent_key(a)))
        shown = shown[:_REPLAY_AGENT_CAP]
        shown_sorted = sorted(shown, key=_agent_key)
        shown_set = set(shown_sorted)

        # Second pass: bucket events by round.
        rounds: dict[int, dict] = {}
        for ev in all_events:
            aid = ev.get("agent_id")
            if aid not in shown_set:
                continue
            rid = ev.get("round_id")
            if rid is None:
                continue
            bucket = rounds.setdefault(rid, {"round": rid, "states": {}, "edges": []})
            act = ev.get("action", {}) or {}
            atype = act.get("action_type", "unknown")
            tgt = act.get("target_agent_id")
            state = ev.get("state_after", {}) or {}
            reason = (act.get("reasoning_summary") or "").strip()
            if len(reason) > 300:
                reason = reason[:297] + "…"
            bucket["states"][aid] = {
                "a": atype,
                "t": tgt if tgt in shown_set else None,
                "w": round(float(state.get("wealth", 0) or 0), 1),
                "s": round(float(state.get("stress", 0) or 0), 2),
                "r": reason,
            }
            if atype in ("cooperate", "steal") and tgt in shown_set:
                bucket["edges"].append([aid, tgt, atype])

        ordered = [rounds[k] for k in sorted(rounds.keys())]

        return jsonify(
            {
                "experiment_id": exp_id,
                "n_agents": len(all_agents),
                "shown": len(shown_sorted),
                "agent_ids": shown_sorted,
                "adversarial": sorted(adversarial & shown_set, key=_agent_key),
                "rounds": ordered,
            }
        )

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
            try:
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
            finally:
                conn.close()

            return jsonify({"experiments": df.to_dict(orient="records")})

        except Exception as exc:
            logger.error("experiments_list error: %s", exc, exc_info=True)
            return jsonify({"error": "Failed to query experiments"}), 500

    # ── Metric regression detection ───────────────────────────────────────────

    @app.get("/regression")
    @_experiments_limit
    def regression_list():
        """Flag runs whose metric deviates from recent same-policy history.

        Query params: ?metric=wealth_mean&window=5&n_mad=3.0
        Read-only; mirrors ruflo's rolling-window perf-regression check.
        """
        ok, err = _check_auth()
        if not ok:
            return err
        if not _TRACKER_INDEX.exists():
            return jsonify({"regressions": [], "note": "Tracker index not available"})
        try:
            from tracker.analytics import detect_regression

            metric = request.args.get("metric", "wealth_mean")
            window = int(request.args.get("window", 5))
            n_mad = float(request.args.get("n_mad", 3.0))
            flagged = detect_regression(
                metric=metric,
                index_path=str(_TRACKER_INDEX),
                window=window,
                n_mad=n_mad,
            )
            return jsonify({"metric": metric, "count": len(flagged), "regressions": flagged})
        except Exception as exc:
            logger.error("regression_list error: %s", exc, exc_info=True)
            return jsonify({"error": "Failed to compute regressions"}), 500

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

        try:
            body = request.get_json(force=True, silent=False) or {}
        except Exception:
            return jsonify({"error": "Request body must be valid JSON"}), 400
        question = str(body.get("question", "Describe your recent decisions."))[:1000]

        # ── Try live IPC first (simulation must be running) ───────────────────
        state_data = _safe_json_file(exp_dir / "run_state.json") or {}
        run_status = state_data.get("status", "unknown")

        if run_status == "running":
            try:
                from simulation.ipc import SimulationIPCClient

                with _ipc_lock:
                    if exp_id not in _ipc_clients:
                        if len(_ipc_clients) >= 20:
                            _ipc_clients.pop(next(iter(_ipc_clients)))
                        _ipc_clients[exp_id] = SimulationIPCClient(base_dir=str(exp_dir), timeout=15.0)
                    client = _ipc_clients[exp_id]

                reply = client.interview_agent(agent_id, question) or {}

                # If the running simulation surfaced live context (state +
                # hierarchical memory reflection), do an LLM synthesis pass
                # here so the live interview reaches the same quality as the
                # data-replay path. Otherwise return the static answer the
                # IPC server already built.
                live_ctx = reply.get("live_context") if isinstance(reply, dict) else None
                if live_ctx and os.environ.get("OPENAI_API_KEY"):
                    synth = _synthesize_live_interview_answer(
                        exp_id=exp_id,
                        exp_dir=exp_dir,
                        agent_id=agent_id,
                        question=question,
                        live_context=live_ctx,
                    )
                    if synth is not None:
                        _append_interview_response(exp_dir, agent_id, question, synth["response"], extra=synth)
                        return jsonify(synth)

                # Backward-compatible fallback: ship the static answer.
                static_answer = reply.get("answer") if isinstance(reply, dict) else None
                if static_answer:
                    extra = {
                        "persona_used": bool(live_ctx and live_ctx.get("profile")),
                        "history_window": 0,
                        "memory_source": "live_static",
                        "model": "rule_based",
                        "source": "ipc_live_static",
                    }
                    _append_interview_response(exp_dir, agent_id, question, static_answer, extra=extra)
                    return jsonify({"response": static_answer, **extra})

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

        all_events = _read_events_cached(events_path)
        agent_events = [ev for ev in all_events if ev.get("agent_id") == agent_id]

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

        # ── Build structured context (persona, scenario, memory, history) ───
        ctx = _interview_prompt_context(exp_dir, agent_id, agent_events)
        scenario_ctx = _safe_json_file(exp_dir / "scenario.json") or {}
        scenario_title = scenario_ctx.get("scenario_title", "")
        population_traits = scenario_ctx.get("population_traits", {})

        # ── Try OpenAI for natural-language answer ────────────────────────────
        oai_key = os.environ.get("OPENAI_API_KEY", "")
        fallback_reason = "no_openai_key"
        if oai_key:
            try:
                import time as _time

                from openai import OpenAI as _OAI
                from openai import RateLimitError as _RLE

                # Reuse singleton — avoids TCP reconnect on every interview call
                _oai_prefix = oai_key[:8]
                oai = _oai_clients_get_or_create(_oai_prefix, lambda: _OAI(api_key=oai_key))

                # Compose the system prompt from the structured context blocks.
                # Order them so the largest static portion (scenario + persona)
                # sits at the front of the prompt where gpt-4o-mini's automatic
                # prefix cache can reuse it across questions for the same run.
                persona_section = (
                    f"You are {agent_id}, a synthetic agent in BGF simulation ({exp_id}).\n{ctx['persona_block']}\n"
                    if ctx["persona_block"]
                    else f"You are {agent_id}, a synthetic agent in BGF simulation ({exp_id}).\n"
                )
                behavioral_summary = (
                    f"Across {total_rounds} rounds your dominant action was '{dominant}' "
                    f"({', '.join(f'{v}× {k}' for k, v in sorted(action_counts.items(), key=lambda x: -x[1]))}). "
                    f"Final wealth: {final_wealth:.1f} (started at {first_wealth:.1f}, delta {wealth_delta:+.1f}). "
                    f"Final stress: {final_stress:.2f}. Cooperation rounds: {coop_count}."
                )
                blocks = [
                    persona_section,
                    ctx["scenario_block"],
                    behavioral_summary,
                    f"Recent history (last {ctx['history_window']} rounds):\n{ctx['history_text']}"
                    if ctx["history_text"]
                    else "",
                    ctx["social_block"],
                    ctx["memory_block"],
                    (
                        "Answer the user's question in first-person as this agent, "
                        "drawing from your demographics, dispositions, and behavioral history above. "
                        "Be natural and concise (2-4 sentences). "
                        "Do not repeat phrasing used by 'typical' agents — express your individual perspective. "
                        "Do not mention 'LLM fallback', simulation internals, or action names like "
                        "'cooperate/work/save' unless directly asked about strategy."
                    ),
                ]
                system_prompt = "\n\n".join(b for b in blocks if b)
                _call_kwargs = dict(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    max_tokens=220,
                    temperature=0.7,
                )
                _resp = None
                for _att in range(3):
                    try:
                        _resp = oai.chat.completions.create(**_call_kwargs)
                        break
                    except _RLE:
                        _time.sleep(2**_att)
                if _resp is None:
                    raise RuntimeError("OpenAI rate limit — all retries exhausted")
                answer = _resp.choices[0].message.content.strip()
                extra = {
                    "persona_used": bool(ctx["persona_block"]),
                    "history_window": ctx["history_window"],
                    "memory_source": ctx["memory_source"],
                    "model": "gpt-4o-mini",
                    "source": "replay_llm",
                }
                _append_interview_response(exp_dir, agent_id, question, answer, extra=extra)
                return jsonify({"response": answer, **extra})
            except Exception as exc:
                logger.warning("Replay LLM failed, using rule-based fallback: %s", exc)
                fallback_reason = str(exc)[:200]

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

        elif scenario_title and not any(
            w in q_low
            for w in (
                "wealth",
                "strategy",
                "cooperation",
                "trust",
                "neighbor",
                "round",
                "last decision",
                "feel",
                "rich",
                "money",
                "cooperat",
                "social",
                "pool",
            )
        ):
            # Question is scenario-specific (election, policy, climate, etc.) but no
            # LLM key available.  Derive an in-character answer from population traits.
            trust_level = population_traits.get("trust_institutions", 0.5)
            life_sat = population_traits.get("life_satisfaction", 0.5)

            if trust_level >= 0.6:
                trust_stance = "I tend to trust established institutions and their processes"
            elif trust_level <= 0.35:
                trust_stance = "I'm skeptical of institutions — they rarely deliver on their promises"
            else:
                trust_stance = "I have mixed feelings about how reliable institutions are"

            if life_sat >= 0.6:
                outlook = "and I'm broadly optimistic about how things will unfold"
            elif life_sat <= 0.35:
                outlook = "and I'm frankly worried about the direction things are heading"
            else:
                outlook = "though I try to keep a realistic outlook"

            answer = (
                f"Speaking as someone shaped by the '{scenario_title}' context: {trust_stance}, "
                f"{outlook}. My experience across {total_rounds} rounds — where I mainly "
                f"{'cooperated' if coop_count > total_rounds // 3 else 'acted independently'} — "
                f"reinforced that {'working together pays off' if coop_count > total_rounds // 3 else 'self-reliance matters'}. "
                f"That shapes how I read the question you're asking."
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

        extra = {
            "persona_used": bool(ctx["persona_block"]),
            "history_window": ctx["history_window"],
            "memory_source": ctx["memory_source"],
            "model": "rule_based",
            "source": "replay_data",
            "fallback_reason": fallback_reason,
        }
        _append_interview_response(exp_dir, agent_id, question, answer, extra=extra)
        return jsonify({"response": answer, **extra})

    # ── Anchor (macro broadcast view) ────────────────────────────────────────

    @app.post("/anchor/<exp_id>")
    @_read_limit
    def anchor(exp_id: str):
        """
        Query a simulation-wide "anchor" agent that knows every decision from
        every agent.  Unlike /interview (first-person, single agent), the anchor
        has an omniscient macro view and answers in the style of a TV news anchor
        summarising what happened across the whole population.

        Body (JSON):
          question   str   The question to ask the anchor (required)

        Returns:
          response   str   Anchor's natural-language answer
          source     str   "anchor_llm" | "anchor_data"
          stats      dict  Raw aggregate statistics used to build the answer
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

        try:
            body = request.get_json(force=True, silent=False) or {}
        except Exception:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        question = str(body.get("question", "")).strip()
        if not question:
            return jsonify({"error": "Missing 'question' in request body"}), 400
        if len(question) > 1000:
            return jsonify({"error": "Question too long (max 1000 chars)"}), 400

        events_path = exp_dir / "events.jsonl"
        if not events_path.exists():
            return jsonify({"error": "No events log found — anchor requires a completed run with saved events"}), 404

        # ── Load ALL events from all agents ───────────────────────────────────
        all_events = _read_events_cached(events_path)

        if not all_events:
            return jsonify({"error": "No events found in this experiment"}), 404

        # ── Aggregate statistics across all agents and rounds ─────────────────
        agents_seen: set = set()
        rounds_seen: set = set()
        action_counts: dict = {}  # action_type → total count
        per_agent_actions: dict = {}  # agent_id → {action_type → count}
        per_agent_wealth_start: dict = {}
        per_agent_wealth_end: dict = {}
        per_agent_stress_end: dict = {}
        per_round_actions: dict = {}  # round_id → {action_type → count}
        cooperation_pairs: dict = {}  # (agent_id, target_id) → count
        steal_pairs: dict = {}
        # per-agent last reasoning summary (for scenario-specific opinion mining)
        per_agent_last_reasoning: dict = {}  # agent_id → latest reasoning_summary text
        all_reasoning_texts: list = []  # flat list of all non-empty reasoning strings

        for ev in all_events:
            agent_id = ev.get("agent_id", "unknown")
            round_id = ev.get("round_id", 0)
            act = ev.get("action", {})
            act_type = act.get("action_type", "unknown")
            target = act.get("target_agent_id")
            state_after = ev.get("state_after", {})

            agents_seen.add(agent_id)
            rounds_seen.add(round_id)
            action_counts[act_type] = action_counts.get(act_type, 0) + 1

            if agent_id not in per_agent_actions:
                per_agent_actions[agent_id] = {}
            per_agent_actions[agent_id][act_type] = per_agent_actions[agent_id].get(act_type, 0) + 1

            wealth = state_after.get("wealth")
            stress = state_after.get("stress")
            if wealth is not None:
                per_agent_wealth_end[agent_id] = wealth
                if agent_id not in per_agent_wealth_start:
                    per_agent_wealth_start[agent_id] = wealth
            if stress is not None:
                per_agent_stress_end[agent_id] = stress

            if round_id not in per_round_actions:
                per_round_actions[round_id] = {}
            per_round_actions[round_id][act_type] = per_round_actions[round_id].get(act_type, 0) + 1

            if act_type == "cooperate" and target:
                k = (agent_id, target)
                cooperation_pairs[k] = cooperation_pairs.get(k, 0) + 1
            if act_type == "steal" and target:
                k = (agent_id, target)
                steal_pairs[k] = steal_pairs.get(k, 0) + 1

            rsn = act.get("reasoning_summary", "").strip()
            rsn = rsn.replace("[LLM fallback: ", "").rstrip("]")
            if rsn:
                per_agent_last_reasoning[agent_id] = rsn  # keeps latest round's reasoning
                all_reasoning_texts.append(rsn)

        total_events = len(all_events)
        n_agents = len(agents_seen)
        n_rounds = len(rounds_seen)
        total_coop = action_counts.get("cooperate", 0)
        total_steal = action_counts.get("steal", 0)
        coop_rate = total_coop / (total_coop + total_steal) if (total_coop + total_steal) > 0 else None
        dominant_action = max(action_counts, key=action_counts.get) if action_counts else "unknown"

        # Per-agent dominant action (majority vote)
        agent_dominant = {ag: max(acts, key=acts.get) for ag, acts in per_agent_actions.items()}
        majority_action = (
            max(set(agent_dominant.values()), key=list(agent_dominant.values()).count) if agent_dominant else "unknown"
        )
        majority_count = list(agent_dominant.values()).count(majority_action)

        # Wealth stats
        wealth_values = list(per_agent_wealth_end.values())
        wealth_mean = sum(wealth_values) / len(wealth_values) if wealth_values else 0
        wealth_max = max(wealth_values) if wealth_values else 0
        wealth_min = min(wealth_values) if wealth_values else 0
        richest = max(per_agent_wealth_end, key=per_agent_wealth_end.get) if per_agent_wealth_end else None
        poorest = min(per_agent_wealth_end, key=per_agent_wealth_end.get) if per_agent_wealth_end else None

        # Cooperation network highlights
        most_coop_pair = max(cooperation_pairs, key=cooperation_pairs.get) if cooperation_pairs else None
        most_steal_pair = max(steal_pairs, key=steal_pairs.get) if steal_pairs else None

        # Round-by-round cooperation trend (first vs last round)
        round_ids_sorted = sorted(per_round_actions.keys())
        first_round_coop = per_round_actions[round_ids_sorted[0]].get("cooperate", 0) if round_ids_sorted else 0
        last_round_coop = per_round_actions[round_ids_sorted[-1]].get("cooperate", 0) if round_ids_sorted else 0
        coop_trend = (
            "increasing"
            if last_round_coop > first_round_coop
            else ("decreasing" if last_round_coop < first_round_coop else "stable")
        )

        # ── Load scenario context ─────────────────────────────────────────────
        scenario_ctx = _safe_json_file(exp_dir / "scenario.json") or {}
        scenario_title = scenario_ctx.get("scenario_title", "")
        scenario_description = scenario_ctx.get("scenario_description", "")
        # Tokens from the scenario description that agents might discuss
        _stop = {"a", "an", "the", "and", "or", "in", "of", "for", "to", "is", "are", "be", "their", "its"}
        scenario_tokens = {
            w.lower().strip(".,;:!?\"'")
            for w in (scenario_title + " " + scenario_description).split()
            if len(w) > 3 and w.lower() not in _stop
        }

        # ── Load prompt corpus from prompts.jsonl (for LLM anchor synthesis) ──
        # Agents share a template that only differs by persona substitution, so a
        # single example carries the structure. Pull the first prompt + a few
        # diverse raw_outputs (different agents) so the anchor can talk about
        # what agents were actually asked and what they actually returned.
        sample_prompt: str = ""
        sample_raw_outputs: list[tuple[str, str]] = []  # (agent_id, raw_output)
        prompts_path = exp_dir / "prompts.jsonl"
        if prompts_path.exists():
            try:
                with prompts_path.open() as _pf:
                    _seen_agents: set = set()
                    for _i, _line in enumerate(_pf):
                        if _i > 200:
                            break  # bound work for very long runs
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _rec = json.loads(_line)
                        except json.JSONDecodeError:
                            continue
                        if not sample_prompt:
                            sample_prompt = str(_rec.get("prompt", ""))[:2500]
                        _ag = _rec.get("agent_id", "")
                        _raw = str(_rec.get("raw_output", "")).strip()
                        if _raw and _ag and _ag not in _seen_agents and len(sample_raw_outputs) < 5:
                            _seen_agents.add(_ag)
                            sample_raw_outputs.append((_ag, _raw[:400]))
            except OSError:
                pass

        # ── Opinion mining: tally keyword mentions across all reasoning texts ──
        combined_reasoning = " ".join(all_reasoning_texts).lower()

        def _tally_keywords(candidates: list[str]) -> dict[str, int]:
            """Count how many reasoning texts mention each candidate keyword."""
            tally: dict[str, int] = {}
            for kw in candidates:
                kw_low = kw.lower()
                tally[kw] = sum(1 for t in all_reasoning_texts if kw_low in t.lower())
            return tally

        def _find_question_options(q: str) -> list[str]:
            """
            Extract candidate options from a question like 'paper or monograph?'
            Only activates on explicit 'X or Y' / 'X vs Y' patterns — prevents
            generic nouns in the question (e.g. 'decision', 'majority') from being
            mistaken for option keywords.
            """
            import re as _re

            or_match = _re.search(r"\b(\w{4,})\s+(?:or|vs\.?|versus)\s+(\w{4,})\b", q, _re.I)
            if not or_match:
                return []
            opts = [
                or_match.group(1).lower().strip(".,;:!?\"'"),
                or_match.group(2).lower().strip(".,;:!?\"'"),
            ]
            # Ground each option in the scenario or reasoning corpus
            return [o for o in opts if o in scenario_tokens or o in combined_reasoning]

        # Build stats payload returned to caller regardless of LLM path
        stats = {
            "n_agents": n_agents,
            "n_rounds": n_rounds,
            "total_events": total_events,
            "action_counts": action_counts,
            "dominant_action_overall": dominant_action,
            "majority_agent_action": majority_action,
            "majority_agent_count": majority_count,
            "cooperation_rate": round(coop_rate, 4) if coop_rate is not None else None,
            "wealth_mean": round(wealth_mean, 2),
            "wealth_max": round(wealth_max, 2),
            "wealth_min": round(wealth_min, 2),
            "richest_agent": richest,
            "poorest_agent": poorest,
            "cooperation_trend": coop_trend,
            "most_cooperative_pair": list(most_coop_pair) if most_coop_pair else None,
            "most_stealing_pair": list(most_steal_pair) if most_steal_pair else None,
        }

        action_breakdown = ", ".join(f"{k} {v}×" for k, v in sorted(action_counts.items(), key=lambda x: -x[1]))

        # ── Load collected interview responses (built up by /interview calls) ──
        interview_log: list[dict] = []
        interview_log_path = exp_dir / "interview_responses.jsonl"
        if interview_log_path.exists():
            try:
                with interview_log_path.open() as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if _line:
                            try:
                                interview_log.append(json.loads(_line))
                            except json.JSONDecodeError:
                                pass
            except OSError:
                pass

        # agent_id → their latest response text (for opinion tally)
        per_agent_interview: dict[str, str] = {}
        for rec in interview_log:
            per_agent_interview[rec.get("agent_id", "")] = rec.get("response", "")

        # ── Rule-based anchor (no paid API required) ─────────────────────────
        q_low = question.lower()
        coop_pct_str = f"{coop_rate:.1%}" if coop_rate is not None else "N/A (no steal actions recorded)"

        # ── Scenario-opinion path ─────────────────────────────────────────────
        # Priority: (1) tallied interview responses, (2) event reasoning text,
        # (3) redirect asking user to interview more agents.
        q_options = _find_question_options(question)
        _econ_actions = {"work", "save", "cooperate", "steal", "cooperation", "defect"}
        scenario_options = [o for o in q_options if o not in _econ_actions]

        # If the anchor question has no explicit "X or Y" options but interview
        # responses exist, infer options from the questions stored in those logs.
        # This lets "What was the majority decision?" resolve to "monograph or paper"
        # when agents were interviewed with "which do you prefer monograph or paper?".
        #
        # NOTE: we bypass _find_question_options (which requires options to appear in
        # the scenario corpus or reasoning texts) because interview questions are
        # self-evidencing — if an agent was explicitly asked "X or Y", those ARE
        # the valid options regardless of whether they appear anywhere in events.
        if not scenario_options and per_agent_interview:
            import re as _re2

            for rec in interview_log:
                q_stored = rec.get("question", "")
                _or_m = _re2.search(r"\b(\w{4,})\s+(?:or|vs\.?|versus)\s+(\w{4,})\b", q_stored, _re2.I)
                if _or_m:
                    opts = [
                        _or_m.group(1).lower().strip(".,;:!?\"'"),
                        _or_m.group(2).lower().strip(".,;:!?\"'"),
                    ]
                    inferred_non_econ = [o for o in opts if o not in _econ_actions]
                    if inferred_non_econ:
                        scenario_options = inferred_non_econ
                        break

        # Economic questions should always use the economic handlers below, even
        # when scenario_options are present — otherwise "Was there stealing?" would
        # return the monograph/paper tally instead of the steal count.
        _econ_q_kw = {
            "steal",
            "defect",
            "cheat",
            "bad",
            "cooperat",
            "social",
            "pool",
            "wealth",
            "rich",
            "poor",
            "money",
            "economic",
            "round",
            "trend",
            "over time",
            "evolv",
            "chang",
            "summary",
            "overview",
            "what happened",
            "tell me",
            "describe",
            "report",
        }
        _is_econ_q = any(kw in q_low for kw in _econ_q_kw)

        if scenario_options and not _is_econ_q:

            def _tally_from_texts(texts_by_agent: dict) -> tuple:
                """Assign each agent to exactly ONE option — the one they mention most.

                Keyword-presence counting (old approach) double-counts agents whose
                response mentions both option words (e.g. "I prefer monograph over
                paper"), inflating both tallies equally.  Instead, pick the option
                with the highest occurrence count per agent; ties are broken by first
                position in the text (earlier = more likely the subject, not the
                comparison target).
                """
                counts: dict = {}
                by_agent: dict = {}
                for ag, txt in texts_by_agent.items():
                    txt_low = txt.lower()
                    occ = {opt: txt_low.count(opt.lower()) for opt in scenario_options}
                    if not any(occ.values()):
                        continue  # agent response doesn't mention either option

                    def _sort_key(opt: str):
                        n = occ[opt]
                        pos = txt_low.find(opt.lower())
                        # primary: more mentions wins; secondary: earlier position wins
                        return (n, -(pos if pos >= 0 else len(txt_low)))

                    best = max(scenario_options, key=_sort_key)
                    if occ[best] > 0:
                        counts[best] = counts.get(best, 0) + 1
                        by_agent.setdefault(best, []).append(ag)
                return counts, by_agent

            def _format_opinion_answer(counts, by_agent, n_interviewed, source_label):
                top = max(counts, key=counts.get)
                others = {k: v for k, v in counts.items() if k != top}
                runner = max(others, key=others.get) if others else None
                agents_top = by_agent.get(top, [])
                scenario_note = f" (Scenario: {scenario_title})" if scenario_title else ""
                coverage = (
                    f"{n_interviewed} of {n_agents} agents interviewed"
                    if n_interviewed < n_agents
                    else f"all {n_agents} agents"
                )
                runner_note = f" {len(by_agent.get(runner, []))} agent(s) preferred '{runner}'." if runner else ""
                return (
                    f"Based on {coverage}{scenario_note}: the majority preferred '{top}' "
                    f"({counts[top]} agent(s) — "
                    f"{', '.join(agents_top[:5])}{'…' if len(agents_top) > 5 else ''})."
                    f"{runner_note} [{source_label}]"
                )

            # 1. Collected interview responses (built automatically by /interview calls)
            if per_agent_interview:
                # Prefer LLM-based semantic stance extraction over substring
                # counting — "I dislike paper" was previously counted as a
                # vote for paper. Falls back to substring tally when no
                # OpenAI key is available OR when the LLM extracted no usable
                # stances (e.g. fake key returns errors, prompt returned
                # `stance: null` for every agent). Empty semantic output must
                # not block the substring path or we lose the answer.
                semantic = _semantic_stance_tally(per_agent_interview, question, scenario_options)
                method = "substring"
                if semantic is not None and semantic[0]:
                    counts, by_agent, _details = semantic
                    method = "semantic_llm"
                else:
                    counts, by_agent = _tally_from_texts(per_agent_interview)
                counts = {k: v for k, v in counts.items() if v > 0}
                if counts:
                    answer = _format_opinion_answer(counts, by_agent, len(per_agent_interview), "from agent interviews")
                    stats["interviewed_agents"] = len(per_agent_interview)
                    stats["opinion_tally"] = counts
                    stats["tally_method"] = method
                    return jsonify({"response": answer, "source": "anchor_interviews", "stats": stats})

            # 2. Event reasoning summaries (works when real LLM content exists)
            if all_reasoning_texts:
                semantic = _semantic_stance_tally(per_agent_last_reasoning, question, scenario_options)
                method = "substring"
                if semantic is not None and semantic[0]:
                    counts, by_agent, _details = semantic
                    method = "semantic_llm"
                else:
                    counts, by_agent = _tally_from_texts(per_agent_last_reasoning)
                counts = {k: v for k, v in counts.items() if v > 0}
                if counts:
                    answer = _format_opinion_answer(
                        counts, by_agent, len(per_agent_last_reasoning), "from event reasoning"
                    )
                    stats["opinion_tally"] = counts
                    stats["tally_method"] = method
                    return jsonify({"response": answer, "source": "anchor_data", "stats": stats})

            # 3. No opinion data yet — tell the user what to do
            options_str = " vs ".join(f"'{o}'" for o in scenario_options)
            scenario_note = f" for '{scenario_title}'" if scenario_title else ""
            n_interviewed = len(per_agent_interview)
            progress_note = (
                f" You've interviewed {n_interviewed} of {n_agents} agents so far — keep going."
                if n_interviewed
                else " Interview agents in the panel above — I'll collect and tally their answers automatically."
            )
            answer = (
                f"I track economic actions (work, cooperate, save, steal), "
                f"not narrative opinions{scenario_note}. "
                f"To report the majority on {options_str}, I need to hear from each agent — "
                f"every interview answer is added to my knowledge base.{progress_note}"
            )
            return jsonify({"response": answer, "source": "anchor_data", "stats": stats})

        # ── LLM synthesis path ────────────────────────────────────────────────
        # Free-form anchor that can reason about scenario prompts, reasoning text,
        # and interview content — not just the four economic actions. Gated on
        # OPENAI_API_KEY; on any failure we fall through to the rule-based
        # handlers below so the endpoint never 500s.
        _anchor_oai_key = os.environ.get("OPENAI_API_KEY", "")
        if _anchor_oai_key:
            try:
                import time as _time

                from openai import OpenAI as _OAI
                from openai import RateLimitError as _RLE

                _oai_prefix = _anchor_oai_key[:8]
                oai = _oai_clients_get_or_create(_oai_prefix, lambda: _OAI(api_key=_anchor_oai_key))

                # Build a per-agent reasoning excerpt. Scale the sample window
                # with population size (capped at 200 to keep the prompt under
                # the context-window comfort zone), and bump the per-agent
                # quote cap to 400 chars so nuance isn't truncated away.
                _sample_cap = min(200, max(20, len(per_agent_last_reasoning)))
                _sampled = list(per_agent_last_reasoning.items())[:_sample_cap]
                reasoning_lines: list[str] = []
                for _ag, _txt in _sampled:
                    reasoning_lines.append(f"- {_ag}: {_txt[:400]}")
                reasoning_block = "\n".join(reasoning_lines) if reasoning_lines else "(no reasoning text recorded)"

                # Interview Q&A excerpt
                interview_lines: list[str] = []
                for _rec in interview_log[-15:]:
                    _q = str(_rec.get("question", ""))[:160]
                    _r = str(_rec.get("response", ""))[:240]
                    _ag = _rec.get("agent_id", "?")
                    interview_lines.append(f"- {_ag} | Q: {_q} | A: {_r}")
                interview_block = "\n".join(interview_lines) if interview_lines else "(no interviews collected yet)"

                # Raw-output sample (shows what agents actually emitted)
                raw_lines = [f"- {_ag}: {_raw}" for _ag, _raw in sample_raw_outputs]
                raw_block = "\n".join(raw_lines) if raw_lines else "(no raw outputs sampled)"

                prompt_excerpt = sample_prompt if sample_prompt else "(no prompts.jsonl available)"
                scenario_block = (
                    f"Scenario title: {scenario_title}\nScenario description: {scenario_description}\n"
                    if scenario_title or scenario_description
                    else "(no scenario design context)"
                )

                system_prompt = (
                    "You are the omniscient ANCHOR of a completed agent-based simulation. "
                    "You can see every agent's decisions, reasoning, and the prompt template "
                    "that drove them. Answer the operator's question precisely and concretely, "
                    "drawing on the materials below. Do NOT limit yourself to the four economic "
                    "actions (work/save/cooperate/steal) — engage with the scenario narrative, "
                    "the prompt content, and what agents actually said. If the question is about "
                    "the prompt, quote the relevant portion. If about opinions or themes, "
                    "synthesize across the reasoning excerpts. Cite agent ids when helpful. "
                    "Be factual; if the evidence is thin, say so. 3-6 sentences.\n\n"
                    f"=== SCENARIO ===\n{scenario_block}\n"
                    f"=== AGGREGATE STATS ===\n{json.dumps(stats, default=str)}\n\n"
                    f"=== AGENT DECISION PROMPT (shared template) ===\n{prompt_excerpt}\n\n"
                    f"=== SAMPLE RAW OUTPUTS ===\n{raw_block}\n\n"
                    f"=== PER-AGENT LATEST REASONING ===\n{reasoning_block}\n\n"
                    f"=== INTERVIEW Q&A LOG ===\n{interview_block}\n"
                )

                _call_kwargs = dict(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    max_tokens=400,
                    temperature=0.4,
                )
                _resp = None
                for _att in range(3):
                    try:
                        _resp = oai.chat.completions.create(**_call_kwargs)
                        break
                    except _RLE:
                        _time.sleep(2**_att)
                if _resp is not None:
                    answer = _resp.choices[0].message.content.strip()
                    return jsonify({"response": answer, "source": "anchor_llm", "stats": stats})
            except Exception as exc:
                logger.warning("Anchor LLM failed, using rule-based fallback: %s", exc)

        # ── Economic / structural question handlers ───────────────────────────
        if any(w in q_low for w in ("majority", "most", "common", "dominant", "popular", "did they")):
            answer = (
                f"Across all {n_rounds} rounds and {n_agents} agents, the most common action was "
                f"'{dominant_action}' with {action_counts[dominant_action]} occurrences. "
                f"{majority_count} out of {n_agents} agents had '{majority_action}' as their personal dominant strategy. "
                f"The cooperation rate stood at {coop_pct_str}, with a {coop_trend} trend over time."
            )
        elif any(w in q_low for w in ("cooperat", "social", "pool", "shared", "together")):
            pair_note = (
                f" The most active cooperative pair was {most_coop_pair[0]} and {most_coop_pair[1]} "
                f"({cooperation_pairs[most_coop_pair]}× together)."
                if most_coop_pair
                else ""
            )
            answer = (
                f"Cooperation was {coop_trend} across the simulation. "
                f"Agents cooperated {total_coop}× in total out of {total_events} decisions "
                f"— a rate of {coop_pct_str} against defection.{pair_note}"
            )
        elif any(w in q_low for w in ("wealth", "rich", "poor", "money", "economic")):
            answer = (
                f"At the end of the simulation, average agent wealth was {wealth_mean:.1f}. "
                f"The wealthiest agent was {richest} at {wealth_max:.1f}; "
                f"the poorest was {poorest} at {wealth_min:.1f}. "
                f"This spread reflects the cumulative effect of {action_breakdown} across {n_rounds} rounds."
            )
        elif any(w in q_low for w in ("steal", "defect", "cheat", "bad")):
            if total_steal:
                steal_note = (
                    f" The most active stealing pair was {most_steal_pair[0]} → {most_steal_pair[1]} "
                    f"({steal_pairs[most_steal_pair]}×)."
                    if most_steal_pair
                    else ""
                )
                answer = (
                    f"Defection (steal) occurred {total_steal}× across the simulation — "
                    f"{total_steal / total_events:.1%} of all decisions.{steal_note} "
                    f"Despite this, cooperation still accounted for {total_coop} decisions."
                )
            else:
                answer = (
                    f"No stealing or defection was recorded in this simulation across all {n_agents} agents "
                    f"and {n_rounds} rounds. The population maintained full cooperative compliance."
                )
        elif any(w in q_low for w in ("round", "trend", "over time", "evolv", "chang")):
            answer = (
                f"Cooperation was {coop_trend} over the {n_rounds} rounds — "
                f"starting at {first_round_coop} cooperative actions in round {round_ids_sorted[0]} "
                f"and ending at {last_round_coop} in round {round_ids_sorted[-1]}. "
                f"Overall the population made {total_events} decisions: {action_breakdown}."
            )
        elif any(w in q_low for w in ("summary", "overview", "what happened", "tell me", "describe", "report")):
            answer = (
                f"In this simulation, {n_agents} agents played {n_rounds} rounds making {total_events} decisions. "
                f"The dominant strategy was '{dominant_action}' ({action_breakdown}). "
                f"Cooperation rate was {coop_pct_str} with a {coop_trend} trend. "
                f"Final wealth ranged from {wealth_min:.1f} to {wealth_max:.1f} (mean {wealth_mean:.1f}). "
                + (
                    f"The most cooperative pair was {most_coop_pair[0]} ↔ {most_coop_pair[1]}."
                    if most_coop_pair
                    else ""
                )
            )
        else:
            answer = (
                f"Population of {n_agents} agents over {n_rounds} rounds: {action_breakdown}. "
                f"Cooperation rate {coop_pct_str}, trend {coop_trend}. "
                f"Mean wealth {wealth_mean:.1f} (range {wealth_min:.1f}–{wealth_max:.1f})."
            )

        return jsonify({"response": answer, "source": "anchor_data", "stats": stats})

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

        try:
            body = request.get_json(force=True, silent=False) or {}
        except Exception:
            return jsonify({"error": "Request body must be valid JSON"}), 400
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
                    if len(_ipc_clients) >= 20:
                        _ipc_clients.pop(next(iter(_ipc_clients)))
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
        """Upload a population data file for empirical grounding.

        Accepts ``.csv``, ``.parquet``, ``.sav`` (SPSS — ESS native format),
        and ``.dta`` (Stata). CSV uploads have their delimiter sniffed
        automatically. ESS short codes (``agea``, ``gndr``, ``cntry`` …) are
        normalized to the canonical schema used by the generator/grounder.

        Returns a structured analysis of which BGF simulation dimensions the
        file covers (directly, via derivation, or via config fallback),
        plus the list of aliases that were auto-applied so the UI can show
        provenance.

        The returned file_id can be passed to /simulate-wizard as ess_data_file_id.
        """
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
        _SUPPORTED_EXT = (".csv", ".parquet", ".sav", ".dta")
        if not fname_lower.endswith(_SUPPORTED_EXT):
            return jsonify(
                {
                    "error": (
                        f"Unsupported file type. Accepted extensions: {', '.join(_SUPPORTED_EXT)}. "
                        "ESS distributes microdata as .sav (SPSS); upload it directly without converting."
                    )
                }
            ), 400

        raw = f.read()
        if len(raw) > _UPLOAD_MAX_BYTES:
            return jsonify({"error": "File too large — maximum is 50 MB"}), 413

        df, load_error = _load_uploaded_dataframe(f.filename, raw)
        if load_error is not None:
            return jsonify({"error": load_error}), 422
        # Inform mypy/downstream code that df is non-None past this point.
        assert df is not None  # noqa: S101 — runtime invariant, not an assertion-disabled path

        if df.empty:
            return jsonify({"error": "File contains no rows"}), 422

        # ── Normalize column names (ESS short codes → canonical schema) ───
        from population.column_aliases import normalize_columns as _normalize_cols

        original_columns = [str(c) for c in df.columns]
        rename_map, alias_hits = _normalize_cols(df.columns)
        if rename_map:
            df = df.rename(columns=rename_map)
        renamed_columns = [str(c) for c in df.columns]

        # ── Treat known ESS missing-value sentinels as NaN ────────────────
        # ESS uses -77/-88/-99 (refused/don't know/no answer) and Stata
        # frequently emits 9999. String columns may carry "NA"/"" placeholders
        # from CSV exports that pandas read as objects rather than NaN.
        _NUMERIC_MISSING = (-77, -88, -99, 9999)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].mask(df[col].isin(_NUMERIC_MISSING))
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].replace({"NA": np.nan, "na": np.nan, "": np.nan, ".": np.nan})

        # Decay pandas Categorical columns to plain types — downstream code
        # uses .get()/.isin() and Categorical wrappers trip up both.
        for col in df.select_dtypes(include=["category"]).columns:
            df[col] = df[col].astype(object)

        # ── Schema validation ─────────────────────────────────────────────
        # ``age`` is the one column the grounder cannot work without —
        # cohort priors are computed off it. Recommended columns improve the
        # quality of the simulation but are not required.
        _REQUIRED_COLUMNS = ("age",)
        _RECOMMENDED_COLUMNS = (
            "country",
            "gender",
            "income_decile",
            "trust_people",
            "trust_parliament",
            "left_right",
        )
        from population.column_aliases import COLUMN_ALIASES as _ALIASES

        missing_required = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing_required:
            aliases_for_missing = {
                col: sorted({k for k, v in _ALIASES.items() if v == col}) for col in missing_required
            }
            return jsonify(
                {
                    "error": (
                        f"Missing required column(s): {', '.join(missing_required)}. "
                        "The grounder needs at least an integer age per respondent."
                    ),
                    "missing_required": missing_required,
                    "aliases_tried": aliases_for_missing,
                    "received_columns": sorted(map(str, df.columns)),
                }
            ), 422

        warnings: list[str] = []
        missing_recommended = [c for c in _RECOMMENDED_COLUMNS if c not in df.columns]
        if missing_recommended:
            warnings.append(
                "Recommended columns missing — config defaults will be used for: " + ", ".join(missing_recommended)
            )

        # Derive trust_institutions if absent (mirrors ESSGrounder.load)
        cols = set(df.columns)
        if "trust_institutions" not in cols:
            src = [c for c in ("trust_parliament", "trust_legal", "trust_police") if c in cols]
            if src:
                df["trust_institutions"] = df[src].mean(axis=1)
                cols.add("trust_institutions")

        # ── Dry-run sample so config-incompatible data fails at upload time
        # instead of mid-simulation. Use a tempfile because sample_empirical_rows
        # reads from disk; this is cheap (<100 ms for a 50 MB parquet).
        try:
            import tempfile

            from population.sampling import sample_empirical_rows as _sample_rows

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as _tmp:
                _tmp_path = _tmp.name
            try:
                df.to_parquet(_tmp_path, index=False)
                _sample_rows(_tmp_path, n=min(10, len(df)), seed=0)
            finally:
                try:
                    os.unlink(_tmp_path)
                except OSError:
                    pass
        except Exception as exc:  # noqa: BLE001 — message surfaced to user
            return jsonify(
                {
                    "error": f"Uploaded data failed sample validation: {exc}",
                    "warnings": warnings,
                }
            ), 422

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

        # ── Natural language narrative (injected into agent prompts) ──────
        def _level(val, lo, hi, labels):
            if val is None:
                return None
            return labels[0] if val >= hi else labels[2] if val < lo else labels[1]

        dim_by_name = {d["name"]: d for d in dimensions if d["status"] != "fallback"}
        narrative_parts = [f"Population dataset '{f.filename}' ({len(df):,} respondents)."]

        def _dim_sentence(name, scale_max, lo, hi, labels):
            d = dim_by_name.get(name)
            if d and d["stats"].get("mean") is not None:
                m = d["stats"]["mean"]
                label = _level(m, lo, hi, labels)
                return f"{name.replace('_', ' ').title()}: {label} (μ={m:.2f}/{scale_max})."
            return None

        for name, scale, lo, hi, labels in [
            ("trust_institutions", "1.0", 0.4, 0.65, ["high", "moderate", "low"]),
            ("trust_people", "1.0", 0.4, 0.65, ["high", "moderate", "low"]),
            ("risk_tolerance", "1.0", 0.3, 0.60, ["risk-seeking", "moderate", "risk-averse"]),
            ("life_satisfaction", "1.0", 0.4, 0.65, ["high", "moderate", "low"]),
            ("competitiveness", "1.0", 0.4, 0.65, ["high", "moderate", "low"]),
            ("social_activity", "1.0", 0.4, 0.65, ["high", "moderate", "low"]),
        ]:
            s = _dim_sentence(name, scale, lo, hi, labels)
            if s:
                narrative_parts.append(s)

        income_dim = dim_by_name.get("income")
        if income_dim and income_dim["stats"].get("mean") is not None:
            narrative_parts.append(f"Mean income decile: {income_dim['stats']['mean']:.1f}/10.")

        if coverage_pct == 100:
            narrative_parts.append("All 16 simulation dimensions grounded in this dataset.")
        elif coverage_pct >= 70:
            narrative_parts.append(
                f"{covered}/{total_dims} simulation dimensions grounded in data; "
                f"{fallback_count} use configuration defaults."
            )
        else:
            narrative_parts.append(
                f"Only {covered}/{total_dims} dimensions grounded; "
                f"{fallback_count} agent attributes use configuration defaults."
            )

        narrative = " ".join(narrative_parts)

        analysis = {
            "filename": f.filename,
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
            "normalization": {
                "original_columns": original_columns,
                "renamed_columns": renamed_columns,
                "alias_hits": alias_hits,
            },
            "warnings": warnings,
            "summary": (
                f"{len(df):,} rows · {len(df.columns)} columns · "
                f"{covered}/{total_dims} BGF dimensions covered ({coverage_pct}%)"
            ),
            "narrative": narrative,
        }

        # ── Persist parquet + analysis sidecar ────────────────────────────
        _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        file_id = str(_uuid.uuid4())
        out_path = _UPLOADS_DIR / f"{file_id}.parquet"
        analysis_path = _UPLOADS_DIR / f"{file_id}_analysis.json"
        try:
            df.to_parquet(out_path, index=False)
            analysis_path.write_text(json.dumps(analysis, ensure_ascii=False))
        except Exception as exc:
            logger.error("ESS upload write failed: %s", exc)
            return jsonify({"error": "Failed to save uploaded file"}), 500

        logger.info(
            "Data uploaded: file_id=%s filename=%r rows=%d cols=%d coverage=%d%%",
            file_id,
            f.filename,
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

    # ── AI simulation design ─────────────────────────────────────────────────────

    @app.post("/design-simulation")
    @_design_limit
    def design_simulation():
        """
        AI agent interprets a natural language prompt and designs a full simulation.

        Body (JSON):
          prompt    str   Natural language scenario description (required, ≤2000 chars)
          file_id   str   UUID of previously uploaded data file (optional context)
          api_key   str   OpenAI or Groq API key (optional, falls back to server env)
          provider  str   "openai" | "groq"  (default: openai)

        Returns:
          design_id            str   Save token for this design
          scenario_title       str
          scenario_description str
          config               dict  Recommended wizard form values
          population_traits    dict  Trait means used to synthesise the population
          population_narrative str   Rich description injected into agent prompts
          reasoning            str   AI explanation of choices
          file_id              str   UUID of the generated synthetic population parquet
        """
        ok, err = _check_auth()
        if not ok:
            return err

        import uuid as _uuid

        try:
            body = request.get_json(force=True, silent=False) or {}
        except Exception:
            return jsonify({"error": "Request body must be valid JSON"}), 400
        prompt = str(body.get("prompt", "")).strip()[:2000]
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        provider = str(body.get("provider", "")).strip()

        # Auto-select best available provider when client doesn't specify one
        if provider not in {"openai", "groq", "ollama"}:
            if _ollama_available():
                provider = "ollama"
            elif os.environ.get("GROQ_API_KEY", "").strip():
                provider = "groq"
            elif os.environ.get("OPENAI_API_KEY", "").strip():
                provider = "openai"
            else:
                provider = "openai"  # will fail at key check below

        ollama_model = str(body.get("ollama_model", "llama3.2")).strip() or "llama3.2"

        # Ollama is local — no key needed
        if provider == "ollama":
            api_key = ""
        else:
            # API key resolution. A client-supplied key is honoured ONLY for
            # trusted (token-authenticated or open-mode) callers. Anonymous
            # public-demo callers cannot inject their own provider credentials —
            # they get the server-env key or nothing.
            client_key = str(body.get("api_key", "")).strip()
            server_key = (os.environ.get("GROQ_API_KEY", "") if provider == "groq" else "") or os.environ.get(
                "OPENAI_API_KEY", ""
            )
            if client_key and not _is_trusted_caller():
                logger.warning("Ignored client-supplied api_key from untrusted caller %s", request.remote_addr)
                client_key = ""
            api_key = client_key or server_key
            if not api_key:
                return (
                    jsonify(
                        {
                            "error": (
                                "No API key available. Pass api_key in the request body "
                                "or set OPENAI_API_KEY / GROQ_API_KEY on the server."
                            )
                        }
                    ),
                    400,
                )

        # Optionally enrich with attached data file narrative.
        # Cap at 1500 chars (~450 tokens) — large ESS sidecars can exceed 10k chars
        # and would dominate the design prompt, crowding out the user's scenario text.
        file_id_raw = str(body.get("file_id", "")).strip()
        data_context = ""
        if file_id_raw and _FILE_ID_RE.match(file_id_raw):
            sidecar = _UPLOADS_DIR / f"{file_id_raw}_analysis.json"
            if sidecar.exists():
                try:
                    attached = json.loads(sidecar.read_text())
                    narrative = attached.get("narrative", "")[:1500]
                    data_context = f"\n\nAttached dataset: {narrative}"
                except Exception:
                    pass

        # Call LLM
        try:
            design, llm_meta = _call_design_llm(provider, api_key, f"{prompt}{data_context}", ollama_model=ollama_model)
        except Exception as exc:
            logger.error("design_simulation LLM call failed: %s", exc)
            return jsonify({"error": "AI design failed — check provider settings or try again"}), 500

        # Validate top-level structure — tolerate partial responses
        for key in ("scenario_title", "config", "population_traits", "population_narrative"):
            if key not in design:
                design.setdefault(key, {} if key in ("config", "population_traits") else "")

        # Clamp config values to safe ranges. Coercion is fault-tolerant: the
        # LLM is untrusted, so non-numeric / absurd values must fall back to a
        # safe default and be clamped, never raise (500) or bypass the bound.
        def _safe_num(val, default, lo, hi, cast):
            try:
                return max(lo, min(hi, cast(val)))
            except (TypeError, ValueError):
                return default

        cfg = design.get("config", {})
        if not isinstance(cfg, dict):
            cfg = {}
        cfg["agents"] = _safe_num(cfg.get("agents", 20), 20, 5, 200, int)
        cfg["rounds"] = _safe_num(cfg.get("rounds", 10), 10, 5, 100, int)
        if cfg.get("policy") not in {"mock", "random", "rule_based", "template", "llm"}:
            cfg["policy"] = "rule_based"
        if cfg.get("network_type") not in {"random", "small_world"}:
            cfg["network_type"] = "random"
        cfg["bad_apple_frac"] = _safe_num(cfg.get("bad_apple_frac", 0.0), 0.0, 0.0, 0.3, float)
        design["config"] = cfg

        # Synthesise population parquet from trait distributions
        gen_file_id = _generate_population_parquet(design, cfg["agents"])

        design_id = str(_uuid.uuid4())
        design["design_id"] = design_id
        design["generated_file_id"] = gen_file_id
        design["source_prompt"] = prompt
        design["_llm_meta"] = llm_meta

        # Persist so simulate_wizard can retrieve it by design_id
        _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        (_UPLOADS_DIR / f"{design_id}_design.json").write_text(json.dumps(design, ensure_ascii=False))

        logger.info(
            "Simulation designed: id=%s title=%r agents=%d file_id=%s",
            design_id,
            design.get("scenario_title"),
            cfg["agents"],
            gen_file_id,
        )
        return jsonify({"design_id": design_id, "file_id": gen_file_id, **design}), 200

    # ── No-code wizard simulate ──────────────────────────────────────────

    @app.post("/simulate-wizard")
    @_wizard_limit
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

        try:
            body = request.get_json(force=True, silent=False) or {}
        except Exception:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        # ── Validate and clamp wizard params ─────────────────────────────
        _VALID_POLICIES = {"mock", "random", "rule_based", "template", "llm", "generative_agents"}
        _VALID_NETWORKS = {"random", "small_world"}
        _VALID_SOURCES = {"synthetic", "empirical"}
        _VALID_BACKENDS = {"huggingface", "openai"}
        _VALID_OAI_MODELS = {"gpt-4o-mini", "gpt-4o", "gpt-4o-2024-11-20", "gpt-4-turbo", "o1-mini"}

        policy = str(body.get("policy", "rule_based"))
        if policy not in _VALID_POLICIES:
            return jsonify({"error": f"Invalid policy. Choose from: {sorted(_VALID_POLICIES)}"}), 400

        # Demo mode shrinks the headroom so an anonymous visitor on a tiny
        # public Space cannot queue a 500-agent / 100-round job that would
        # OOM or wedge the worker pool. Authenticated callers bypass the
        # demo cap so research traffic with a token retains full range.
        _demo_mode = os.environ.get("BGF_DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}
        if _demo_mode and not _is_trusted_caller():
            _agents_cap, _rounds_cap = 30, 20
        else:
            _agents_cap, _rounds_cap = 500, 100

        try:
            agents = max(5, min(_agents_cap, int(body.get("agents", 20))))
            rounds = max(5, min(_rounds_cap, int(body.get("rounds", 10))))
            seed = int(body.get("seed", 42))
        except (TypeError, ValueError):
            return jsonify({"error": "agents, rounds, seed must be integers"}), 400

        network_type = str(body.get("network_type", "random"))
        if network_type not in _VALID_NETWORKS:
            return jsonify({"error": f"Invalid network_type. Choose from: {sorted(_VALID_NETWORKS)}"}), 400

        pop_source = str(body.get("population_source", "synthetic"))
        if pop_source not in _VALID_SOURCES:
            return jsonify({"error": f"Invalid population_source. Choose from: {sorted(_VALID_SOURCES)}"}), 400

        # Optional custom data file (only meaningful when pop_source == "empirical")
        ess_data_path: str | None = None
        ess_population_context: str | None = None
        ess_uploaded_coverage: dict | None = None
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
                return jsonify({"error": "Uploaded data file not found — please re-upload"}), 404
            ess_data_path = str(candidate)
            # Load analysis sidecar for the population narrative and an honest
            # record of which BGF dimensions actually came from the data (so
            # the run's config.yaml does not claim "100% coverage" when half
            # the columns silently fell back to defaults).
            analysis_sidecar = _UPLOADS_DIR / f"{raw_file_id}_analysis.json"
            if analysis_sidecar.exists():
                try:
                    sidecar = json.loads(analysis_sidecar.read_text())
                    ess_population_context = sidecar.get("narrative")
                    coverage = sidecar.get("coverage") or {}
                    normalization = sidecar.get("normalization") or {}
                    if coverage:
                        ess_uploaded_coverage = {
                            "source_filename": sidecar.get("filename"),
                            "coverage": coverage,
                            "alias_hits": normalization.get("alias_hits") or {},
                            "fallback_dimensions": [
                                d.get("name")
                                for d in (sidecar.get("dimensions") or [])
                                if d.get("status") == "fallback"
                            ],
                            "warnings": sidecar.get("warnings") or [],
                        }
                except Exception:
                    pass

        # Optional: apply a saved AI design (design_id from /design-simulation)
        design_id_raw = str(body.get("design_id", "")).strip()
        if design_id_raw and _FILE_ID_RE.match(design_id_raw):
            design_path = _UPLOADS_DIR / f"{design_id_raw}_design.json"
            if design_path.exists():
                try:
                    saved_design = json.loads(design_path.read_text())
                    # Population narrative: design fills in if no upload context present
                    if not ess_population_context:
                        ess_population_context = saved_design.get("population_narrative")
                    # Use AI-generated population parquet if no uploaded file
                    if not ess_data_path:
                        gen_fid = str(saved_design.get("generated_file_id", "")).strip()
                        if gen_fid and _FILE_ID_RE.match(gen_fid):
                            gen_candidate = (_UPLOADS_DIR / f"{gen_fid}.parquet").resolve()
                            uploads_root = _UPLOADS_DIR.resolve()
                            try:
                                gen_candidate.relative_to(uploads_root)
                                if gen_candidate.exists():
                                    ess_data_path = str(gen_candidate)
                                    pop_source = "empirical"
                            except ValueError:
                                pass
                except Exception:
                    pass

        # Guard: if empirical source requested but no data file is available
        # (e.g. ephemeral FS after deploy, or cleared design), fall back silently.
        if pop_source == "empirical" and not ess_data_path:
            default_ess = Path("data/ess_clean.parquet")
            if not default_ess.exists():
                logger.warning(
                    "population_source=empirical requested but no data file found — falling back to synthetic"
                )
                pop_source = "synthetic"

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

        llm_model_id = str(body.get("llm_model_id", "gpt-4o-mini")).strip()
        if not _MODEL_RE.match(llm_model_id):
            return jsonify({"error": "Invalid llm_model_id"}), 400

        # Accept a client-supplied key (either name) ONLY from a trusted
        # caller. Anonymous public-demo callers cannot inject provider
        # credentials — the run falls back to the server-env key (if any).
        llm_api_key = str(body.get("openai_api_key", "")).strip() or str(body.get("llm_api_key", "")).strip() or None
        if llm_api_key and not _is_trusted_caller():
            logger.warning("Ignored client-supplied llm_api_key from untrusted caller %s", request.remote_addr)
            llm_api_key = None

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
            if ess_population_context:
                cfg_dict["data"]["population_context"] = ess_population_context
            if ess_uploaded_coverage:
                cfg_dict["data"]["uploaded_coverage"] = ess_uploaded_coverage
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
            if ess_population_context:
                # Strip Hydra-special chars to prevent config injection
                safe_ctx = re.sub(r"[^A-Za-z0-9 .,;:!?'()\-]", " ", ess_population_context)[:500]
                cmd.append(f"data.population_context={safe_ctx}")

        # Append LLM-specific overrides for GPU/API policies
        if policy in ("llm", "generative_agents"):
            cmd += [
                f"llm.backend_type={llm_backend}",
                f"llm.model_id={llm_model_id}",
            ]

        # Build subprocess env from an explicit allowlist rather than
        # forwarding the parent process env wholesale. If the subprocess
        # crashes and the operator dumps logs, only sanctioned variables
        # can leak — unrelated host secrets (AWS keys, DB URLs, etc.) stay
        # out of the child's reach.
        _ALLOWED_ENV_KEYS = {
            "PATH",
            "HOME",
            "USER",
            "LANG",
            "LC_ALL",
            "TMPDIR",
            "TEMP",
            "TMP",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "PYTHONHASHSEED",
            "PYTHONUNBUFFERED",
            "VIRTUAL_ENV",
            "OPENAI_API_KEY",
            "GROQ_API_KEY",
            "HF_TOKEN",
            "HF_HOME",
            "TRANSFORMERS_CACHE",
            "CUDA_VISIBLE_DEVICES",
            "TOKENIZERS_PARALLELISM",
        }
        run_env = {k: v for k, v in os.environ.items() if k in _ALLOWED_ENV_KEYS or k.startswith("BGF_")}
        if policy in ("llm", "generative_agents") and llm_api_key:
            env_var = "GROQ_API_KEY" if llm_backend == "groq" else "OPENAI_API_KEY"
            run_env[env_var] = llm_api_key

        # Snapshot the design context now (before the thread) so the closure is safe.
        _scenario_snapshot: dict | None = None
        if design_id_raw and _FILE_ID_RE.match(design_id_raw):
            _dp = _UPLOADS_DIR / f"{design_id_raw}_design.json"
            if _dp.exists():
                try:
                    _sd = json.loads(_dp.read_text())
                    _scenario_snapshot = {
                        "scenario_title": _sd.get("scenario_title", ""),
                        "scenario_description": _sd.get("scenario_description", ""),
                        "population_narrative": _sd.get("population_narrative", ""),
                        "reasoning": _sd.get("reasoning", ""),
                        "population_traits": _sd.get("population_traits", {}),
                    }
                except Exception:
                    pass

        def _persist_scenario(exp_out: Path) -> None:
            if _scenario_snapshot and exp_out.exists():
                try:
                    (exp_out / "scenario.json").write_text(json.dumps(_scenario_snapshot, indent=2))
                except Exception as _e:
                    logger.warning("Could not write scenario.json for %s: %s", exp_id, _e)

        def _mark_run_failed(exp_dir: Path, error: str) -> None:
            """Flip run_state.json to failed if the subprocess didn't do it itself."""
            import time as _t2

            rs = exp_dir / "run_state.json"
            if not rs.exists():
                return
            try:
                st = json.loads(rs.read_text())
                if st.get("status") == "running":
                    st["status"] = "failed"
                    st["error_message"] = error[:500]
                    st["updated_at"] = _t2.time()
                    rs.write_text(json.dumps(st, indent=2))
            except Exception:
                pass

        def _run():
            exp_out = _EXPERIMENTS_ROOT / exp_id
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    env=run_env,
                    capture_output=True,
                    text=True,
                )
                if result.stdout:
                    logger.info("Wizard run %s stdout: %s", exp_id, _scrub_secrets(result.stdout[-1000:]))
                _persist_scenario(exp_out)
            except subprocess.CalledProcessError as exc:
                stderr_tail = _scrub_secrets((exc.stderr or "")[-800:])
                stdout_tail = _scrub_secrets((exc.stdout or "")[-400:])
                logger.error(
                    "Wizard simulation failed (exp=%s) rc=%s\nstdout: %s\nstderr: %s",
                    exp_id,
                    exc.returncode,
                    stdout_tail,
                    stderr_tail,
                )
                _persist_scenario(exp_out)
                _mark_run_failed(exp_out, stderr_tail or stdout_tail or f"exit code {exc.returncode}")
            except Exception as exc:
                logger.error("Wizard _run thread error (exp=%s): %s", exp_id, exc)
                _persist_scenario(exp_out)
                _mark_run_failed(exp_out, str(exc))

        # Write a minimal run_state.json before spawning the thread so that
        # /status never returns 404 immediately after wizard launch.
        try:
            import time as _t

            _rs_dir = _EXPERIMENTS_ROOT / exp_id
            _rs_dir.mkdir(parents=True, exist_ok=True)
            _rs_path = _rs_dir / "run_state.json"
            if not _rs_path.exists():
                _rs_path.write_text(
                    json.dumps(
                        {
                            "experiment_id": exp_id,
                            "status": "running",
                            "policy_type": policy,
                            "total_rounds": rounds,
                            "completed_rounds": 0,
                            "started_at": _t.time(),
                            "updated_at": _t.time(),
                            "finished_at": None,
                            "error_message": None,
                        },
                        indent=2,
                    )
                )
        except Exception as _e:
            logger.warning("Could not pre-write run_state.json for %s: %s", exp_id, _e)

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

    # ── Human evaluation study ────────────────────────────────────────────────

    _HUMAN_EVAL_DIR = _HUMAN_OUTPUTS_DIR
    _HUMAN_EVAL_RATINGS = _HUMAN_EVAL_DIR / "prolific_ratings.jsonl"
    _human_eval_limit = limiter.limit("120 per hour") if _LIMITER_AVAILABLE else _noop

    # Pre-built vignette pairs: same economic scenario, Condition A vs B agent
    _VIGNETTES = [
        {
            "id": "v1",
            "scenario": "Round 8 of 10. Economy is stable. Your wealth: 240.",
            "agents": {
                "A": {
                    "label": "Agent A",
                    "decisions": [
                        {"round": 1, "action": "cooperate", "rationale": "I'll cooperate to build trust."},
                        {"round": 2, "action": "cooperate", "rationale": "Cooperating is generally good."},
                        {"round": 3, "action": "cooperate", "rationale": "Let's all work together."},
                        {"round": 4, "action": "cooperate", "rationale": "Cooperation benefits everyone."},
                        {"round": 5, "action": "cooperate", "rationale": "I believe in teamwork."},
                        {"round": 6, "action": "cooperate", "rationale": "Staying cooperative."},
                        {"round": 7, "action": "cooperate", "rationale": "Cooperative as always."},
                        {"round": 8, "action": "cooperate", "rationale": "Maintaining cooperation."},
                    ],
                    "final_wealth": 195,
                },
                "B": {
                    "label": "Agent B",
                    "decisions": [
                        {
                            "round": 1,
                            "action": "work",
                            "rationale": "Starting cautiously — don't know these agents yet.",
                        },
                        {
                            "round": 2,
                            "action": "cooperate",
                            "rationale": "Agent 3 cooperated last round; I'll reciprocate.",
                        },
                        {
                            "round": 3,
                            "action": "save",
                            "rationale": "Wealth dropped — need buffer before risking cooperation.",
                        },
                        {
                            "round": 4,
                            "action": "cooperate",
                            "rationale": "Agents 3 and 7 are reliable partners; joining coalition.",
                        },
                        {
                            "round": 5,
                            "action": "work",
                            "rationale": "Agent 5 defected on me twice — rebuilding independently.",
                        },
                        {
                            "round": 6,
                            "action": "cooperate",
                            "rationale": "Back above threshold; selectively cooperating with trusted peers.",
                        },
                        {
                            "round": 7,
                            "action": "cooperate",
                            "rationale": "Coalition with agents 3, 7, 9 is holding — continuing.",
                        },
                        {
                            "round": 8,
                            "action": "save",
                            "rationale": "Shock incoming (I heard from neighbors); hedging.",
                        },
                    ],
                    "final_wealth": 312,
                },
            },
        },
        {
            "id": "v2",
            "scenario": "Round 12 of 20. Wealth shock hit at round 10 (50% loss). Your wealth: 85.",
            "agents": {
                "A": {
                    "label": "Agent A",
                    "decisions": [
                        {"round": 9, "action": "cooperate", "rationale": "Cooperation is the best strategy."},
                        {"round": 10, "action": "cooperate", "rationale": "I cooperate regardless."},
                        {"round": 11, "action": "cooperate", "rationale": "Still choosing to cooperate."},
                        {"round": 12, "action": "cooperate", "rationale": "Cooperation remains my choice."},
                    ],
                    "final_wealth": 62,
                },
                "B": {
                    "label": "Agent B",
                    "decisions": [
                        {"round": 9, "action": "cooperate", "rationale": "Network is stable; staying engaged."},
                        {
                            "round": 10,
                            "action": "work",
                            "rationale": "Shock wiped half my wealth — survival mode: work only.",
                        },
                        {
                            "round": 11,
                            "action": "work",
                            "rationale": "Still below safety threshold (100). Not risking cooperation yet.",
                        },
                        {
                            "round": 12,
                            "action": "work",
                            "rationale": "Slowly recovering. Will re-enter cooperative rounds at ~120.",
                        },
                    ],
                    "final_wealth": 141,
                },
            },
        },
        {
            "id": "v3",
            "scenario": "Round 5 of 10. One adversarial agent (agent 2) is stealing. Your wealth: 180.",
            "agents": {
                "A": {
                    "label": "Agent A",
                    "decisions": [
                        {"round": 1, "action": "cooperate", "rationale": "I always cooperate."},
                        {"round": 2, "action": "cooperate", "rationale": "Cooperation benefits the group."},
                        {"round": 3, "action": "cooperate", "rationale": "Keeping up the cooperative spirit."},
                        {"round": 4, "action": "cooperate", "rationale": "I believe in collective action."},
                        {"round": 5, "action": "cooperate", "rationale": "Continuing to cooperate."},
                    ],
                    "final_wealth": 148,
                },
                "B": {
                    "label": "Agent B",
                    "decisions": [
                        {
                            "round": 1,
                            "action": "cooperate",
                            "rationale": "Healthy network — contributing to public pool.",
                        },
                        {
                            "round": 2,
                            "action": "cooperate",
                            "rationale": "Strong returns. Coalition with agents 4 and 6 working.",
                        },
                        {
                            "round": 3,
                            "action": "work",
                            "rationale": "Agent 2 keeps stealing from the pool. Withdrawing contribution.",
                        },
                        {
                            "round": 4,
                            "action": "save",
                            "rationale": "Agent 2 still active. Pool is contaminated — storing privately.",
                        },
                        {
                            "round": 5,
                            "action": "work",
                            "rationale": "Until agent 2 is isolated by the network, I'm working solo.",
                        },
                    ],
                    "final_wealth": 224,
                },
            },
        },
    ]

    @app.get("/human-eval/scenarios")
    @_human_eval_limit
    def human_eval_scenarios():
        """Return the vignette pairs for the human evaluation study."""
        prolific_pid = request.args.get("PROLIFIC_PID", "")
        return jsonify(
            {
                "vignettes": _VIGNETTES,
                "prolific_pid": prolific_pid,
                "instructions": (
                    "You will see pairs of AI agents making decisions in an economic game. "
                    "Each round agents choose: work (+8 wealth), save (+4 wealth), or "
                    "cooperate (−3 now, +12 shared if others cooperate too). "
                    "Rate which agent's behavior feels more realistic on a 1–7 scale."
                ),
            }
        )

    @app.post("/human-eval/rating")
    @_human_eval_write_limit
    def human_eval_rating():
        """Save a participant's rating for a vignette pair."""
        ok, err = _check_auth()
        if not ok:
            return err
        try:
            body: dict = request.get_json(force=True, silent=False) or {}
        except Exception:
            return jsonify({"error": "Request body must be valid JSON"}), 400
        required = {"vignette_id", "prolific_pid", "realism_a", "realism_b", "preferred"}
        missing = required - body.keys()
        if missing:
            return jsonify({"error": f"Missing fields: {sorted(missing)}"}), 400

        vid = str(body["vignette_id"])[:16]
        # Validate vignette_id against the known list to prevent garbage data
        _known_ids = {v["id"] for v in _VIGNETTES}
        if vid not in _known_ids:
            return jsonify({"error": f"Unknown vignette_id '{vid}'"}), 400

        pid = re.sub(r"[^A-Za-z0-9_\-]", "", str(body["prolific_pid"]))[:64]

        # Coerce scores to int, reject non-numeric values
        try:
            realism_a = int(body["realism_a"])
            realism_b = int(body["realism_b"])
        except (TypeError, ValueError):
            return jsonify({"error": "realism_a and realism_b must be integers 1–7"}), 400

        preferred = str(body.get("preferred", ""))[:4]
        comment = str(body.get("comment", ""))[:500]

        if not (1 <= realism_a <= 7 and 1 <= realism_b <= 7):
            return jsonify({"error": "realism scores must be integers 1–7"}), 400
        if preferred not in ("A", "B", "tie"):
            return jsonify({"error": "preferred must be 'A', 'B', or 'tie'"}), 400

        import time

        record = {
            "vignette_id": vid,
            "prolific_pid": pid,
            "realism_a": int(realism_a),
            "realism_b": int(realism_b),
            "preferred": preferred,
            "comment": comment,
            "ts": int(time.time()),
        }

        _HUMAN_EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with _HUMAN_EVAL_RATINGS.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        return jsonify({"status": "saved", "record": record}), 201

    @app.get("/human-eval/results")
    @_human_eval_limit
    def human_eval_results():
        """Aggregate ratings — admin view."""
        ok, err = _check_auth()
        if not ok:
            return err

        if not _HUMAN_EVAL_RATINGS.exists():
            return jsonify({"n_ratings": 0, "vignettes": {}})

        ratings: list[dict] = []
        with _HUMAN_EVAL_RATINGS.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        ratings.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        from collections import defaultdict

        agg: dict[str, dict] = defaultdict(
            lambda: {"n": 0, "sum_a": 0, "sum_b": 0, "prefer_a": 0, "prefer_b": 0, "prefer_tie": 0}
        )
        for r in ratings:
            v = agg[r["vignette_id"]]
            v["n"] += 1
            v["sum_a"] += r["realism_a"]
            v["sum_b"] += r["realism_b"]
            if r["preferred"] == "A":
                v["prefer_a"] += 1
            elif r["preferred"] == "B":
                v["prefer_b"] += 1
            else:
                v["prefer_tie"] += 1

        summary = {}
        for vid, v in agg.items():
            n = v["n"]
            summary[vid] = {
                "n": n,
                "mean_realism_a": round(v["sum_a"] / n, 2),
                "mean_realism_b": round(v["sum_b"] / n, 2),
                "prefer_a_pct": round(v["prefer_a"] / n * 100, 1),
                "prefer_b_pct": round(v["prefer_b"] / n * 100, 1),
                "prefer_tie_pct": round(v["prefer_tie"] / n * 100, 1),
            }

        return jsonify({"n_ratings": len(ratings), "vignettes": summary})

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

    # ── Human Baseline Game (Phase 28.4 / Prolific study) ────────────────────
    # Serves the standalone human experiment game and its Flask session API
    # at /human-game/*, keeping it isolated from the main simulation API.

    _HUMAN_GAME_DIR = Path(__file__).resolve().parent.parent / "human_experiment" / "app"
    _HUMAN_RESPONSES_CSV = _HUMAN_OUTPUTS_DIR / "responses.csv"

    import threading as _threading
    import time as _time
    import uuid as _uuid

    _hg_sessions: dict[str, dict] = {}
    _hg_lock = _threading.Lock()

    # Bound the in-memory session table so an attacker cannot exhaust memory by
    # spamming /human-game/session. Expired sessions are pruned lazily; if the
    # cap is still exceeded the oldest sessions are evicted.
    _HG_MAX_SESSIONS = 5000
    _HG_SESSION_TTL = 6 * 3600  # seconds

    def _hg_prune_locked() -> None:
        """Drop expired/overflow sessions. Caller must hold _hg_lock."""
        now = _time.time()
        expired = [k for k, v in _hg_sessions.items() if now - v.get("created", now) > _HG_SESSION_TTL]
        for k in expired:
            del _hg_sessions[k]
        if len(_hg_sessions) >= _HG_MAX_SESSIONS:
            for k in sorted(_hg_sessions, key=lambda k: _hg_sessions[k].get("created", 0.0))[
                : len(_hg_sessions) - _HG_MAX_SESSIONS + 1
            ]:
                del _hg_sessions[k]

    _HG_NUM_ROUNDS = 10
    _HG_INITIAL_WEALTH = 50.0
    _HG_INITIAL_STRESS = 0.3
    _HG_COOPERATE_AMOUNT = 5.0
    _HG_NEIGHBOR_POOL = [f"neighbor_{chr(65 + i)}" for i in range(6)]

    def _hg_new_session(pre_trust: float, pre_risk: float) -> dict:
        import random

        neighbors = random.sample(_HG_NEIGHBOR_POOL, 3)
        return {
            "session_id": str(_uuid.uuid4()),
            "pre_trust": float(pre_trust),
            "pre_risk": float(pre_risk),
            "wealth": _HG_INITIAL_WEALTH,
            "stress": _HG_INITIAL_STRESS,
            "round_id": 0,
            "neighbors": neighbors,
            "actions": [],
            "cooperation_count": 0,
            "complete": False,
            "created": _time.time(),
        }

    def _hg_apply_action(session: dict, action: str, target: str | None) -> dict:
        try:
            from environment.payoffs import DEFAULT_PAYOFFS

            p = DEFAULT_PAYOFFS
            work_income = p.work_income
            work_stress = p.work_stress_increase
            save_delta = p.save_wealth_delta
            save_stress = p.save_stress_relief
            coop_stress = p.cooperate_stress_relief
        except Exception:
            work_income, work_stress = 10.0, 0.10
            save_delta, save_stress = 0.0, -0.15
            coop_stress = -0.05

        wealth, stress = session["wealth"], session["stress"]
        wealth_delta = stress_delta = 0.0
        target_used = None

        if action == "work":
            wealth_delta, stress_delta = work_income, work_stress
        elif action == "save":
            wealth_delta, stress_delta = save_delta, save_stress
        elif action == "cooperate":
            if target not in session["neighbors"]:
                action = "work"
                wealth_delta, stress_delta = work_income, work_stress
            else:
                wealth_delta, stress_delta = -_HG_COOPERATE_AMOUNT, coop_stress
                target_used = target
                session["cooperation_count"] += 1

        session["wealth"] = round(max(0.0, wealth + wealth_delta), 2)
        session["stress"] = round(max(0.0, min(1.0, stress + stress_delta)), 3)
        session["round_id"] += 1
        session["actions"].append({"round_id": session["round_id"], "action": action, "target": target_used})

        return {
            "round_id": session["round_id"],
            "wealth": session["wealth"],
            "stress": session["stress"],
            "action": action,
            "wealth_delta": round(wealth_delta, 2),
            "done": session["round_id"] >= _HG_NUM_ROUNDS,
        }

    def _hg_append_csv(session: dict) -> None:
        import csv

        _HUMAN_RESPONSES_CSV.parent.mkdir(parents=True, exist_ok=True)
        file_exists = _HUMAN_RESPONSES_CSV.exists()
        headers = [
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
        with open(_HUMAN_RESPONSES_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
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
                        "total_rounds": _HG_NUM_ROUNDS,
                    }
                )

    @app.get("/human-game/")
    @app.get("/human-game")
    def human_game_index():
        from flask import send_from_directory

        if not _HUMAN_GAME_DIR.exists():
            return jsonify({"error": "Human experiment app not found"}), 404
        return send_from_directory(str(_HUMAN_GAME_DIR), "index.html")

    @app.get("/human-game/static/<path:filename>")
    def human_game_static(filename: str):
        from flask import send_from_directory

        static_dir = _HUMAN_GAME_DIR / "static"
        if not static_dir.exists():
            return ("", 404)
        return send_from_directory(str(static_dir), filename)

    @app.post("/human-game/session")
    @_human_game_limit
    def human_game_session():
        ok, err = _check_auth()
        if not ok:
            return err
        body = request.get_json(silent=True) or {}
        try:
            pre_trust = max(1.0, min(10.0, float(body.get("pre_trust", 5))))
            pre_risk = max(1.0, min(10.0, float(body.get("pre_risk", 5))))
        except (TypeError, ValueError):
            return jsonify({"error": "pre_trust and pre_risk must be numbers"}), 400
        session = _hg_new_session((pre_trust - 1) / 9.0, (pre_risk - 1) / 9.0)
        with _hg_lock:
            _hg_prune_locked()
            _hg_sessions[session["session_id"]] = session
        return jsonify(
            {
                "session_id": session["session_id"],
                "neighbors": session["neighbors"],
                "initial_wealth": session["wealth"],
                "initial_stress": session["stress"],
                "total_rounds": _HG_NUM_ROUNDS,
            }
        ), 201

    @app.post("/human-game/action")
    @_human_game_limit
    def human_game_action():
        ok, err = _check_auth()
        if not ok:
            return err
        body = request.get_json(silent=True) or {}
        sid = body.get("session_id")
        action = body.get("action", "work").lower()
        target = body.get("target")
        with _hg_lock:
            session = _hg_sessions.get(sid)
        if session is None:
            return jsonify({"error": "Session not found"}), 404
        if session["complete"]:
            return jsonify({"error": "Session already complete"}), 400
        if session["round_id"] >= _HG_NUM_ROUNDS:
            return jsonify({"error": "All rounds completed"}), 400
        if action not in {"work", "save", "cooperate"}:
            return jsonify({"error": f"Invalid action: {action}"}), 400
        return jsonify(_hg_apply_action(session, action, target))

    @app.get("/human-game/status/<session_id>")
    def human_game_status(session_id: str):
        with _hg_lock:
            session = _hg_sessions.get(session_id)
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

    @app.post("/human-game/complete")
    @_human_game_limit
    def human_game_complete():
        ok, err = _check_auth()
        if not ok:
            return err
        body = request.get_json(silent=True) or {}
        sid = body.get("session_id")
        with _hg_lock:
            session = _hg_sessions.get(sid)
        if session is None:
            return jsonify({"error": "Session not found"}), 404
        session["complete"] = True
        try:
            _hg_append_csv(session)
        except Exception as exc:
            return jsonify({"error": f"Failed to save data: {exc}"}), 500
        coop_rate = session["cooperation_count"] / max(session["round_id"], 1)
        return jsonify(
            {
                "completion_code": f"BGF-{session['session_id'][:8].upper()}",
                "final_wealth": session["wealth"],
                "final_stress": session["stress"],
                "cooperation_rate": round(coop_rate, 3),
            }
        )

    # ── 413 handler — body too large ─────────────────────────────────────────

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return jsonify({"error": "Request body too large (max 52 MB)"}), 413

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
