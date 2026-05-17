# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| main    | Yes       |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report security issues by emailing the maintainer directly (see `CITATION.cff`).
Include a description of the issue, steps to reproduce, and potential impact.
You should receive a response within 72 hours.

## Known Security Considerations

### REST API (`api/app.py`)

The BGF REST API is designed for **research use on trusted networks** (localhost or
private LANs). Before exposing it on the public internet:

1. **Set `BGF_API_TOKEN`** to a strong random token (see `.env.example`).
   When set, write/compute `POST`/`PUT`/`DELETE` endpoints require
   `Authorization: Bearer <token>`. **Exceptions** (public by design, bounded by
   per-IP rate limiting instead of the token): all `GET`/`HEAD`/`OPTIONS`, and
   the demo POST routes in `_PUBLIC_POST_PATHS` — `/design-simulation`,
   `/simulate-wizard`, `/human-eval/rating`,
   `/human-game/{session,action,complete}`.
   Generate a token: `python -c "import secrets; print(secrets.token_hex(32))"`

2. **Run behind a reverse proxy** (nginx / Caddy) with TLS — the Flask dev server
   does not support HTTPS.

3. **Restrict CORS origins** via `BGF_CORS_ORIGINS` (comma-separated) or by
   passing `allowed_origins=[...]` to `create_app()`. When neither is set CORS
   defaults to the public Space origin plus localhost dev ports — **not** `*`.
   Set this explicitly for any other deployment.

4. **Do not run with `--debug`** in production — Flask debug mode enables the
   interactive debugger, which allows arbitrary code execution.

### `trust_remote_code` in `LLMBackend`

`LLMBackend` accepts an `allow_remote_code=True` parameter which maps to
HuggingFace's `trust_remote_code=True`. This flag permits arbitrary Python
code in the model repository to execute during model loading.

**Only enable this for model checkpoints you control or have audited.**
The default is `False` (disabled).

### API Key Handling

API keys (`OPENAI_API_KEY`, `BGF_REPORT_API_KEY`) must be supplied via environment
variables, never as URL query parameters or in committed config files.
The `/report` endpoint reads the key from the server environment exclusively.

Client-supplied keys (`api_key` / `openai_api_key` / `llm_api_key` in the body
of `/design-simulation` and `/simulate-wizard`) are honoured **only for trusted
callers** — i.e. open mode, or a request bearing the valid `BGF_API_TOKEN`
(`_is_trusted_caller()`). An anonymous public-demo caller cannot inject
provider credentials; such requests fall back to the server-env key (or fail
cleanly if none is set). This keeps the public endpoints from being used as an
open relay for arbitrary third-party API keys.

Persisted run snapshots (`experiments/<id>/config.yaml`) are scrubbed by
`utils.io.redact_secrets()` before write, and subprocess stdout/stderr is passed
through `_scrub_secrets()` before logging, so credentials cannot leak via the
public `GET /results` / `/status` endpoints or server logs.

### Rate Limiting & Resource Bounds

`flask-limiter` enforces per-IP limits on every write/compute endpoint
(`flask-limiter` is a hard dependency in `requirements*.txt`). The
unauthenticated paid-LLM endpoint `/design-simulation` and the
subprocess-spawning `/simulate-wizard` each carry a dedicated stricter cap
(`/simulate-wizard`: 3/min, 20/hour, 80/day). Simulation size is bounded in
`configs/schema.py`
(`rounds ≤ 200`, `population_size ≤ 1000`) so an oversized uploaded or
LLM-designed config cannot exhaust compute/disk. The in-memory human-game
session table is capped and TTL-pruned to prevent memory-exhaustion DoS.

### Config Path Sandboxing

The `/simulate` endpoint restricts `config_path` to the `configs/` directory,
preventing API callers from pointing simulations at arbitrary YAML files.
