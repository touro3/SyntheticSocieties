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
   All endpoints require `Authorization: Bearer <token>` when this is set.
   Generate a token: `python -c "import secrets; print(secrets.token_hex(32))"`

2. **Run behind a reverse proxy** (nginx / Caddy) with TLS — the Flask dev server
   does not support HTTPS.

3. **Restrict CORS origins** by passing `allowed_origins=["https://yourdomain"]`
   to `create_app()`.

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

### Config Path Sandboxing

The `/simulate` endpoint restricts `config_path` to the `configs/` directory,
preventing API callers from pointing simulations at arbitrary YAML files.
