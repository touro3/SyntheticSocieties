"""Deploy the repo to a Hugging Face Docker Space.

HF Spaces reject plain binary blobs pushed over git; large analysis figures
and past experiment runs are also not needed at runtime. This uploads only the
build context the Dockerfile needs, letting huggingface_hub handle LFS/Xet for
the few binary data files (ESS parquet distributions) automatically.

Env vars:
  HF_TOKEN     write token (required)
  HF_SPACE_ID  "<user>/<space>", e.g. touro3/synthetic-societies (required)
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

# Secrets the deployed Space needs to be useful. Printed as a reminder after a
# successful upload because Spaces do not read them from the repo — the
# operator has to configure them via Settings → Variables and secrets.
_SUGGESTED_SECRETS = [
    ("BGF_API_TOKEN", "recommended", "Bearer token gating POST endpoints"),
    ("OPENAI_API_KEY", "for LLM features", "interview/anchor synthesis, /design-simulation, /report"),
    ("GROQ_API_KEY", "optional", "cheaper LLM fallback for design"),
    ("BGF_DATA_ROOT", "for persistence", "set to /data when persistent storage is enabled"),
    ("BGF_CORS_ORIGINS", "optional", "comma-separated allowed origins (auto-derived if unset)"),
    ("BGF_DEMO_MODE", "public demos", "true caps wizard runs at 30 agents / 20 rounds"),
]

# Excluded from the Space (and thus the image): heavy artifacts not needed to
# serve the app. The app recreates experiments/ at runtime; figures are
# regenerated offline. Mirrors .dockerignore plus the runtime-unneeded dirs.
IGNORE = [
    ".git*",
    ".git/**",
    "venv/**",
    "node_modules/**",
    "frontend/node_modules/**",
    "**/__pycache__/**",
    "*.pyc",
    # User-write paths: rerouted at runtime via BGF_DATA_ROOT. The local
    # copies are dead weight in the build context and may contain
    # researcher uploads / participant data we don't want in a public
    # Space. The running app recreates these dirs on demand.
    "experiments/**",
    "uploads/**",
    # NB: tracker/ also contains analytics.py (imported by api/app.py for
    # /regressions), so ignore only the data file, not the whole dir.
    "tracker/*.parquet",
    "human_outputs/**",
    "data/human/**",
    "data/bgf_human_responses*",
    # Analysis outputs: regenerated offline from experiments/, not needed
    # at serve time.
    "analysis/figures/**",
    "analysis/networks/**",
    "analysis/tables/**",
    "analysis/reports/**",
    # Heavy artifacts and caches.
    "*.pdf",
    "notebooks/**",
    "paper/**",
    "llama.cpp/**",
    "graphify-out/**",
    "*.gguf",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "hf_cache/**",
    ".cache/**",
    "logs/**",
    "*.log",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".mypy_cache/**",
    ".env",
    ".env.*",
]


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    space_id = os.environ.get("HF_SPACE_ID")
    if not token or not space_id:
        print("HF_TOKEN and HF_SPACE_ID must be set", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent.parent

    # Pre-flight: the Vue SPA must be pre-built into api/static/ because the
    # Dockerfile does not rebuild it. Shipping a Space without the SPA would
    # serve only the JSON API — visitors see a blank page.
    spa_index = repo_root / "api" / "static" / "index.html"
    if not spa_index.exists():
        print(
            f"ERROR: missing built SPA at {spa_index.relative_to(repo_root)}.\n"
            "Run `npm --prefix frontend install && npm --prefix frontend run build`\n"
            "before deploying, or the Space will serve only the JSON API.",
            file=sys.stderr,
        )
        return 2

    # Pre-flight: confirm the Dockerfile is present (HF Spaces with sdk=docker
    # requires it at the repo root).
    if not (repo_root / "Dockerfile").exists():
        print("ERROR: Dockerfile missing at repo root.", file=sys.stderr)
        return 3

    api = HfApi(token=token)
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=False,
    )
    api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(repo_root),
        ignore_patterns=IGNORE,
        commit_message="Deploy from CI",
    )

    space_url = f"https://huggingface.co/spaces/{space_id}"
    settings_url = f"{space_url}/settings"
    print(f"\nDeployed to {space_url}")
    print(f"\nConfigure secrets at: {settings_url}")
    print("Recommended variables (Settings → Variables and secrets):")
    width = max(len(name) for name, _, _ in _SUGGESTED_SECRETS)
    for name, when, purpose in _SUGGESTED_SECRETS:
        print(f"  {name.ljust(width)}  [{when}] — {purpose}")
    print(
        "\nIf you have not yet enabled persistent storage, do so under\n"
        f"  {settings_url}#storage  → Small (5 GB), then set BGF_DATA_ROOT=/data\n"
        "so experiments and uploads survive Space restarts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
