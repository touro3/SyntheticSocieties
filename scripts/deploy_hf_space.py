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

from huggingface_hub import HfApi

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
    "experiments/**",
    "analysis/figures/**",
    "analysis/networks/**",
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
        folder_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ignore_patterns=IGNORE,
        commit_message="Deploy from CI",
    )
    print(f"Deployed to https://huggingface.co/spaces/{space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
