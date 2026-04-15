"""
download_models.py — Pre-cache LLM weights used in BGF experiments.

Usage:
    # Cache Mistral-7B (minimum required, ~14 GB):
    python scripts/download_models.py

    # Cache all models used in the cross-model study:
    python scripts/download_models.py --all

    # Cache a specific model:
    python scripts/download_models.py --model mistralai/Mistral-7B-Instruct-v0.3

Requirements:
    pip install huggingface_hub transformers

GPU requirements:
    Inference requires at least 16 GB VRAM.
    Tested on: NVIDIA A100 40 GB, RTX 3090 24 GB.
    Quantised (4-bit) inference possible with bitsandbytes on 8 GB VRAM.

Cache location:
    By default, models are cached to ~/.cache/huggingface/hub/.
    To override:
        export HF_HOME=/mnt/raid/workspace/lucastourinho/models
    This is the path used in BGF experiment configs (configs/base_config.yaml).
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Primary BGF model (Condition A/B baselines, all ablations)",
        "vram_gb": 14,
        "required": True,
    },
    "llama3-8b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "description": "Cross-model generalizability study (Phase 16)",
        "vram_gb": 16,
        "required": False,
    },
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Cross-model generalizability study (Phase 16)",
        "vram_gb": 14,
        "required": False,
    },
}


def cache_model(hf_id: str, description: str) -> None:
    """Download and cache a HuggingFace model's tokenizer and config."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("ERROR: huggingface_hub not installed.\nRun: pip install huggingface_hub transformers")

    print(f"\nCaching: {hf_id}")
    print(f"  {description}")

    cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface/hub")
    print(f"  Cache dir: {cache_dir}")

    try:
        path = snapshot_download(
            repo_id=hf_id,
            ignore_patterns=["*.gguf", "original/*"],  # skip non-HF formats
        )
        print(f"  Cached to: {path}")
    except Exception as exc:
        print(f"  WARNING: Download failed — {exc}")
        print("  You may need a HuggingFace token for gated models (e.g. Llama-3).")
        print("  Run:  huggingface-cli login")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-cache BGF LLM weights from HuggingFace Hub.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Cache all models (required + optional cross-model study).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID to cache (overrides default selection).",
    )
    args = parser.parse_args()

    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")
    print(f"HF_HOME: {hf_home}")
    print("To change cache location: export HF_HOME=/your/path\n")

    if args.model:
        cache_model(args.model, "User-specified model")
        return

    for key, info in MODELS.items():
        if args.all or info["required"]:
            cache_model(info["hf_id"], info["description"])
        else:
            print(f"Skipping {key} (optional, use --all to include)")

    print("\nAll requested models cached.")
    print("Set llm.model_id in configs/base_config.yaml to switch models.")


if __name__ == "__main__":
    main()
