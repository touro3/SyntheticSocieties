"""
Download LLM model weights for BGF simulation.

Cache-dir resolution (first non-empty wins):
    1. --cache-dir CLI argument
    2. BGF_MODEL_CACHE_DIR env var
    3. HF_HOME env var
    4. HuggingFace default (~/.cache/huggingface)

If you have a dedicated large-storage mount, set BGF_MODEL_CACHE_DIR once in
your shell profile:

    export BGF_MODEL_CACHE_DIR=/mnt/large-disk/hf-cache

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model mistralai/Mistral-7B-Instruct-v0.3
    python scripts/download_model.py --cache-dir /data/models
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_CACHE = (
    os.environ.get("BGF_MODEL_CACHE_DIR")
    or os.environ.get("HF_HOME")
    or str(Path.home() / ".cache" / "huggingface")
)


def parse_args():
    parser = argparse.ArgumentParser(description="Download LLM model for BGF.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE, help="Download directory")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("BGF Model Download")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Cache dir: {args.cache_dir}")

    # Create cache directory
    cache_path = Path(args.cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Set HuggingFace cache
    os.environ["HF_HOME"] = str(cache_path)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nDownloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=str(cache_path),
        trust_remote_code=True,
    )
    print(f"  Tokenizer ready: vocab_size={tokenizer.vocab_size}")

    print("\nDownloading model (this may take several minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        cache_dir=str(cache_path),
        trust_remote_code=True,
        device_map="cpu",  # Download to CPU first
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Model ready: {n_params:.1f}B parameters")

    # Verify size on disk
    model_files = list(cache_path.rglob("*.safetensors")) + list(cache_path.rglob("*.bin"))
    total_size = sum(f.stat().st_size for f in model_files) / 1e9
    print(f"  Disk usage: {total_size:.1f} GB")

    print(f"\nModel cached at: {cache_path}")
    print("Done! You can now run LLM simulations.")

    # Cleanup
    del model
    import gc

    gc.collect()


if __name__ == "__main__":
    main()
