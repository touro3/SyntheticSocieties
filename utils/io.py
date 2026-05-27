import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


_REDACTED = "***REDACTED***"
_SECRET_EXACT_NAMES = frozenset({"token", "secret", "password", "passwd", "apikey", "api_key"})
_SECRET_SUFFIXES = ("_api_key", "_apikey", "_token", "_secret", "_password", "_passwd")


def _is_secret_key(key: str) -> bool:
    # Match credential-shaped key names only. Earlier substring match treated
    # `max_new_tokens` as a secret because "token" is a substring — that turned
    # the int 256 into the string "***REDACTED***" on the resume snapshot and
    # crashed transformers' generate() on every resumed LLM run.
    k = key.lower()
    if k in _SECRET_EXACT_NAMES:
        return True
    return any(k.endswith(suf) for suf in _SECRET_SUFFIXES)


def redact_secrets(data: Any) -> Any:
    """Return a deep copy of ``data`` with secret-looking values masked.

    Any dict key whose name contains an entry from ``_SECRET_KEY_HINTS``
    (case-insensitive) has its value replaced with ``***REDACTED***``. Used
    before persisting config snapshots so credentials never land in
    world-readable ``experiments/<id>/config.yaml`` (served via public GET
    endpoints).
    """
    if isinstance(data, dict):
        out: dict[Any, Any] = {}
        for k, v in data.items():
            if isinstance(k, str) and _is_secret_key(k):
                out[k] = _REDACTED if v not in (None, "") else v
            else:
                out[k] = redact_secrets(v)
        return out
    if isinstance(data, list):
        return [redact_secrets(v) for v in data]
    return data


def set_global_seed(seed: int) -> None:
    """Set seed across all RNG sources for full reproducibility.

    Also enables cuDNN deterministic mode so conv kernels produce bit-exact
    outputs across runs.  ``benchmark=False`` prevents cuDNN from picking a
    different fast-path kernel each time, which would break reproducibility
    even with a fixed seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # cuDNN: deterministic algorithms + disable auto-tuner.
        # Cost: ~5-15% slower conv on first batch; negligible for transformer
        # inference where attention dominates, not conv.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
