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
