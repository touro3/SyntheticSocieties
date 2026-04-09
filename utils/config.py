from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return the raw dict.

    Use this for generic YAML files (suite configs, experiment specs, etc.).
    For BGF simulation configs use :func:`load_bgf_config` instead, which
    additionally validates the contents against the BGFConfig schema.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_bgf_config(path: str | Path):
    """Load and validate a BGF simulation YAML config file.

    Parses the YAML and constructs a fully-validated BGFConfig instance.
    Any typos, missing required fields, or out-of-range values raise a
    Pydantic ValidationError at load time rather than deep inside a run.

    Args:
        path: Path to the BGF YAML config file.

    Returns:
        Validated BGFConfig instance.
    """
    from configs.schema import BGFConfig

    raw = load_config(path)
    return BGFConfig(**raw)
