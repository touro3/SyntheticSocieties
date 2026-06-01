"""Structured CLI output formatter (ruflo OutputFormatter pattern).

A tiny, dependency-free helper so scripts can emit either human-readable
text or machine-readable JSON from the same call — useful for CI, where
``--format json`` lets a workflow parse results instead of scraping stdout.

Intentionally minimal: this is glue for the new sweep/verify/regression
scripts, not a logging framework.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any


class OutputFormatter:
    def __init__(self, fmt: str = "text", stream=None) -> None:
        if fmt not in ("text", "json"):
            raise ValueError(f"unsupported format: {fmt!r} (use 'text' or 'json')")
        self.fmt = fmt
        self.stream = stream or sys.stdout

    def _emit(self, obj: Any) -> None:
        print(json.dumps(obj, default=str), file=self.stream)

    def message(self, text: str, **fields: Any) -> None:
        if self.fmt == "json":
            self._emit({"type": "message", "text": text, **fields})
        else:
            print(text, file=self.stream)

    def result(self, name: str, payload: Any) -> None:
        if self.fmt == "json":
            self._emit({"type": "result", "name": name, "payload": payload})
        else:
            print(f"{name}: {payload}", file=self.stream)

    def table(self, rows: Sequence[dict], columns: Sequence[str] | None = None) -> None:
        rows = list(rows)
        if self.fmt == "json":
            self._emit({"type": "table", "rows": rows})
            return
        if not rows:
            print("(no rows)", file=self.stream)
            return
        cols = list(columns) if columns else list(rows[0].keys())
        widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
        header = "  ".join(c.ljust(widths[c]) for c in cols)
        print(header, file=self.stream)
        print("  ".join("-" * widths[c] for c in cols), file=self.stream)
        for r in rows:
            print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols), file=self.stream)
