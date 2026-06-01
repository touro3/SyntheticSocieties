#!/usr/bin/env python
"""Resilient multi-seed sweep orchestrator (ruflo autopilot pattern).

``run_experiment_matrix.py`` already skips cells whose ``summary.json``
exists, but it has no durable record of *intent* — if the process dies
mid-cell you can't tell a never-started cell from an in-flight one, and
there's no resumable progress log.

This driver wraps the same matrix using a persistent ``sweep_state.json``
checklist (one entry per condition×seed×ablation×temperature cell:
``pending`` → ``running`` → ``done`` / ``failed``).  Killing the process
and restarting with the same args resumes only the cells that are not yet
``done``, re-running anything stuck in ``running`` (assumed interrupted).

The state logic (:class:`SweepState`) is GPU-free and unit-tested; the
``run`` path reuses ``run_simulation`` and the matrix helpers verbatim.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

from scripts.run_experiment_matrix import _build_overrides, _cell_id, parse_seed_list


class SweepState:
    """Durable per-cell checklist persisted as JSON."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.cells: dict[str, dict] = {}
        if self.path.is_file():
            self.cells = json.loads(self.path.read_text()).get("cells", {})

    def register(self, cell_ids: list[str]) -> None:
        for cid in cell_ids:
            self.cells.setdefault(cid, {"status": "pending", "attempts": 0})
        self._save()

    def pending(self, cell_ids: list[str]) -> list[str]:
        """Cells still needing work: not ``done``.  ``running`` is treated
        as interrupted and re-queued."""
        return [c for c in cell_ids if self.cells.get(c, {}).get("status") != "done"]

    def mark(self, cid: str, status: str, error: str | None = None) -> None:
        entry = self.cells.setdefault(cid, {"status": "pending", "attempts": 0})
        entry["status"] = status
        if status == "running":
            entry["attempts"] = entry.get("attempts", 0) + 1
        if error:
            entry["error"] = error[:200]
        self._save()

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.cells.values():
            out[e["status"]] = out.get(e["status"], 0) + 1
        return out

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cells": self.cells}, indent=2, sort_keys=True))
        tmp.replace(self.path)  # atomic


def parse_args():
    p = argparse.ArgumentParser(description="Resilient BGF sweep orchestrator")
    p.add_argument("--conditions", nargs="+", default=["A", "B"], choices=["A", "B", "C", "D"])
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--agents", type=int, default=100)
    p.add_argument("--ablation-levels", nargs="*", default=None)
    p.add_argument("--temperatures", nargs="*", default=None)
    p.add_argument("--state", default="experiments/sweep_state.json")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _cells(args) -> list[tuple]:
    seeds = parse_seed_list(args.seeds)
    abls = [int(x) for x in args.ablation_levels] if args.ablation_levels else [None]
    temps = [float(x) for x in args.temperatures] if args.temperatures else [None]
    return list(product(args.conditions, seeds, abls, temps))


def main() -> int:
    args = parse_args()
    cells = _cells(args)
    cell_ids = [_cell_id(c, s, a, t) for (c, s, a, t) in cells]

    state = SweepState(args.state)
    state.register(cell_ids)
    todo = set(state.pending(cell_ids))

    print(f"Sweep: {len(cell_ids)} cells, {len(todo)} pending  (state: {args.state})")
    if args.dry_run:
        for cid in cell_ids:
            tag = "TODO" if cid in todo else "done"
            print(f"  [{tag}] {cid}")
        return 0

    from scripts.run_config_simulation import run_simulation

    for (cond, seed, abl, temp), cid in zip(cells, cell_ids):
        if cid not in todo:
            print(f"  SKIP {cid} (done)")
            continue
        state.mark(cid, "running")
        overrides = _build_overrides(cond, seed, args.rounds, args.agents, abl, temp, cid)
        t0 = time.time()
        print(f"  RUN  {cid} ...", end="", flush=True)
        try:
            run_simulation("configs/base_config.yaml", overrides)
            state.mark(cid, "done")
            print(f" ✓ ({time.time() - t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001 - resilience is the point
            state.mark(cid, "failed", error=str(exc))
            print(f" ✗ ({time.time() - t0:.1f}s): {str(exc)[:80]}")

    print(f"\n  Sweep state: {state.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
