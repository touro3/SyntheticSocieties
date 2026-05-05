"""Live simulation status monitor.

Reads all experiments/*/heartbeat.json files and prints a live-updating table.
Run this in a second tmux pane while the simulation is running.

Usage
-----
    python scripts/sim_status.py                     # watch all experiments
    python scripts/sim_status.py --active            # only show OK/STALE/DEAD (hide MISSING)
    python scripts/sim_status.py --exp cmp_llm_s2    # filter by prefix
    python scripts/sim_status.py --once              # print once and exit
    python scripts/sim_status.py --refresh 5         # refresh every 5 seconds (default 10)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HEARTBEAT_FILE = "heartbeat.json"
DEFAULT_STALE_MINUTES = 5.0
DEFAULT_REFRESH_S = 10


# ANSI colours (suppressed when not a TTY)
_IS_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _IS_TTY:
        return text
    return f"\033[{code}m{text}\033[0m"


GREEN = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
RED = lambda t: _c("31", t)
CYAN = lambda t: _c("36", t)
BOLD = lambda t: _c("1", t)
DIM = lambda t: _c("2", t)


def _read_heartbeat(exp_dir: Path) -> dict | None:
    hb = exp_dir / HEARTBEAT_FILE
    if not hb.exists():
        return None
    try:
        return json.loads(hb.read_text())
    except Exception:
        return None


def _classify(hb: dict | None, stale_s: float, now: float) -> str:
    if hb is None:
        return "MISSING"
    age = now - hb["ts"]
    if age > stale_s * 2:
        return "DEAD"
    if age > stale_s:
        return "STALE"
    return "OK"


def _age_str(ts: float, now: float) -> str:
    age = now - ts
    if age < 60:
        return f"{age:4.0f}s"
    if age < 3600:
        return f"{age / 60:4.1f}m"
    return f"{age / 3600:4.1f}h"


def _status_cell(status: str) -> str:
    if status == "OK":
        return GREEN("  OK   ")
    if status == "STALE":
        return YELLOW(" STALE ")
    if status == "DEAD":
        return RED("  DEAD ")
    return DIM("  ---  ")  # MISSING


def _render_table(exp_dirs: list[Path], stale_s: float, active_only: bool) -> str:
    now = time.time()
    rows = []
    counts = {"OK": 0, "STALE": 0, "DEAD": 0, "MISSING": 0}

    for exp_dir in exp_dirs:
        hb = _read_heartbeat(exp_dir)
        status = _classify(hb, stale_s, now)
        counts[status] += 1

        if active_only and status == "MISSING":
            continue

        exp_id = exp_dir.name
        if hb:
            round_id = hb.get("round_id", "?")
            n_agents = hb.get("n_agents", "?")
            age = _age_str(hb["ts"], now)
            rows.append(
                f"{_status_cell(status)} {exp_id:<40s}  "
                f"round={BOLD(str(round_id)):>6s}  agents={n_agents:>3}  last={age}"
            )
        else:
            rows.append(f"{_status_cell(status)} {DIM(exp_id):<40s}  (no heartbeat)")

    header = BOLD("STATUS  EXPERIMENT                                       ROUND  AGENTS  LAST HB")
    separator = "─" * 80

    summary = (
        f"  {GREEN('OK')}: {counts['OK']}  "
        f"{YELLOW('STALE')}: {counts['STALE']}  "
        f"{RED('DEAD')}: {counts['DEAD']}  "
        f"{DIM('MISSING')}: {counts['MISSING']}"
    )

    ts_line = DIM(f"  Updated: {time.strftime('%H:%M:%S')}  —  stale threshold: {stale_s / 60:.0f}m")

    lines = [separator, header, separator] + rows + [separator, summary, ts_line]
    return "\n".join(lines)


def _clear_screen():
    if _IS_TTY:
        subprocess.run(["clear"], check=False)


def main():
    parser = argparse.ArgumentParser(description="Live simulation status monitor")
    parser.add_argument(
        "--experiments-root",
        type=str,
        default="experiments",
        help="Root directory containing experiment subdirs (default: experiments/)",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Filter experiments whose name contains this string",
    )
    parser.add_argument(
        "--active",
        action="store_true",
        help="Hide MISSING experiments (show only those with a heartbeat)",
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=DEFAULT_STALE_MINUTES,
        help=f"Minutes before a run is STALE (default: {DEFAULT_STALE_MINUTES})",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=DEFAULT_REFRESH_S,
        help=f"Seconds between refreshes (default: {DEFAULT_REFRESH_S})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print once and exit",
    )
    args = parser.parse_args()

    root = Path(args.experiments_root)
    if not root.exists():
        print(f"experiments root not found: {root}", file=sys.stderr)
        sys.exit(1)

    stale_s = args.stale_after * 60

    while True:
        exp_dirs = sorted(p for p in root.iterdir() if p.is_dir())
        if args.exp:
            exp_dirs = [d for d in exp_dirs if args.exp in d.name]

        table = _render_table(exp_dirs, stale_s, args.active)

        if not args.once:
            _clear_screen()

        print(table)

        if args.once:
            break

        time.sleep(args.refresh)


if __name__ == "__main__":
    main()
