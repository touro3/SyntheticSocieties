"""Simulation watchdog — detects stalled runs and optionally restarts them.

Usage
-----
# Watch a specific experiment (alert only):
    python scripts/sim_watchdog.py --exp-id cmp_llm_s2

# Watch all active experiments in experiments/:
    python scripts/sim_watchdog.py --all

# Kill and restart a stalled sim (requires --pid or --restart-cmd):
    python scripts/sim_watchdog.py --exp-id cmp_llm_s2 --pid 12345 --stale-after 10
    python scripts/sim_watchdog.py --exp-id cmp_llm_s2 --restart-cmd "bash pipeline_bad_apple.sh"

Detection
---------
Each experiment directory may contain a heartbeat.json written by SimulationKernel
after every round:
    {"round_id": 42, "ts": 1712345678.3, "n_agents": 50}

A run is considered STALE if the heartbeat is older than --stale-after minutes.
A run is MISSING if no heartbeat.json exists yet (may still be loading).
A run is DEAD if heartbeat is older than 2× --stale-after.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HEARTBEAT_FILE = "heartbeat.json"
POLL_INTERVAL_S = 30  # how often to check, in seconds


def _read_heartbeat(exp_dir: Path) -> dict | None:
    hb = exp_dir / HEARTBEAT_FILE
    if not hb.exists():
        return None
    try:
        return json.loads(hb.read_text())
    except Exception:
        return None


def _status(hb: dict | None, stale_after_s: float, now: float) -> str:
    if hb is None:
        return "MISSING"
    age = now - hb["ts"]
    if age > stale_after_s * 2:
        return "DEAD"
    if age > stale_after_s:
        return "STALE"
    return "OK"


def _age_str(ts: float, now: float) -> str:
    age = now - ts
    if age < 60:
        return f"{age:.0f}s ago"
    return f"{age / 60:.1f}m ago"


def watch_once(exp_dirs: list[Path], stale_after_s: float, verbose: bool = True) -> list[tuple[Path, str]]:
    """Check all exp_dirs once. Returns list of (dir, status) pairs."""
    now = time.time()
    results = []
    for exp_dir in exp_dirs:
        hb = _read_heartbeat(exp_dir)
        status = _status(hb, stale_after_s, now)
        results.append((exp_dir, status, hb))
        if verbose:
            exp_id = exp_dir.name
            if hb:
                print(f"  [{status:7s}] {exp_id:40s}  round={hb['round_id']:4d}  ts={_age_str(hb['ts'], now)}")
            else:
                print(f"  [{status:7s}] {exp_id:40s}  (no heartbeat yet)")
    return results


def kill_pid(pid: int) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"  Sent SIGTERM to PID {pid}")
        time.sleep(3)
        # If still alive, escalate
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"  Sent SIGKILL to PID {pid} (SIGTERM ignored)")
        except ProcessLookupError:
            pass  # already dead
        return True
    except ProcessLookupError:
        print(f"  PID {pid} not found (already dead?)")
        return False
    except PermissionError:
        print(f"  No permission to kill PID {pid}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Simulation stall watchdog")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp-id", type=str, help="Single experiment ID to watch")
    group.add_argument("--all", action="store_true", help="Watch all experiments/ subdirs")

    parser.add_argument(
        "--experiments-root",
        type=str,
        default="experiments",
        help="Root directory containing experiment subdirs (default: experiments/)",
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=5.0,
        help="Minutes without a heartbeat before run is considered STALE (default: 5)",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=POLL_INTERVAL_S,
        help=f"Seconds between checks (default: {POLL_INTERVAL_S})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check once and exit instead of looping",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="PID of the simulation process to kill when STALE/DEAD",
    )
    parser.add_argument(
        "--restart-cmd",
        type=str,
        default=None,
        help="Shell command to run after killing the stalled process",
    )
    args = parser.parse_args()

    root = Path(args.experiments_root)
    stale_s = args.stale_after * 60

    if args.exp_id:
        exp_dirs = [root / args.exp_id]
    else:
        exp_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    if not exp_dirs:
        print(f"No experiment directories found under {root}", file=sys.stderr)
        sys.exit(1)

    action_taken = False

    while True:
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{now_str}] Checking {len(exp_dirs)} experiment(s)…")

        results = watch_once(exp_dirs, stale_s)

        stalled = [(d, st, hb) for d, st, hb in results if st in ("STALE", "DEAD")]

        if stalled and not action_taken:
            for exp_dir, status, hb in stalled:
                print(f"\n  !! {status}: {exp_dir.name}")

                if args.pid:
                    print(f"  Killing PID {args.pid}…")
                    kill_pid(args.pid)
                    action_taken = True

                if args.restart_cmd:
                    print(f"  Restarting: {args.restart_cmd}")
                    subprocess.Popen(args.restart_cmd, shell=True)
                    action_taken = True

                if not args.pid and not args.restart_cmd:
                    print("  (pass --pid or --restart-cmd to auto-recover)")

        if args.once:
            # Exit with non-zero if any stalls detected
            sys.exit(1 if stalled else 0)

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
