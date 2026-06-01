#!/usr/bin/env python
"""Batch reproducibility-witness verifier.

Usage:
    python scripts/verify_witness.py experiments/<exp_id> [more dirs...]
    python scripts/verify_witness.py --all          # every experiments/* dir

Exit code is non-zero if any checked experiment fails verification, so this
is CI-friendly.  Complements scripts/research_integrity_audit.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

from bgf_logging.witness import verify_witness


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    if argv[0] == "--all":
        targets = sorted(p for p in Path("experiments").glob("*") if p.is_dir())
    else:
        targets = [Path(a) for a in argv]

    failures = 0
    for d in targets:
        result = verify_witness(d)
        status = "OK  " if result["ok"] else "FAIL"
        if not result["ok"]:
            failures += 1
        print(f"[{status}] {d}  — {result['reason']}")

    print(f"\n{len(targets) - failures}/{len(targets)} verified")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
