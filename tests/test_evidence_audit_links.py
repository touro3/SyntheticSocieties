"""CI gate: every ✅ row in docs/evidence_audit.md must point to a real file.

This converts the evidence audit from a manual-hygiene document into a
machine-checked invariant. The test parses the markdown tables in
``docs/evidence_audit.md`` and for every row marked ✅ (VERIFIED) it
extracts file paths from the "Evidence path" column and asserts each
one exists.

The test is intentionally tolerant:

* Only rows whose status cell contains a literal ✅ are checked.
* Multiple paths in a single cell are split on commas, ``+``, and the
  word "and".
* Backticked tokens are extracted; bare prose is ignored.
* Line-number suffixes (``foo.json lines 36-42``) are stripped before
  the existence check.
* Paths starting with ``http`` are skipped (external references).
* Wildcard paths (``experiments/*/summary.json``) are checked via
  ``glob`` rather than literal existence.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_PATH = REPO_ROOT / "docs" / "evidence_audit.md"

VERIFIED_MARKER = "✅"
BACKTICK_RE = re.compile(r"`([^`]+)`")
LINE_SUFFIX_RE = re.compile(r"\s+lines?\s+\d+.*$", re.IGNORECASE)
# Python "module.py:symbol" reference — strip the ":symbol" tail so we
# verify the file exists, not the symbol.
SYMBOL_SUFFIX_RE = re.compile(r"\.py:[A-Za-z_][\w.]*$")


def _looks_like_path(token: str) -> bool:
    if not token or token.startswith("http"):
        return False
    if token.startswith("#") or token.startswith("§"):
        return False
    if "/" not in token and "." not in token:
        return False
    # Heuristic: too short to be a useful path.
    return len(token) >= 5


def _candidate_paths(evidence_cell: str) -> list[str]:
    raw_tokens: list[str] = []
    # Split the cell on common separators that join multiple references.
    fragments = re.split(r",|\+| and ", evidence_cell)
    for frag in fragments:
        for m in BACKTICK_RE.finditer(frag):
            tok = m.group(1).strip()
            tok = LINE_SUFFIX_RE.sub("", tok).strip()
            if SYMBOL_SUFFIX_RE.search(tok):
                tok = tok.rsplit(":", 1)[0]
            if _looks_like_path(tok):
                raw_tokens.append(tok)
    return raw_tokens


def _parse_verified_rows() -> list[tuple[str, str, list[str]]]:
    """Return (claim_id, claim_short, candidate_paths) for ✅ rows."""
    if not AUDIT_PATH.exists():
        pytest.skip(f"{AUDIT_PATH} not present — nothing to verify")

    rows: list[tuple[str, str, list[str]]] = []
    for raw_line in AUDIT_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or VERIFIED_MARKER not in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        claim_id, claim_short, evidence_cell, status_cell = cells[:4]
        if VERIFIED_MARKER not in status_cell:
            continue
        # Skip header / separator lines.
        if set(claim_id) <= {"-", ":", " "}:
            continue
        paths = _candidate_paths(evidence_cell)
        if not paths:
            continue
        rows.append((claim_id, claim_short, paths))
    return rows


VERIFIED_ROWS = _parse_verified_rows()


@pytest.mark.parametrize(
    "claim_id, claim_short, paths",
    VERIFIED_ROWS,
    ids=lambda val: val if isinstance(val, str) else "",
)
def test_verified_audit_row_paths_exist(claim_id: str, claim_short: str, paths: list[str]) -> None:
    missing: list[str] = []
    for path_str in paths:
        # Wildcard paths are treated as existing if at least one match.
        if "*" in path_str:
            matches = list(REPO_ROOT.glob(path_str))
            if not matches:
                missing.append(path_str)
            continue
        candidate = (REPO_ROOT / path_str).resolve()
        if not candidate.exists():
            missing.append(path_str)
    if missing:
        pytest.fail(f"Audit row {claim_id!r} ({claim_short!r}) marked ✅ but paths missing on disk: {missing}")


def test_audit_has_at_least_one_verified_row() -> None:
    assert VERIFIED_ROWS, (
        "docs/evidence_audit.md has zero ✅ rows with parseable paths — "
        "either the audit is empty or the parser regex is stale."
    )
