"""Guard the §6.1 Proposition 3 analytic certificate emission contract.

The paper cites `analysis/brm_sensitivity.py --emit-certificate` as the
source of the four Δ_j sub-component values that make the weight-robust
BRM ordering claim verifiable. If the emission contract changes shape,
the paper's `certificate.vertex_deltas{jsd, gini_gap, coop_gap, stability}`
+ `min_delta` + `verdict` references will go stale silently. This test
exercises the flag end-to-end and asserts the shape contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CERT_JSON = REPO_ROOT / "analysis" / "tables" / "brm_sensitivity.json"
EXPECTED_VERTEX_KEYS = {"jsd", "gini_gap", "coop_gap", "stability"}


def _run_emit_certificate() -> dict:
    """Invoke the script as the paper claims and return the parsed JSON."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "analysis" / "brm_sensitivity.py"),
        "--emit-certificate",
        "--n-samples",
        "200",  # keep the Dirichlet sweep small — certificate is exact, sweep size doesn't matter
        "--out-json",
        str(CERT_JSON),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"brm_sensitivity --emit-certificate failed (exit {result.returncode}):\n"
        f"STDOUT: {result.stdout[-500:]}\nSTDERR: {result.stderr[-500:]}"
    )
    assert CERT_JSON.exists(), f"certificate JSON not written to {CERT_JSON}"
    return json.loads(CERT_JSON.read_text())


def test_certificate_block_present_and_well_shaped():
    data = _run_emit_certificate()
    assert "certificate" in data, (
        "--emit-certificate must add a top-level 'certificate' key to the JSON summary "
        "(§6.1 Proposition 3 analytic certificate)"
    )
    cert = data["certificate"]

    # Four vertex deltas — these are the Δ_j values the paper cites
    assert "vertex_deltas" in cert, "certificate must expose vertex_deltas dict"
    assert set(cert["vertex_deltas"].keys()) == EXPECTED_VERTEX_KEYS, (
        f"vertex_deltas must have exactly {EXPECTED_VERTEX_KEYS}, "
        f"got {set(cert['vertex_deltas'].keys())}"
    )
    for k, v in cert["vertex_deltas"].items():
        assert isinstance(v, (int, float)), f"vertex_delta[{k}] must be numeric, got {type(v)}"

    # delta_vector aligned with the four keys
    assert "delta_vector" in cert and len(cert["delta_vector"]) == 4, (
        "delta_vector must be a 4-element list aligned with the four BRM sub-components"
    )

    # min_delta and verdict — the Proposition 3 closed-form decision rule
    assert "min_delta" in cert and isinstance(cert["min_delta"], (int, float))
    assert "argmin_vertex" in cert and cert["argmin_vertex"] in EXPECTED_VERTEX_KEYS
    assert "verdict" in cert and cert["verdict"] in ("ROBUST", "NOT_ROBUST")
    # The rule must match what the paper cites verbatim
    assert cert["verdict"] == ("ROBUST" if cert["min_delta"] > 0 else "NOT_ROBUST"), (
        "verdict must follow the rule 'min_j Δ_j > 0 ⇒ ROBUST' (Proposition 3)"
    )

    # Provenance fields the paper uses to cite the method
    assert "method" in cert and "LP-vertex" in cert["method"], (
        "method field must reference the LP-vertex enumeration method"
    )
    assert "theorem" in cert and "Theorem 2" in cert["theorem"], (
        "theorem field must point readers to docs/theorems.md Theorem 2"
    )
