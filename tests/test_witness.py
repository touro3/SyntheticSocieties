"""Tests for the reproducibility witness (ruflo Ed25519 witness pattern)."""

from __future__ import annotations

import json

from bgf_logging.witness import verify_witness, write_witness


def _make_exp(tmp_path):
    exp = tmp_path / "exp_test"
    exp.mkdir()
    (exp / "config.yaml").write_text("policy:\n  type: mock\n")
    (exp / "events.jsonl").write_text('{"round": 0}\n')
    return exp


def test_write_then_verify_passes(tmp_path):
    exp = _make_exp(tmp_path)
    out = write_witness(exp)
    assert out.is_file()
    result = verify_witness(exp)
    assert result["ok"] is True


def test_tampered_config_fails_verification(tmp_path):
    exp = _make_exp(tmp_path)
    write_witness(exp)
    (exp / "config.yaml").write_text("policy:\n  type: TAMPERED\n")
    result = verify_witness(exp)
    assert result["ok"] is False
    assert result["reason"] == "digest mismatch"


def test_missing_witness_reports_cleanly(tmp_path):
    exp = _make_exp(tmp_path)
    result = verify_witness(exp)
    assert result["ok"] is False
    assert "no witness" in result["reason"]


def test_witness_records_git_and_digest(tmp_path):
    exp = _make_exp(tmp_path)
    write_witness(exp)
    data = json.loads((exp / "witness.json").read_text())
    assert "digest_sha256" in data
    assert "git" in data["manifest"]
    assert "config.yaml" in data["manifest"]["inputs"]
