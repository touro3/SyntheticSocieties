"""Reproducibility witness — a Python adaptation of ruflo's Ed25519 witness.

ruflo signs a manifest of build/run state so a third party can verify
nothing changed.  The research analogue: after an experiment finishes we
record a content hash of the *exact* inputs that produced it — the run
config, the resolved input data files, and the code revision — into
``witness.json``.  ``verify_witness`` recomputes and compares, turning
"is this result reproducible from these inputs?" into a one-command check.

Signing is optional: if ``cryptography`` is installed and a private key is
configured (``BGF_WITNESS_KEY`` env var, hex-encoded Ed25519 seed) the
manifest hash is signed; otherwise an unsigned hash manifest is written,
which is still sufficient for tamper-evidence within a trusted store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

WITNESS_FILE = "witness.json"


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev() -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout.strip()
        )
        return {"commit": rev, "dirty": dirty}
    except Exception:  # pragma: no cover - git absent
        return {"commit": None, "dirty": None}


def _manifest(exp_dir: Path, extra_inputs: list[str] | None = None) -> dict:
    """Build the hashable manifest for an experiment directory."""
    config_path = exp_dir / "config.yaml"
    inputs: dict[str, Optional[str]] = {
        "config.yaml": _sha256_file(config_path),
        "events.jsonl": _sha256_file(exp_dir / "events.jsonl"),
    }
    for rel in extra_inputs or []:
        p = Path(rel)
        inputs[str(rel)] = _sha256_file(p)

    return {
        "experiment_dir": exp_dir.name,
        "inputs": inputs,
        "git": _git_rev(),
    }


def _digest(manifest: dict) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _sign(digest_hex: str) -> Optional[str]:
    seed_hex = os.environ.get("BGF_WITNESS_KEY")
    if not seed_hex:
        return None
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
        return key.sign(bytes.fromhex(digest_hex)).hex()
    except Exception as exc:  # pragma: no cover - depends on env
        logger.warning("Witness signing failed (%s); writing unsigned manifest", exc)
        return None


def write_witness(exp_dir: str | Path, extra_inputs: list[str] | None = None) -> Path:
    """Write ``witness.json`` for a completed experiment directory.

    Returns the path written.  Safe to call unconditionally at run
    finalization; failures are logged, not raised.
    """
    exp_dir = Path(exp_dir)
    manifest = _manifest(exp_dir, extra_inputs)
    digest = _digest(manifest)
    witness = {
        "version": 1,
        "manifest": manifest,
        "digest_sha256": digest,
        "signature_ed25519": _sign(digest),
    }
    out = exp_dir / WITNESS_FILE
    out.write_text(json.dumps(witness, indent=2, sort_keys=True))
    logger.info("Wrote reproducibility witness: %s", out)
    return out


def verify_witness(exp_dir: str | Path) -> dict:
    """Recompute the manifest and compare against the stored witness.

    Returns ``{"ok": bool, "reason": str, ...}``.  ``ok`` is True only when
    the recomputed digest matches the stored one (and the signature, if
    present and a key is configured, verifies).
    """
    exp_dir = Path(exp_dir)
    wpath = exp_dir / WITNESS_FILE
    if not wpath.is_file():
        return {"ok": False, "reason": "no witness.json"}

    stored = json.loads(wpath.read_text())
    extra = [k for k in stored["manifest"]["inputs"] if k not in ("config.yaml", "events.jsonl")]
    recomputed = _digest(_manifest(exp_dir, extra))

    if recomputed != stored.get("digest_sha256"):
        return {
            "ok": False,
            "reason": "digest mismatch",
            "expected": stored.get("digest_sha256"),
            "recomputed": recomputed,
        }

    sig = stored.get("signature_ed25519")
    seed_hex = os.environ.get("BGF_WITNESS_KEY")
    if sig and seed_hex:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

            pub = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex)).public_key()
            pub.verify(bytes.fromhex(sig), bytes.fromhex(recomputed))
        except Exception as exc:
            return {"ok": False, "reason": f"signature invalid: {exc}"}

    return {"ok": True, "reason": "digest matches", "digest": recomputed}
