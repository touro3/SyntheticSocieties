"""Disk-persistent semantic memory store for agents.

A Python adaptation of ruflo's AgentDB + HNSW pattern: an on-disk vector
store with semantic recall.  This complements the in-process
``HierarchicalMemory`` (which is lost on process exit) by giving each
experiment a durable ``memory.db`` that survives across runs and supports
similarity-based retrieval.

Design constraints (research-validity first):
  - **Additive, opt-in.**  Nothing here runs unless ``memory.persistent``
    is enabled in config; the M0–M3 ablation and default experiments are
    byte-for-byte unaffected.
  - **Zero hard dependencies.**  Storage uses the stdlib ``sqlite3``.
    Embeddings use ``sentence-transformers`` *if installed*, otherwise a
    deterministic hashing-bag fallback so CPU/CI runs need no extra deps
    (mirrors ruflo's WASM-fallback philosophy).
  - **ANN optional.**  ``hnswlib`` is used when present; otherwise a
    brute-force cosine scan, which is fine at research population sizes.

The store is keyed off the per-experiment directory convention already
used by ``bgf_logging`` (``experiments/<exp_id>/memory.db``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import struct
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.memory import MemoryItem

logger = logging.getLogger(__name__)

_EMBED_DIM = 384  # all-MiniLM-L6-v2 dimensionality; fallback matches it.


# ── Embedding backend ──────────────────────────────────────────────────────


class _Embedder:
    """Lazily-loaded text embedder with a dependency-free fallback.

    Tries ``sentence-transformers``; if unavailable (or the model fails to
    load) falls back to a deterministic hashing-bag embedding so the store
    still works in CI without network or heavy deps.  The fallback is not
    semantically strong but is stable and unit-testable.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._tried = False

    def _ensure(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
            logger.info("PersistentMemory: using sentence-transformers (%s)", self.model_name)
        except Exception as exc:  # pragma: no cover - depends on env
            logger.info(
                "PersistentMemory: sentence-transformers unavailable (%s); using deterministic hashing fallback",
                exc,
            )
            self._model = None

    def encode(self, text: str) -> list[float]:
        self._ensure()
        if self._model is not None:  # pragma: no cover - depends on env
            vec = self._model.encode(text, normalize_embeddings=True)
            return [float(x) for x in vec]
        return self._hashing_embed(text)

    @staticmethod
    def _hashing_embed(text: str) -> list[float]:
        """Deterministic bag-of-hashed-tokens embedding, L2-normalized."""
        vec = [0.0] * _EMBED_DIM
        tokens = text.lower().split()
        for tok in tokens:
            # MD5 used only as a deterministic feature hash for a bag-of-tokens
            # embedding — no secrecy or integrity claim. `usedforsecurity=False`
            # tells Bandit / FIPS this is not a security context.
            h = int.from_bytes(
                hashlib.md5(tok.encode(), usedforsecurity=False).digest()[:8],
                "little",
            )
            idx = h % _EMBED_DIM
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


# ── Store ──────────────────────────────────────────────────────────────────


class PersistentMemoryStore:
    """SQLite-backed semantic memory for one experiment.

    Parameters
    ----------
    db_path:
        Destination ``.db`` file.  Parent dirs are created.  Use
        ``:memory:`` for ephemeral (test) stores.
    embedding_model:
        Passed to :class:`_Embedder`.
    """

    def __init__(self, db_path: str | Path, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._embedder = _Embedder(embedding_model)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER,
                partner_id TEXT,
                event_type TEXT,
                content TEXT,
                outcome TEXT,
                importance REAL,
                embedding BLOB
            )
            """
        )
        self._conn.commit()

    # ── Write ──────────────────────────────────────────────────────────────

    def add(self, item: MemoryItem) -> None:
        """Persist a single :class:`~agents.memory.MemoryItem`."""
        emb = self._embedder.encode(item.content or item.event_type)
        self._conn.execute(
            "INSERT INTO memory "
            "(round_id, partner_id, event_type, content, outcome, importance, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                item.round_id,
                item.partner_id,
                item.event_type,
                item.content,
                json.dumps(item.outcome),
                item.importance,
                _pack(emb),
            ),
        )

    def add_batch(self, items: list[MemoryItem]) -> None:
        for it in items:
            self.add(it)
        self.flush()

    def flush(self) -> None:
        """Commit pending writes to disk."""
        self._conn.commit()

    # ── Read ───────────────────────────────────────────────────────────────

    def recall(self, query: str, k: int = 5) -> list[dict]:
        """Return up to ``k`` rows most semantically similar to ``query``.

        Uses ``hnswlib`` when available, else a brute-force cosine scan.
        Each result is a dict mirroring the stored columns plus ``score``.
        """
        q = self._embedder.encode(query)
        rows = self._conn.execute(
            "SELECT round_id, partner_id, event_type, content, outcome, importance, embedding FROM memory"
        ).fetchall()
        if not rows:
            return []

        scored: list[tuple[float, tuple]] = []
        try:
            import hnswlib  # type: ignore

            dim = len(q)
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(max_elements=len(rows), ef_construction=100, M=16)
            for i, r in enumerate(rows):
                index.add_items([_unpack(r[6])], [i])
            index.set_ef(max(k * 4, 16))
            labels, dists = index.knn_query([q], k=min(k, len(rows)))
            for lbl, dist in zip(labels[0], dists[0]):
                scored.append((1.0 - float(dist), rows[int(lbl)]))
        except Exception:
            for r in rows:
                scored.append((_cosine(q, _unpack(r[6])), r))
            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:k]

        results = []
        for score, r in scored:
            results.append(
                {
                    "round_id": r[0],
                    "partner_id": r[1],
                    "event_type": r[2],
                    "content": r[3],
                    "outcome": json.loads(r[4]) if r[4] else {},
                    "importance": r[5],
                    "score": round(float(score), 6),
                }
            )
        return results

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0])

    def close(self) -> None:
        try:
            self._conn.commit()
            self._conn.close()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self) -> PersistentMemoryStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
