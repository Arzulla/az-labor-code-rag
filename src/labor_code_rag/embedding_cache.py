"""Embedding cache keyed by ``(embedding_model, sha256(text))`` (CLAUDE.md §6).

SQLite from the stdlib: one file, keyed lookups, no extra dependency. Vectors are stored
as JSON text; ~1k chunks x 1536 floats is small enough that a binary format buys nothing.
The model name is part of the key, so switching models never returns a stale vector.
"""

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from labor_code_rag.llm import LLMClient

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    model TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    vector TEXT NOT NULL,
    PRIMARY KEY (model, text_sha256)
)
"""


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path | str) -> None:
        """``path`` may be ``":memory:"`` in tests."""
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: Gradio calls handlers from worker threads; writes
        # happen only during indexing (single-threaded).
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(_SCHEMA)

    def get_many(self, model: str, hashes: Sequence[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for h in hashes:
            row = self._conn.execute(
                "SELECT vector FROM embeddings WHERE model = ? AND text_sha256 = ?", (model, h)
            ).fetchone()
            if row:
                found[h] = json.loads(row[0])
        return found

    def put_many(self, model: str, items: dict[str, list[float]]) -> None:
        with self._conn:  # one transaction
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                [(model, h, json.dumps(v)) for h, v in items.items()],
            )

    def close(self) -> None:
        self._conn.close()


def embed_with_cache(
    texts: Sequence[str], llm: LLMClient, cache: EmbeddingCache, purpose: str
) -> tuple[list[list[float]], int, float]:
    """Embed ``texts``, calling the API only for cache misses.

    Returns ``(vectors in input order, cache_hits, cost_usd of the misses)``.
    """
    model = llm.settings.embedding_model
    hashes = [text_sha256(t) for t in texts]
    cached = cache.get_many(model, hashes)
    # Keyed by hash: identical texts are embedded once.
    missing = {h: t for h, t in zip(hashes, texts, strict=True) if h not in cached}
    cost = 0.0
    if missing:
        result = llm.embed(list(missing.values()), purpose)
        fresh = dict(zip(missing, result.vectors, strict=True))
        cache.put_many(model, fresh)
        cached.update(fresh)
        cost = result.cost_usd
    hits = sum(1 for h in hashes if h not in missing)
    return [cached[h] for h in hashes], hits, cost
