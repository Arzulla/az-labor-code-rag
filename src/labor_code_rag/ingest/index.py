"""Embed chunks and store them in a persistent Chroma collection (``make index``), ADR-005.

One collection per embedding model (``labor_code__text-embedding-3-small``): vectors from
different models live in different spaces and must never be mixed. Distance is cosine,
set explicitly (Chroma's default is L2). Re-running is idempotent: upsert by ``chunk_id``,
chunks that disappeared from ``chunks.jsonl`` are deleted, and unchanged texts come from
the embedding cache (zero API calls).
"""

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from labor_code_rag.config import load_settings
from labor_code_rag.embedding_cache import EmbeddingCache, embed_with_cache
from labor_code_rag.errors import LLMError, SourceDownloadError
from labor_code_rag.ingest.chunk import CHUNKS_FILE
from labor_code_rag.llm import LLMClient, create_openai_client
from labor_code_rag.logging_setup import configure_logging, set_request_id
from labor_code_rag.models import Chunk

# Explicit name: under `python -m` __name__ would be "__main__".
logger = logging.getLogger("labor_code_rag.ingest.index")

_UPSERT_BATCH = 500  # well under Chroma's max batch size


def collection_name(prefix: str, embedding_model: str) -> str:
    """``labor_code__text-embedding-3-small``; chars Chroma rejects (``/`` in
    ``BAAI/bge-m3``) become ``-``."""
    return f"{prefix}__{re.sub(r'[^a-zA-Z0-9._-]', '-', embedding_model)}"


def open_collection(client: ClientAPI, name: str) -> Collection:
    """Get or create a cosine collection; we always pass our own embeddings."""
    collection = client.get_or_create_collection(
        name,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,  # no Chroma default model: embeddings come from llm.py
    )
    # get_or_create ignores `configuration` for an existing collection: verify it.
    hnsw = (collection.configuration or {}).get("hnsw") or {}
    if hnsw.get("space") != "cosine":
        raise ValueError(f"collection {name} uses {hnsw.get('space')!r}, expected cosine")
    return collection


def chunk_metadata(chunk: Chunk) -> dict[str, str]:
    """Chroma metadata values cannot be None: ``point=None`` is stored as ``""``."""
    return {
        "chunk_id": chunk.chunk_id,
        "article_no": chunk.article_no,
        "point": chunk.point or "",
        "title": chunk.title,
        "part": chunk.part,
        "chapter": chunk.chapter,
        "url": chunk.url,
        "source_downloaded_at": chunk.source_downloaded_at.isoformat(),
    }


def chunk_from_metadata(metadata: dict[str, Any], text: str) -> Chunk:
    """Inverse of ``chunk_metadata``; ``""`` maps back to ``point=None``."""
    return Chunk(
        chunk_id=metadata["chunk_id"],
        article_no=metadata["article_no"],
        point=metadata["point"] or None,
        title=metadata["title"],
        part=metadata["part"],
        chapter=metadata["chapter"],
        url=metadata["url"],
        text=text,
        source_downloaded_at=datetime.fromisoformat(metadata["source_downloaded_at"]),
    )


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise SourceDownloadError(f"{path} not found; run `make ingest` first")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Chunk.model_validate_json(line) for line in lines if line]


def index_chunks(
    chunks: list[Chunk], llm: LLMClient, cache: EmbeddingCache, collection: Collection
) -> dict[str, Any]:
    """Embed (via cache) and upsert ``chunks``; return the ``index.completed`` fields."""
    started = time.perf_counter()
    vectors, cache_hits, cost = embed_with_cache(
        [c.text for c in chunks], llm, cache, purpose="index"
    )
    for start in range(0, len(chunks), _UPSERT_BATCH):
        batch = chunks[start : start + _UPSERT_BATCH]
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors[start : start + _UPSERT_BATCH],  # type: ignore[arg-type]
            documents=[c.text for c in batch],
            metadatas=[chunk_metadata(c) for c in batch],
        )
    # Upsert never removes: delete ids that are no longer in chunks.jsonl.
    stale = set(collection.get(include=[])["ids"]) - {c.chunk_id for c in chunks}
    if stale:
        collection.delete(ids=sorted(stale))
    stats = {
        "chunks": len(chunks),
        "embedded": len(chunks) - cache_hits,
        "cache_hits": cache_hits,
        "deleted": len(stale),
        "model": llm.settings.embedding_model,
        "collection": collection.name,
        "cost_usd": round(cost, 6),
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }
    logger.info("index.completed", extra=stats)
    return stats


def main() -> None:
    settings = load_settings()
    configure_logging(
        settings.logging.level, settings.logging.format, settings.logging.log_payloads
    )
    set_request_id()
    if settings.data is None or settings.llm is None:
        raise LLMError("missing 'data' or 'llm' section in config.yaml")
    llm = LLMClient(create_openai_client(settings.llm, settings.openai_api_key), settings.llm)
    cache = EmbeddingCache(settings.index.cache_path)
    client = chromadb.PersistentClient(path=str(settings.index.chroma_dir))
    name = collection_name(settings.index.collection_prefix, settings.llm.embedding_model)
    try:
        chunks = load_chunks(settings.data.processed_dir / CHUNKS_FILE)
        index_chunks(chunks, llm, cache, open_collection(client, name))
    finally:
        cache.close()


if __name__ == "__main__":
    main()
