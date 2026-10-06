"""Dense retrieval: embed the question, take the top-k nearest chunks from Chroma."""

import logging
import time

from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError

from labor_code_rag.errors import LLMError, RetrievalError
from labor_code_rag.ingest.index import chunk_from_metadata
from labor_code_rag.llm import LLMClient
from labor_code_rag.models import RetrievedChunk
from labor_code_rag.text import normalize

logger = logging.getLogger(__name__)


def search(
    question: str, llm: LLMClient, collection: Collection, k: int
) -> tuple[list[RetrievedChunk], float]:
    """Return the ``k`` most similar chunks (best first) and the embedding cost in USD.

    Raises:
        RetrievalError: embedding or the Chroma query failed.
    """
    started = time.perf_counter()
    try:
        # Same NFC form as the indexed text: "ə" can be one code point or two.
        embedded = llm.embed([normalize(question)], purpose="query_embedding")
        result = collection.query(
            query_embeddings=embedded.vectors,  # type: ignore[arg-type]
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
    except LLMError as exc:
        raise RetrievalError(f"query embedding failed: {exc}") from exc
    except (ChromaError, ValueError) as exc:  # ValueError: client-side validation
        raise RetrievalError(f"vector query failed: {type(exc).__name__}: {exc}") from exc

    # Chroma returns one list per query embedding; we sent one, hence [0].
    metadatas = (result["metadatas"] or [[]])[0]
    documents = (result["documents"] or [[]])[0]
    distances = (result["distances"] or [[]])[0]
    chunks = [
        RetrievedChunk(
            **chunk_from_metadata(dict(meta), doc).model_dump(),
            score=1.0 - dist,  # cosine distance = 1 - cosine similarity
            rank=rank,
        )
        for rank, (meta, doc, dist) in enumerate(
            zip(metadatas, documents, distances, strict=True), start=1
        )
    ]
    logger.info(
        "retrieval.completed",
        extra={
            "source": "vector",
            "k": k,
            "article_nos": list(dict.fromkeys(c.article_no for c in chunks)),  # rank order
            "chunk_ids": [c.chunk_id for c in chunks],
            "latency_ms": round((time.perf_counter() - started) * 1000),
        },
    )
    return chunks, embedded.cost_usd
