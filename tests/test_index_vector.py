import logging

import openai
import pytest
from chromadb.api.models.Collection import Collection

from labor_code_rag.embedding_cache import EmbeddingCache
from labor_code_rag.errors import RetrievalError
from labor_code_rag.ingest.index import collection_name, index_chunks
from labor_code_rag.llm import LLMClient
from labor_code_rag.models import Chunk
from labor_code_rag.retrieval.vector import search

from conftest import FakeOpenAI, api_error, make_chunk


def test_collection_name_includes_model() -> None:
    assert collection_name("labor_code", "text-embedding-3-small") == (
        "labor_code__text-embedding-3-small"
    )
    assert collection_name("labor_code", "BAAI/bge-m3") == "labor_code__BAAI-bge-m3"


def test_collection_uses_cosine(collection: Collection) -> None:
    assert collection.configuration["hnsw"]["space"] == "cosine"


def test_upsert_is_idempotent(
    llm: LLMClient, fake_openai: FakeOpenAI, collection: Collection, chunks: list[Chunk]
) -> None:
    cache = EmbeddingCache(":memory:")
    first = index_chunks(chunks, llm, cache, collection)
    second = index_chunks(chunks, llm, cache, collection)

    assert collection.count() == len(chunks)
    assert (first["embedded"], first["cache_hits"]) == (4, 0)
    assert (second["embedded"], second["cache_hits"], second["cost_usd"]) == (0, 4, 0)
    assert len(fake_openai.embed_calls) == 1  # second run: zero embedding calls


def test_removed_chunks_are_deleted(
    llm: LLMClient, collection: Collection, chunks: list[Chunk]
) -> None:
    cache = EmbeddingCache(":memory:")
    index_chunks(chunks, llm, cache, collection)
    stats = index_chunks(chunks[:2], llm, cache, collection)
    assert stats["deleted"] == 2
    assert sorted(collection.get(include=[])["ids"]) == ["114.1", "114.2"]


def test_index_logs_completed(
    llm: LLMClient, collection: Collection, chunks: list[Chunk], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    index_chunks(chunks, llm, EmbeddingCache(":memory:"), collection)
    (record,) = [r for r in caplog.records if r.getMessage() == "index.completed"]
    for field in ("chunks", "embedded", "cache_hits", "model", "cost_usd", "latency_ms"):
        assert hasattr(record, field)


def test_top_k_order_and_scores(
    llm: LLMClient, collection: Collection, chunks: list[Chunk]
) -> None:
    index_chunks(chunks, llm, EmbeddingCache(":memory:"), collection)

    results, cost = search("Hamilə qadınla xitam olarmı?", llm, collection, k=3)

    assert [r.chunk_id for r in results][0] == "79"
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score > results[1].score >= results[2].score
    assert cost > 0


def test_metadata_round_trips(llm: LLMClient, collection: Collection) -> None:
    whole = make_chunk("305", None, "Yekun", "Mətn hamilə.")
    point = make_chunk("7-1", "1", "Başlıq", "Mətn məzuniyyət.")
    index_chunks([whole, point], llm, EmbeddingCache(":memory:"), collection)

    results, _ = search("hamilə məzuniyyət", llm, collection, k=2)
    by_id = {r.chunk_id: r for r in results}

    for original in (whole, point):
        got = by_id[original.chunk_id]
        assert Chunk(**got.model_dump(exclude={"score", "rank"})) == original
    assert by_id["305"].point is None  # stored as "" in Chroma, mapped back


def test_question_is_nfc_normalized(
    llm: LLMClient, fake_openai: FakeOpenAI, collection: Collection, chunks: list[Chunk]
) -> None:
    index_chunks(chunks, llm, EmbeddingCache(":memory:"), collection)
    decomposed = "hamil" + "é"  # not Azerbaijani, just a combining sequence
    search(decomposed, llm, collection, k=1)
    assert fake_openai.embed_calls[-1] == ["hamil" + "é"]


def test_embedding_failure_raises_retrieval_error(
    llm: LLMClient, fake_openai: FakeOpenAI, collection: Collection
) -> None:
    fake_openai.embed_errors = [api_error(openai.AuthenticationError, 401)]
    with pytest.raises(RetrievalError, match="query embedding failed"):
        search("sual", llm, collection, k=3)
