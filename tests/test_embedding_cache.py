from pathlib import Path

from labor_code_rag.embedding_cache import EmbeddingCache, embed_with_cache, text_sha256
from labor_code_rag.llm import LLMClient

from conftest import EMBED_MODEL, FakeOpenAI, fake_vector


def test_second_run_makes_zero_embed_calls(llm: LLMClient, fake_openai: FakeOpenAI) -> None:
    cache = EmbeddingCache(":memory:")
    texts = ["məzuniyyət günü", "xitam", "hamilə"]

    first, hits1, cost1 = embed_with_cache(texts, llm, cache, purpose="index")
    calls_after_first = len(fake_openai.embed_calls)
    second, hits2, cost2 = embed_with_cache(texts, llm, cache, purpose="index")

    assert calls_after_first == 1
    assert len(fake_openai.embed_calls) == calls_after_first  # zero new calls
    assert (hits1, hits2) == (0, 3)
    assert cost1 > 0 and cost2 == 0
    assert first == second == [fake_vector(t) for t in texts]


def test_only_changed_text_is_embedded(llm: LLMClient, fake_openai: FakeOpenAI) -> None:
    cache = EmbeddingCache(":memory:")
    embed_with_cache(["a", "b"], llm, cache, purpose="index")

    _, hits, _ = embed_with_cache(["a", "b changed"], llm, cache, purpose="index")

    assert hits == 1
    assert fake_openai.embed_calls[-1] == ["b changed"]


def test_duplicate_texts_are_embedded_once(llm: LLMClient, fake_openai: FakeOpenAI) -> None:
    vectors, _, _ = embed_with_cache(["x", "x"], llm, EmbeddingCache(":memory:"), "index")
    assert fake_openai.embed_calls == [["x"]]
    assert len(vectors) == 2


def test_cache_key_includes_model_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "emb.sqlite"
    cache = EmbeddingCache(path)
    cache.put_many(EMBED_MODEL, {text_sha256("x"): [1.0, 2.0]})
    cache.close()

    reopened = EmbeddingCache(path)
    assert reopened.get_many(EMBED_MODEL, [text_sha256("x")]) == {text_sha256("x"): [1.0, 2.0]}
    assert reopened.get_many("other-model", [text_sha256("x")]) == {}
