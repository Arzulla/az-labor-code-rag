import json
import logging

import openai
import pytest
from chromadb.api.models.Collection import Collection
from pydantic import ValidationError

from labor_code_rag.embedding_cache import EmbeddingCache
from labor_code_rag.errors import LLMError, RetrievalError
from labor_code_rag.generation.answer import answer_question
from labor_code_rag.generation.prompts import REFUSAL_TEXT
from labor_code_rag.ingest.index import index_chunks
from labor_code_rag.llm import LLMClient
from labor_code_rag.logging_setup import get_request_id
from labor_code_rag.models import Answer, Chunk, Citation
from labor_code_rag.text import ArticleRef

from conftest import FakeOpenAI, api_error

QUESTION = "Əmək məzuniyyəti neçə gündür?"


@pytest.fixture
def indexed(llm: LLMClient, collection: Collection, chunks: list[Chunk]) -> Collection:
    index_chunks(chunks, llm, EmbeddingCache(":memory:"), collection)
    return collection


def _reply(text: str, citations: list[tuple[str, str | None]], refused: bool = False) -> str:
    cites = [{"article_no": a, "point": p} for a, p in citations]
    return json.dumps({"text": text, "citations": cites, "refused": refused}, ensure_ascii=False)


def _events(caplog: pytest.LogCaptureFixture, name: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == name]


def test_answer_with_valid_citation(
    llm: LLMClient,
    fake_openai: FakeOpenAI,
    indexed: Collection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    fake_openai.chat_replies = [_reply("21 təqvim günü (Maddə 114.1).", [("114", "1")])]

    result = answer_question(QUESTION, llm, indexed, k=3)

    assert not result.answer.refused
    assert result.cited_articles == [ArticleRef(article="114", point="1")]
    assert result.invalid_citations == []
    assert len(result.retrieved) == 3
    assert fake_openai.chat_calls[0]["temperature"] == 0.0

    (completed,) = _events(caplog, "request.completed")
    assert completed.cited_articles == ["Maddə 114.1"]
    assert completed.total_latency_ms >= 0
    # 1 query embedding (fake: 4 words = 4 tokens) + 1 chat call (1200 in, 300 out)
    assert completed.total_cost_usd == pytest.approx(4 * 0.02e-6 + 0.00096)
    assert result.cost_usd == pytest.approx(completed.total_cost_usd)
    # answer_question binds a fresh request_id; every log line of the request carries it.
    assert result.request_id == get_request_id() != "-"
    assert _events(caplog, "request.received")[0].question_chars == len(QUESTION)


def test_model_refusal_returns_fixed_sentence(
    llm: LLMClient,
    fake_openai: FakeOpenAI,
    indexed: Collection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    fake_openai.chat_replies = [_reply("Bilmirəm.", [], refused=True)]

    result = answer_question("Gəlir vergisi neçə faizdir?", llm, indexed, k=3)

    assert result.answer.refused
    assert result.answer.text == REFUSAL_TEXT
    assert result.cited_articles == []
    (refused,) = _events(caplog, "answer.refused")
    assert refused.reason == "model_refused"


def test_empty_collection_refuses_without_llm_call(
    llm: LLMClient,
    fake_openai: FakeOpenAI,
    collection: Collection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    result = answer_question(QUESTION, llm, collection, k=3)
    assert result.answer.refused
    assert fake_openai.chat_calls == []
    assert _events(caplog, "answer.refused")[0].reason == "no_context"


def test_invalid_citation_is_logged_and_returned(
    llm: LLMClient,
    fake_openai: FakeOpenAI,
    indexed: Collection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    fake_openai.chat_replies = [_reply("Bax Maddə 114.1 və 250-ci maddə.", [("114", "1")])]

    result = answer_question(QUESTION, llm, indexed, k=3)

    assert result.invalid_citations == [ArticleRef(article="250")]
    assert len(_events(caplog, "citation.invalid")) == 1
    (completed,) = _events(caplog, "request.completed")
    assert completed.invalid_citations == 1


@pytest.mark.parametrize(
    ("stage", "expected"),
    [("generation", LLMError), ("retrieval", RetrievalError)],
)
def test_failure_logs_request_failed_with_stage(
    llm: LLMClient,
    fake_openai: FakeOpenAI,
    indexed: Collection,
    caplog: pytest.LogCaptureFixture,
    stage: str,
    expected: type[Exception],
) -> None:
    error = api_error(openai.AuthenticationError, 401)
    if stage == "generation":
        fake_openai.chat_errors = [error]
    else:
        fake_openai.embed_errors = [error]
    with pytest.raises(expected):
        answer_question(QUESTION, llm, indexed, k=3)
    (failed,) = _events(caplog, "request.failed")
    assert failed.levelname == "ERROR"
    assert failed.stage == stage
    assert failed.exc_info is not None


def test_refused_answer_must_not_cite() -> None:
    with pytest.raises(ValidationError, match="refused answer must not have citations"):
        Answer(text="x", citations=[Citation(article_no="1", point=None)], refused=True)
