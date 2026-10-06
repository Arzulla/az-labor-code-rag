import logging

import openai
import pytest
from pydantic import SecretStr
from tenacity import wait_none

from labor_code_rag.errors import LLMError
from labor_code_rag.llm import LLMClient, cost_usd, create_openai_client
from labor_code_rag.models import Answer

from conftest import ANSWER_MODEL, EMBED_MODEL, FakeOpenAI, api_error, make_settings, timeout_error


def _chat(llm: LLMClient) -> Answer:
    return llm.chat_structured(
        [{"role": "user", "content": "sual"}],
        Answer,
        purpose="answer",
        model=ANSWER_MODEL,
        temperature=0.0,
        prompt_version="test-v1",
    ).value


REPLY = '{"text": "Cavab (Maddə 114.1).", "citations": [], "refused": false}'


def _events(caplog: pytest.LogCaptureFixture, name: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == name]


def test_transient_error_is_retried_then_succeeds(
    llm: LLMClient, fake_openai: FakeOpenAI, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    fake_openai.chat_errors = [api_error(openai.RateLimitError, 429)]
    fake_openai.chat_replies = [REPLY]

    assert _chat(llm).text == "Cavab (Maddə 114.1)."

    assert len(fake_openai.chat_calls) == 2
    (retry,) = _events(caplog, "llm.retry")
    assert retry.levelname == "WARNING"
    assert (retry.purpose, retry.error_type, retry.attempt) == ("answer", "RateLimitError", 1)
    (called,) = _events(caplog, "llm.called")
    assert called.attempt == 2
    assert called.prompt_version == "test-v1"


@pytest.mark.parametrize(
    "error",
    [api_error(openai.InternalServerError, 503), timeout_error()],
    ids=["5xx", "timeout"],
)
def test_transient_errors_exhaust_max_attempts(
    llm: LLMClient, fake_openai: FakeOpenAI, error: Exception
) -> None:
    fake_openai.chat_errors = [error] * 3  # max_attempts=3 in make_settings

    with pytest.raises(LLMError, match="answer"):
        _chat(llm)
    assert len(fake_openai.chat_calls) == 3


@pytest.mark.parametrize(
    ("cls", "status"),
    [(openai.AuthenticationError, 401), (openai.BadRequestError, 400)],
)
def test_non_transient_error_fails_fast(
    llm: LLMClient,
    fake_openai: FakeOpenAI,
    caplog: pytest.LogCaptureFixture,
    cls: type[openai.APIStatusError],
    status: int,
) -> None:
    fake_openai.chat_errors = [api_error(cls, status)]

    with pytest.raises(LLMError, match=cls.__name__):
        _chat(llm)
    assert len(fake_openai.chat_calls) == 1
    assert not _events(caplog, "llm.retry")


def test_invalid_structured_output_fails_fast(llm: LLMClient, fake_openai: FakeOpenAI) -> None:
    # refused=true with citations violates the Answer schema.
    fake_openai.chat_replies = [
        '{"text": "x", "citations": [{"article_no": "1", "point": null}], "refused": true}'
    ]
    with pytest.raises(LLMError, match="invalid structured output"):
        _chat(llm)
    assert len(fake_openai.chat_calls) == 1


def test_model_safety_refusal_is_an_error(llm: LLMClient, fake_openai: FakeOpenAI) -> None:
    fake_openai.chat_refusal = "I can't help with that."
    with pytest.raises(LLMError, match="no parsed output"):
        _chat(llm)


def test_cost_by_hand() -> None:
    settings = make_settings()
    # 1200 * 0.40 / 1M + 300 * 1.60 / 1M = 0.00048 + 0.00048
    assert cost_usd(settings, ANSWER_MODEL, 1200, 300) == pytest.approx(0.00096)
    # 5000 * 0.02 / 1M
    assert cost_usd(settings, EMBED_MODEL, 5000, 0) == pytest.approx(0.0001)


def test_chat_logs_cost_and_tokens(
    llm: LLMClient, fake_openai: FakeOpenAI, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    fake_openai.chat_replies = [REPLY]
    _chat(llm)
    (called,) = _events(caplog, "llm.called")
    assert called.model == ANSWER_MODEL
    assert (called.prompt_tokens, called.completion_tokens) == (1200, 300)
    assert called.cost_usd == pytest.approx(0.00096)


def test_embed_batches_and_keeps_order(fake_openai: FakeOpenAI) -> None:
    llm = LLMClient(fake_openai, make_settings(embed_batch_size=2), wait=wait_none())
    texts = ["məzuniyyət", "xitam", "hamilə", "vergi", "gün"]

    result = llm.embed(texts, purpose="index")

    assert [len(b) for b in fake_openai.embed_calls] == [2, 2, 1]
    # The fake returns each batch reversed; the client must restore input order.
    assert [v.index(1.0) for v in result.vectors] == [0, 1, 2, 4, 5]
    assert result.prompt_tokens == 5


def test_embed_retries_transient_errors(fake_openai: FakeOpenAI, llm: LLMClient) -> None:
    fake_openai.embed_errors = [timeout_error()]
    assert len(llm.embed(["gün"], purpose="query_embedding").vectors) == 1
    assert len(fake_openai.embed_calls) == 2


def test_client_factory_requires_key_and_disables_sdk_retries() -> None:
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        create_openai_client(make_settings(), None)
    client = create_openai_client(make_settings(), SecretStr("sk-test"))
    assert client.max_retries == 0
