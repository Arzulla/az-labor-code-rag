"""Shared test doubles: a fake OpenAI SDK client injected into the real ``LLMClient``.

No test touches the network. The fake mimics the SDK surface ``llm.py`` uses:
``embeddings.create`` and ``chat.completions.parse``.
"""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import chromadb
import httpx2
import openai
import pytest
from chromadb.api.models.Collection import Collection
from tenacity import wait_none

from labor_code_rag.config import LLMSettings, ModelPrice
from labor_code_rag.ingest.index import open_collection
from labor_code_rag.llm import LLMClient
from labor_code_rag.models import Chunk
from labor_code_rag.text import az_lower

EMBED_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4.1-mini-2025-04-14"
JUDGE_MODEL = "gpt-4.1-2025-04-14"

# Deterministic "embedding": keyword counts, so tests can predict the nearest chunk.
VOCAB = ["məzuniyyət", "xitam", "hamilə", "əmək", "vergi", "gün"]


def fake_vector(text: str) -> list[float]:
    words = az_lower(text)
    return [float(words.count(w)) for w in VOCAB] + [0.1]  # 0.1: never a zero vector


def make_settings(**overrides: Any) -> LLMSettings:
    values: dict[str, Any] = {
        "embedding_model": EMBED_MODEL,
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "max_attempts": 3,
        "prices_usd_per_1m": {
            EMBED_MODEL: ModelPrice(input=0.02),
            ANSWER_MODEL: ModelPrice(input=0.40, output=1.60),
            JUDGE_MODEL: ModelPrice(input=2.00, output=8.00),
        },
    }
    return LLMSettings(**{**values, **overrides})


def api_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    request = httpx2.Request("POST", "https://api.test/v1")
    return cls("boom", response=httpx2.Response(status, request=request), body=None)


def timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=httpx2.Request("POST", "https://api.test/v1"))


class FakeOpenAI:
    """Queue errors with ``fail_*``; chat replies are JSON strings validated like the SDK."""

    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_errors: list[Exception] = []
        self.chat_errors: list[Exception] = []
        self.chat_replies: list[str] = []
        self.chat_refusal: str | None = None
        self.embeddings = SimpleNamespace(create=self._embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _embed(self, *, model: str, input: list[str], timeout: float) -> Any:
        self.embed_calls.append(input)
        if self.embed_errors:
            raise self.embed_errors.pop(0)
        # Reversed on purpose: llm.py must reorder by `index`.
        data = [SimpleNamespace(index=i, embedding=fake_vector(t)) for i, t in enumerate(input)]
        tokens = sum(len(t.split()) for t in input)
        return SimpleNamespace(data=data[::-1], usage=SimpleNamespace(prompt_tokens=tokens))

    def _parse(self, **kwargs: Any) -> Any:
        self.chat_calls.append(kwargs)
        if self.chat_errors:
            raise self.chat_errors.pop(0)
        parsed = None
        if self.chat_refusal is None:
            # Like the SDK: pydantic validation errors propagate to the caller.
            parsed = kwargs["response_format"].model_validate_json(self.chat_replies.pop(0))
        message = SimpleNamespace(parsed=parsed, refusal=self.chat_refusal)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=300),
        )


@pytest.fixture
def fake_openai() -> FakeOpenAI:
    return FakeOpenAI()


@pytest.fixture
def llm(fake_openai: FakeOpenAI) -> LLMClient:
    return LLMClient(fake_openai, make_settings(), wait=wait_none())


@pytest.fixture
def collection() -> Collection:
    # EphemeralClient shares state within a process: a unique name isolates each test.
    return open_collection(chromadb.EphemeralClient(), f"test_{uuid.uuid4().hex}")


DOWNLOADED_AT = datetime(2026, 10, 5, 15, 49, tzinfo=UTC)


def make_chunk(article_no: str, point: str | None, title: str, body: str) -> Chunk:
    return Chunk(
        chunk_id=f"{article_no}.{point}" if point else article_no,
        article_no=article_no,
        point=point,
        title=title,
        part="V bölmə",
        chapter="On yeddinci fəsil",
        url="https://e-qanun.az/framework/46943",
        text=f"Maddə {article_no}. {title}\n{body}",
        source_downloaded_at=DOWNLOADED_AT,
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        make_chunk("114", "1", "Əmək məzuniyyətinin müddəti", "Məzuniyyət 21 təqvim günü."),
        make_chunk("114", "2", "Əmək məzuniyyətinin müddəti", "Əlavə məzuniyyət günləri."),
        make_chunk("70", "1", "Əmək müqaviləsinə xitam", "Xitam əsasları."),
        make_chunk("79", None, "Hamilə qadınlar", "Hamilə qadınla müqaviləyə xitam olmaz."),
    ]
