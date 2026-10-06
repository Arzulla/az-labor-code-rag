"""End-to-end smoke test against the real OpenAI API and the real index.

Run with ``uv run pytest -m integration``; needs OPENAI_API_KEY and ``make index``.
Cost: 3 questions, about $0.01.
"""

import chromadb
import pytest

from labor_code_rag.config import load_settings
from labor_code_rag.generation.answer import answer_question
from labor_code_rag.ingest.index import collection_name, open_collection
from labor_code_rag.llm import LLMClient, create_openai_client

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pipeline() -> tuple[LLMClient, chromadb.Collection, int]:
    settings = load_settings()
    if settings.openai_api_key is None or settings.llm is None:
        pytest.skip("OPENAI_API_KEY not set")
    name = collection_name(settings.index.collection_prefix, settings.llm.embedding_model)
    client = chromadb.PersistentClient(path=str(settings.index.chroma_dir))
    if name not in [c.name for c in client.list_collections()]:
        pytest.skip(f"collection {name} missing; run `make index`")
    llm = LLMClient(create_openai_client(settings.llm, settings.openai_api_key), settings.llm)
    return llm, open_collection(client, name), settings.retrieval.k


def test_answers_with_valid_citations(pipeline: tuple[LLMClient, chromadb.Collection, int]) -> None:
    for question in ("Əmək məzuniyyəti minimum neçə gündür?", "114-cü maddə nə deyir?"):
        result = answer_question(question, *pipeline)
        assert not result.answer.refused, question
        assert result.cited_articles, question
        assert result.invalid_citations == [], question


def test_out_of_scope_is_refused(pipeline: tuple[LLMClient, chromadb.Collection, int]) -> None:
    result = answer_question("Azərbaycanda gəlir vergisi neçə faizdir?", *pipeline)
    assert result.answer.refused
    assert result.cited_articles == []
