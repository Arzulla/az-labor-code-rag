import logging

import pytest

from labor_code_rag.generation.citations import check_citations
from labor_code_rag.models import Answer, Chunk, Citation
from labor_code_rag.text import ArticleRef


def _answer(text: str, citations: list[Citation] | None = None) -> Answer:
    return Answer(text=text, citations=citations or [], refused=False)


def _check(text: str, chunks: list[Chunk]) -> tuple[list[str], list[str]]:
    result = check_citations(_answer(text), chunks)
    return [str(r) for r in result.valid], [str(r) for r in result.invalid]


def test_valid_point(chunks: list[Chunk]) -> None:
    assert _check("Müddət 21 gündür (Maddə 114.1).", chunks) == (["Maddə 114.1"], [])


def test_point_not_retrieved_is_invalid(chunks: list[Chunk]) -> None:
    # Article 114 was retrieved, but only points 1 and 2.
    assert _check("Bax Maddə 114.5.", chunks) == ([], ["Maddə 114.5"])


def test_valid_article_level(chunks: list[Chunk]) -> None:
    assert _check("Maddə 70 xitamı tənzimləyir.", chunks) == (["Maddə 70"], [])


def test_sub_point_inside_retrieved_point_is_valid(chunks: list[Chunk]) -> None:
    assert _check("Maddə 114.2.1 üzrə", chunks) == (["Maddə 114.2.1"], [])


def test_point_of_whole_article_chunk_is_valid(chunks: list[Chunk]) -> None:
    # Article 79 is one chunk (no numbered points): a point ref cannot be checked finer.
    assert _check("Maddə 79.1", chunks) == (["Maddə 79.1"], [])


def test_invalid_article_is_logged_with_level(
    chunks: list[Chunk], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    assert _check("Maddə 999 və Maddə 998.3 deyir.", chunks) == ([], ["Maddə 999", "Maddə 998.3"])

    records = [r for r in caplog.records if r.getMessage() == "citation.invalid"]
    assert [(r.cited, r.level) for r in records] == [
        ("Maddə 999", "article"),
        ("Maddə 998.3", "point"),
    ]
    assert records[0].retrieved_article_nos == ["114", "70", "79"]


def test_ref_in_suffix_form(chunks: list[Chunk]) -> None:
    assert _check("114-cü maddəyə görə", chunks) == (["Maddə 114"], [])
    assert _check("120-ci maddəyə görə", chunks) == ([], ["Maddə 120"])


def test_structured_citations_are_checked_too(chunks: list[Chunk]) -> None:
    answer = _answer(
        "Cavab (Maddə 114.1).",
        [Citation(article_no="114", point="1"), Citation(article_no="500", point=None)],
    )
    result = check_citations(answer, chunks)
    assert result.valid == [ArticleRef(article="114", point="1")]  # deduplicated
    assert result.invalid == [ArticleRef(article="500")]


def test_no_refs_no_findings(chunks: list[Chunk]) -> None:
    assert _check("Heç bir istinad yoxdur.", chunks) == ([], [])
