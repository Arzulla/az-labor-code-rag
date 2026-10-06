from labor_code_rag.app import format_answer, format_sources
from labor_code_rag.models import Answer, AnswerResult, Chunk, RetrievedChunk
from labor_code_rag.text import ArticleRef


def _result(chunks: list[Chunk]) -> AnswerResult:
    retrieved = [
        RetrievedChunk(**c.model_dump(), score=0.9 - i / 10, rank=i + 1)
        for i, c in enumerate(chunks)
    ]
    return AnswerResult(
        request_id="req-1",
        question="sual",
        answer=Answer(text="21 gün (Maddə 114.1).", citations=[], refused=False),
        retrieved=retrieved,
        cited_articles=[ArticleRef(article="114", point="1")],
        invalid_citations=[ArticleRef(article="999")],
        latency_ms=1234,
        cost_usd=0.0012,
    )


def test_answer_links_cited_articles_and_flags_invalid(chunks: list[Chunk]) -> None:
    text = format_answer(_result(chunks))
    assert "[Maddə 114.1](https://e-qanun.az/framework/46943)" in text
    assert "Maddə 999" in text.split("⚠️")[1]


def test_sources_show_rank_score_and_request_id(chunks: list[Chunk]) -> None:
    text = format_sources(_result(chunks))
    assert "**1. 114.1** · score 0.900" in text
    assert "request_id=req-1" in text
