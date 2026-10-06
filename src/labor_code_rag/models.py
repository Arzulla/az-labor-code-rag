"""Pydantic models shared across module boundaries (CLAUDE.md §6)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator

from labor_code_rag.text import ArticleRef


class Article(BaseModel):
    """One article of the Labor Code, as parsed from the official HTML."""

    model_config = ConfigDict(frozen=True)

    article_no: str  # str, not int: "7-1" exists
    title: str  # heading text after "Maddə N.", without [N] markers
    part: str  # "V bölmə. İstirahət vaxtı və işçilərin məzuniyyət hüquqları"
    chapter: str  # "On yeddinci fəsil. Əmək məzuniyyətlərinin müddətləri"
    text: str  # body only (no heading line); paragraphs joined with "\n"; NFC; no [N]
    footnote_ids: list[int]  # endnote numbers from title and body, in order, unique
    repealed_points: list[str]  # point numbers marked "ləğv edilmişdir": ["1", "2"]
    url: str


class Chunk(BaseModel):
    """One retrievable unit: a top-level point of an article, or a whole article (ADR-003)."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str  # "114.2", "7-1.1", "3.2-1"; "305" for an article without numbered points
    article_no: str
    point: str | None  # "2", "2-1"; None when the chunk is the whole article
    title: str
    part: str
    chapter: str
    url: str
    text: str  # what gets embedded: "Maddə <no>. <title>" + "\n" + point text
    source_downloaded_at: datetime


class RetrievedChunk(Chunk):
    """A chunk returned by a retriever, with its score and 1-based rank."""

    score: float  # cosine similarity for vector search (1 - cosine distance)
    rank: int


class Citation(BaseModel):
    """An article (and optionally point) the answering model says it relied on."""

    model_config = ConfigDict(frozen=True)

    article_no: str  # "114", "7-1"
    point: str | None  # "2"; None for an article-level citation

    def __str__(self) -> str:
        return f"Maddə {self.article_no}.{self.point}" if self.point else f"Maddə {self.article_no}"


class Answer(BaseModel):
    """Structured output of the answering model (validated, never trusted blindly)."""

    text: str  # Azerbaijani answer with inline "Maddə 114.2" citations
    citations: list[Citation]
    refused: bool  # True when the context does not contain the answer

    @model_validator(mode="after")
    def _refusal_has_no_citations(self) -> "Answer":
        # A refusal that cites articles is self-contradictory: reject it as invalid output.
        if self.refused and self.citations:
            raise ValueError("refused answer must not have citations")
        return self


class AnswerResult(BaseModel):
    """What the pipeline returns to the UI / scripts for one question."""

    request_id: str
    question: str
    answer: Answer
    retrieved: list[RetrievedChunk]
    cited_articles: list[ArticleRef]  # citations that passed the check
    invalid_citations: list[ArticleRef]  # cited but not retrieved (logged as citation.invalid)
    latency_ms: int
    cost_usd: float
