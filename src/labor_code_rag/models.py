"""Pydantic models shared across module boundaries (CLAUDE.md §6)."""

from datetime import datetime
from typing import Any, Literal

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


# --- Evaluation (Phase 3, ADR-006 / ADR-007) -----------------------------------------------

Category = Literal["factual", "exact_article", "colloquial", "multi_article", "out_of_scope"]
CATEGORIES: tuple[Category, ...] = (
    "factual",
    "exact_article",
    "colloquial",
    "multi_article",
    "out_of_scope",
)


def article_of(chunk_id: str) -> str:
    """Article number of a chunk id: ``"114.2"`` -> ``"114"``, ``"7-1.1"`` -> ``"7-1"``."""
    return chunk_id.split(".", 1)[0]


class GoldenItem(BaseModel):
    """One golden-set question. Article-level relevance is derived from ``relevant_chunks``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str  # "fact-01"; unique across splits
    question: str  # Azerbaijani, as a user would ask it
    expected_answer: str  # short Azerbaijani reference answer with "Maddə N.P" refs
    relevant_chunks: list[str]  # chunk ids ("114.2", "70"); empty for out_of_scope
    category: Category
    verified: bool = False  # set to true only by the owner, after checking against the law
    notes: str = ""

    @model_validator(mode="after")
    def _out_of_scope_has_no_chunks(self) -> "GoldenItem":
        if (self.category == "out_of_scope") != (not self.relevant_chunks):
            raise ValueError(f"{self.id}: out_of_scope <=> empty relevant_chunks")
        return self

    @property
    def relevant_articles(self) -> list[str]:
        """Article numbers of ``relevant_chunks``, deduplicated, order kept."""
        return list(dict.fromkeys(article_of(c) for c in self.relevant_chunks))


Score = Literal[1, 2, 3, 4, 5]


class JudgeScore(BaseModel):
    """Structured output of the LLM judge (rubric in ``generation/prompts.py``)."""

    accuracy: Score
    completeness: Score
    relevance: Score
    rationale: str


class EvalItemResult(BaseModel):
    """Per-question details saved in every eval result file."""

    id: str
    category: Category
    question: str
    relevant_chunks: list[str]
    request_id: str
    retrieved_chunk_ids: list[str]  # rank order
    retrieved_article_nos: list[str]  # rank order, deduplicated (first occurrence)
    answer: str | None = None  # None in --retrieval-only runs
    refused: bool | None = None
    cited: list[ArticleRef] = []  # every ref in the answer (text + structured list)
    invalid_citations: list[ArticleRef] = []  # cited but not retrieved (citations.py)
    judge: JudgeScore | None = None
    judge_skipped: Literal["out_of_scope", "refused"] | None = None
    latency_ms: int
    cost_usd: float  # answer pipeline + judge call


class GoldenVerified(BaseModel):
    verified_n: int
    total_n: int


class EvalResult(BaseModel):
    """One eval run, written to ``eval/results/<UTC timestamp>_<label>_<split>.json``."""

    label: str
    split: Literal["dev", "test"]
    created_at: datetime
    git_sha: str
    git_dirty: bool
    config: dict[str, Any]  # settings snapshot, secrets excluded
    prompt_version: str
    judge_prompt_version: str
    embedding_model: str
    answer_model: str
    judge_model: str
    k: int
    retrieval_only: bool
    golden_file: str
    golden_sha256: str
    golden_verified: GoldenVerified
    metrics: dict[str, float | None]
    metrics_by_category: dict[str, dict[str, float | None]]
    items: list[EvalItemResult]
    total_cost_usd: float
    wall_time_s: float
    budget_exceeded: bool
    stopped_reason: str | None  # "budget_exceeded" or an error; None for a complete run
