"""Post-generation citation check (CLAUDE.md §4, ADR-005).

Every article the answer cites must be in the retrieved set; otherwise the model cited
something it was not shown (hallucinated or from memory). Refs come from two places: the
answer text, parsed with ``text.parse_article_refs`` (so ``114-cü maddə`` counts too), and
the structured ``citations`` list. Invalid refs are logged and returned, never dropped.

Rules (lenient on purpose for the baseline; Phase 3 measures point-level accuracy):
- ``Maddə N`` (article level) is valid if any chunk of article N was retrieved.
- ``Maddə N.P`` is valid if chunk ``N.<top-level part of P>`` was retrieved, or article N
  was retrieved as one whole chunk (it has no numbered points to be more precise about).
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from labor_code_rag.models import Answer, Chunk
from labor_code_rag.text import ArticleRef, parse_article_refs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CitationCheck:
    valid: list[ArticleRef]
    invalid: list[ArticleRef]


def answer_refs(answer: Answer) -> list[ArticleRef]:
    """Refs from the text first, then the structured list; order kept, duplicates dropped."""
    from_list = [ArticleRef(article=c.article_no, point=c.point) for c in answer.citations]
    return list(dict.fromkeys([*parse_article_refs(answer.text), *from_list]))


def is_supported(ref: ArticleRef, retrieved: Sequence[Chunk]) -> bool:
    if ref.point is None:
        return any(c.article_no == ref.article for c in retrieved)
    top_point = ref.point.split(".")[0]  # "2.1" (Maddə 114.2.1) is inside chunk 114.2
    return any(c.article_no == ref.article and c.point in (top_point, None) for c in retrieved)


def check_citations(answer: Answer, retrieved: Sequence[Chunk]) -> CitationCheck:
    valid: list[ArticleRef] = []
    invalid: list[ArticleRef] = []
    for ref in answer_refs(answer):
        (valid if is_supported(ref, retrieved) else invalid).append(ref)
    for ref in invalid:
        logger.warning(
            "citation.invalid",
            extra={
                "cited": str(ref),
                "level": "point" if ref.point else "article",
                "retrieved_article_nos": list(dict.fromkeys(c.article_no for c in retrieved)),
                "retrieved_chunk_ids": [c.chunk_id for c in retrieved],
            },
        )
    return CitationCheck(valid, invalid)
