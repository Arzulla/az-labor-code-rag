"""Split parsed articles into retrievable chunks (``make ingest``), see ADR-003.

Rule: one chunk per top-level point of an article; an article without numbered points is
one chunk. Top-level points look like ``2.``, ``2-1.`` or, in a few articles, carry the
article number as a prefix (``7-1.1.`` in Maddə 7-1 is point ``1``). Deeper dotted points
(``7-1.1.1.``, ``2-3.1.``), lettered sub-points (``a)`` … ``ç)``), unnumbered list lines
and a trailing ``Qeyd:`` note stay inside the point they follow. Repealed points
(``ləğv edilmişdir``) produce no chunk.

Each chunk's ``text`` starts with the article heading, so a point retrieved alone still
says which article it belongs to.
"""

import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

from labor_code_rag.config import load_settings
from labor_code_rag.errors import ParseError, SourceDownloadError
from labor_code_rag.ingest.download import SourceMeta, raw_html_path
from labor_code_rag.ingest.parse import (
    decode_source,
    extract_paragraphs,
    parse_articles,
    validate_articles,
)
from labor_code_rag.logging_setup import configure_logging, set_request_id
from labor_code_rag.models import Article, Chunk

# Explicit name: under `python -m` __name__ would be "__main__".
logger = logging.getLogger("labor_code_rag.ingest.chunk")

# "2. Mətn", "2-1. Mətn". Not "2-3.1. Mətn": the dot must be followed by whitespace.
_TOP_POINT_RE = re.compile(r"(\d+(?:-\d+)?)\.\s")

CHUNKS_FILE = "chunks.jsonl"


def top_level_point(line: str, article_no: str) -> str | None:
    """Return the top-level point number a line starts, or None for any other line.

    >>> top_level_point("2-1. Mətn", "3"), top_level_point("7-1.1. Mətn", "7-1")
    ('2-1', '1')
    >>> top_level_point("7-1.1.1. Mətn", "7-1"), top_level_point("a) mətn", "114")
    (None, None)
    """
    # Strip "<article_no>." only when a digit follows: in Maddə 3, "3. İşəgötürən" is
    # point 3, while in Maddə 7-1, "7-1.1." is point 1.
    prefix = f"{article_no}."
    if line.startswith(prefix) and line[len(prefix) : len(prefix) + 1].isdigit():
        line = line[len(prefix) :]
    match = _TOP_POINT_RE.match(line)
    return match[1] if match else None


def split_points(article: Article) -> list[tuple[str | None, str]]:
    """Split the article body into ``(point, text)`` pairs, in order.

    Lines that do not start a top-level point join the current point. Lines before the
    first point (none in the current law) join the first point. An article without
    numbered points comes back as one ``(None, text)`` pair.
    """
    groups: list[tuple[str | None, list[str]]] = []
    preamble: list[str] = []
    for line in article.text.split("\n"):
        point = top_level_point(line, article.article_no)
        if point is not None:
            groups.append((point, [*preamble, line]))
            preamble = []
        elif groups:
            groups[-1][1].append(line)
        else:
            preamble.append(line)
    if not groups:
        return [(None, "\n".join(preamble))]
    return [(point, "\n".join(lines)) for point, lines in groups]


def chunk_header(article: Article) -> str:
    return f"Maddə {article.article_no}. {article.title}"


def chunk_article(article: Article, source_downloaded_at: datetime) -> list[Chunk]:
    """Turn one article into chunks; repealed points are skipped."""
    header = chunk_header(article)
    return [
        Chunk(
            chunk_id=f"{article.article_no}.{point}" if point else article.article_no,
            article_no=article.article_no,
            point=point,
            title=article.title,
            part=article.part,
            chapter=article.chapter,
            url=article.url,
            text=f"{header}\n{text}",
            source_downloaded_at=source_downloaded_at,
        )
        for point, text in split_points(article)
        if point not in article.repealed_points
    ]


def chunk_articles(articles: list[Article], source_downloaded_at: datetime) -> list[Chunk]:
    return [c for a in articles for c in chunk_article(a, source_downloaded_at)]


def validate_chunks(chunks: list[Chunk]) -> None:
    """Sanity-check the chunks.

    Raises:
        ParseError: the list is empty, or a ``chunk_id`` repeats (a point numbered twice
            in one article would silently shadow another in the index).
    """
    if not chunks:
        raise ParseError("no chunks produced")
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            raise ParseError(f"duplicate chunk id: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    """One Chunk per line, UTF-8, Azerbaijani letters unescaped; atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (c.model_dump_json(ensure_ascii=False) + "\n" for c in chunks)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    os.replace(tmp, path)


def run_ingest(html_path: Path, meta_path: Path, out_path: Path) -> list[Chunk]:
    """raw HTML -> articles -> chunks -> JSONL; logs ``ingest.completed``."""
    if not meta_path.exists():
        # download.py writes meta.json last: without it the HTML may be incomplete.
        raise SourceDownloadError(f"{meta_path} not found; run `make data` first")
    started = time.perf_counter()
    meta = SourceMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
    paragraphs = extract_paragraphs(decode_source(html_path.read_bytes()))
    articles = parse_articles(paragraphs, meta.ui_url)
    validate_articles(articles)
    chunks = chunk_articles(articles, meta.downloaded_at)
    validate_chunks(chunks)
    write_chunks(chunks, out_path)
    logger.info(
        "ingest.completed",
        extra={
            "articles": len(articles),
            "chunks": len(chunks),
            "repealed_points": sum(len(a.repealed_points) for a in articles),
            "latency_ms": round((time.perf_counter() - started) * 1000),
        },
    )
    return chunks


def main() -> None:
    settings = load_settings()
    configure_logging(
        settings.logging.level, settings.logging.format, settings.logging.log_payloads
    )
    set_request_id()
    if settings.data is None:
        raise SourceDownloadError("missing 'data' section in config.yaml")
    data = settings.data
    html_path = raw_html_path(data.source_url, data.raw_dir)
    run_ingest(html_path, html_path.with_suffix(".meta.json"), data.processed_dir / CHUNKS_FILE)


if __name__ == "__main__":
    main()
