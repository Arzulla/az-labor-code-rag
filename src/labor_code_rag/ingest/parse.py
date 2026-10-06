"""Parse the official Labor Code HTML into articles.

Pipeline: ``decode_source`` (bytes -> str) -> ``extract_paragraphs`` (HTML -> plain-text
paragraphs) -> ``parse_articles`` (paragraphs -> ``Article``) -> ``validate_articles``.

Source format: Word-exported HTML in windows-1251; Azerbaijani letters are HTML entities
(``&#601;`` = ``ə``). Each ``<p>`` is one paragraph; Word wraps long lines with ``\\r\\n``
inside a paragraph. Endnote references look like ``[25]``. Text removed by amendments
is kept in the source but struck through (``<s>...</s>``); it is not law in force, so it
is dropped (ADR-001). The law body ends at the ``ƏLAVƏLƏR`` heading; the endnote list
after it quotes old wordings (``Maddə 49. ...``) that must not become articles.
"""

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from labor_code_rag.errors import ParseError
from labor_code_rag.models import Article
from labor_code_rag.text import normalize

_BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP_TAGS = {"style", "script"}

_WHITESPACE_RE = re.compile(r"\s+")
_FOOTNOTE_RE = re.compile(r"\s*\[(\d+)\]")
_ARTICLE_RE = re.compile(r"Maddə (\d+(?:-\d+)?)\.\s*(.*)")
_PART_RE = re.compile(r"[IVXL]+ bölmə")
_CHAPTER_RE = re.compile(r"[^\W\d_]+(?: [^\W\d_]+)* fəsil")  # "On yeddinci fəsil"
_REPEALED_RE = re.compile(r"(\d+)\. ləğv edilmişdir\.?")

# The appendices heading is split over two paragraphs in the source.
_END_HEADING = "ƏLAVƏLƏR"
_END_HEADING_PREFIX = "Azərbaycan Respublikasının Əmək Məcəlləsinə"


def decode_source(raw: bytes) -> str:
    """Decode the raw windows-1251 source into a str.

    HTML entities are left as they are; ``extract_paragraphs`` resolves them.

    Raises:
        ParseError: the bytes are not valid windows-1251.
    """
    try:
        return raw.decode("windows-1251")
    except UnicodeDecodeError as e:
        raise ParseError(f"source is not valid windows-1251: {e}") from e


class _ParagraphCollector(HTMLParser):
    """SAX-style handler: collects the text of block elements, skipping unwanted text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)  # entities arrive already decoded
        self.paragraphs: list[str] = []
        self._in_body = False
        self._skip_depth = 0  # inside <style>/<script>
        self._strike_depth = 0  # inside <s>; a counter, because tags can nest
        self._buffer: list[str] | None = None  # not None while inside a block element

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "body":
            self._in_body = True
        elif tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "s":
            self._strike_depth += 1
        elif tag in _BLOCK_TAGS and self._in_body:
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "s":
            self._strike_depth = max(0, self._strike_depth - 1)
        elif tag in _BLOCK_TAGS and self._buffer is not None:
            text = normalize(_WHITESPACE_RE.sub(" ", "".join(self._buffer)).strip())
            if text:
                self.paragraphs.append(text)
            self._buffer = None

    def handle_data(self, data: str) -> None:
        if self._buffer is not None and not self._skip_depth and not self._strike_depth:
            self._buffer.append(data)


def extract_paragraphs(html: str) -> list[str]:
    """Return the plain text of every ``<p>`` and ``<h1>``-``<h6>`` inside ``<body>``, in order.

    Headings matter: the source has a few article headings as ``<h3>`` (Maddə 298).

    - HTML entities resolved (``&#601;`` -> ``ə``), text normalized to NFC
      (``labor_code_rag.text.normalize``).
    - Whitespace (including ``\\r\\n`` and ``&nbsp;``) collapsed to single spaces, stripped.
    - Struck-through text (inside ``<s>``) is dropped: it was removed by an amendment.
    - ``<style>``, ``<script>``, comments and Word conditional comments
      (``<!--[if gte mso 9]>...``) never produce text. Text inside downlevel-revealed
      blocks (``<![if !supportFootnotes]>[25]<![endif]>``) is kept: it is the endnote marker.
    - Empty paragraphs dropped.
    """
    collector = _ParagraphCollector()
    collector.feed(html)
    collector.close()
    return collector.paragraphs


def strip_footnote_markers(text: str) -> tuple[str, list[int]]:
    """Remove ``[N]`` endnote markers from ``text``.

    Returns the cleaned text (whitespace tidied where a marker was removed) and the
    marker numbers in order of appearance, without duplicates.

    >>> strip_footnote_markers("2. ləğv edilmişdir.[351]")
    ('2. ləğv edilmişdir.', [351])
    """
    ids = list(dict.fromkeys(int(n) for n in _FOOTNOTE_RE.findall(text)))
    cleaned = _WHITESPACE_RE.sub(" ", _FOOTNOTE_RE.sub(" ", text)).strip()
    return cleaned, ids


@dataclass
class _OpenArticle:
    """An article whose heading was seen and whose body is still being collected."""

    article_no: str
    title: str
    part: str
    chapter: str
    body: list[str] = field(default_factory=list)


def _build_article(open_article: _OpenArticle, url: str) -> Article:
    title, footnote_ids = strip_footnote_markers(open_article.title)
    lines: list[str] = []
    repealed_points: list[str] = []
    for paragraph in open_article.body:
        line, ids = strip_footnote_markers(paragraph)
        lines.append(line)
        footnote_ids.extend(i for i in ids if i not in footnote_ids)
        if match := _REPEALED_RE.fullmatch(line):
            repealed_points.append(match[1])
    return Article(
        article_no=open_article.article_no,
        title=title,
        part=open_article.part,
        chapter=open_article.chapter,
        text="\n".join(lines),
        footnote_ids=footnote_ids,
        repealed_points=repealed_points,
        url=url,
    )


def _is_end_of_body(paragraphs: list[str], i: int) -> bool:
    if paragraphs[i] == _END_HEADING:
        return True
    next_paragraph = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
    return paragraphs[i] == _END_HEADING_PREFIX and next_paragraph == _END_HEADING


def parse_articles(paragraphs: list[str], url: str) -> list[Article]:
    """Group paragraphs into articles.

    - An article starts at a heading ``Maddə <no>. <title>`` (``<no>`` may be dashed:
      ``7-1``); following paragraphs up to the next heading are its body.
    - ``N bölmə`` / ``... fəsil`` headings, each followed by a title paragraph, set the
      ``part`` / ``chapter`` of the articles after them, as ``"<heading>. <title>"``
      (``"V bölmə. İstirahət vaxtı ..."``), without ``[N]`` markers.
    - Paragraphs before the first heading (decree text, law title) are skipped.
    - Parsing stops at the ``ƏLAVƏLƏR`` heading (appendices and endnotes follow).
    - Paragraphs that are empty once ``[N]`` markers are removed are skipped. They are
      leftovers of struck-through text: Maddə 298 is repealed and only its ``[584]``
      remains, which must not attach to the previous article.
    - ``[N]`` markers go to ``footnote_ids``; points whose text is ``ləğv edilmişdir`` go
      to ``repealed_points``.

    Raises:
        ParseError: an article heading appears before any bölmə/fəsil heading.
    """
    articles: list[Article] = []
    current: _OpenArticle | None = None
    part: str | None = None
    chapter: str | None = None
    awaiting_title: str | None = None  # "part" or "chapter": the next paragraph is its title

    def close_current() -> None:
        nonlocal current
        if current is not None:
            articles.append(_build_article(current, url))
            current = None

    for i, paragraph in enumerate(paragraphs):
        if _is_end_of_body(paragraphs, i):
            break
        if not strip_footnote_markers(paragraph)[0]:
            continue

        if awaiting_title is not None:
            title = strip_footnote_markers(paragraph)[0]
            if awaiting_title == "part":
                part = f"{part}. {title}"
            else:
                chapter = f"{chapter}. {title}"
            awaiting_title = None
        elif _PART_RE.fullmatch(paragraph):
            close_current()
            part, chapter, awaiting_title = paragraph, None, "part"
        elif _CHAPTER_RE.fullmatch(paragraph):
            close_current()
            chapter, awaiting_title = paragraph, "chapter"
        elif match := _ARTICLE_RE.fullmatch(paragraph):
            close_current()
            if part is None or chapter is None:
                raise ParseError(f"Maddə {match[1]} appears before any bölmə/fəsil heading")
            current = _OpenArticle(match[1], match[2], part, chapter)
        elif current is not None:
            current.body.append(paragraph)

    close_current()
    return articles


def validate_articles(articles: list[Article]) -> None:
    """Sanity-check the parsed articles.

    Raises:
        ParseError: the list is empty, or an ``article_no`` repeats.
    """
    if not articles:
        raise ParseError("no articles parsed")
    seen: set[str] = set()
    for article in articles:
        if article.article_no in seen:
            raise ParseError(f"duplicate article number: {article.article_no}")
        seen.add(article.article_no)
