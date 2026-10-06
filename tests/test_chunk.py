import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from labor_code_rag.errors import ParseError, SourceDownloadError
from labor_code_rag.ingest.chunk import (
    chunk_article,
    chunk_articles,
    run_ingest,
    split_points,
    top_level_point,
    validate_chunks,
)
from labor_code_rag.ingest.download import SourceMeta
from labor_code_rag.ingest.parse import decode_source, extract_paragraphs, parse_articles
from labor_code_rag.models import Article, Chunk

FIXTURE = Path(__file__).parent / "fixtures" / "labor_code_sample.html"
FULL_SOURCE = Path(__file__).parents[1] / "data" / "raw" / "f_46943.html"
URL = "https://e-qanun.az/framework/46943"
DOWNLOADED_AT = datetime(2026, 10, 5, 15, 49, tzinfo=UTC)


@pytest.fixture(scope="module")
def articles() -> list[Article]:
    return parse_articles(extract_paragraphs(decode_source(FIXTURE.read_bytes())), URL)


@pytest.fixture(scope="module")
def chunks(articles: list[Article]) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunk_articles(articles, DOWNLOADED_AT)}


def make_article(article_no: str, text: str, repealed_points: list[str] | None = None) -> Article:
    return Article(
        article_no=article_no,
        title="Başlıq",
        part="I bölmə. Ümumi normalar",
        chapter="Birinci fəsil. Əsas müddəalar",
        text=text,
        footnote_ids=[],
        repealed_points=repealed_points or [],
        url=URL,
    )


def body(chunk: Chunk) -> str:
    """Chunk text without its header line."""
    return chunk.text.split("\n", 1)[1]


# --- top_level_point ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "article_no", "expected"),
    [
        ("2. Mətn", "114", "2"),
        ("2-1. Mətn", "3", "2-1"),  # dashed point inserted by an amendment
        ("7-1.1. Mətn", "7-1", "1"),  # point carrying the article number as prefix
        ("7-1. Mətn", "3", "7-1"),  # same characters, but point 7-1 of Maddə 3
        ("3. İşəgötürən", "3", "3"),  # prefix "3." followed by a space: not a prefix
        ("7-1.1.1. Mətn", "7-1", None),  # deeper dotted point
        ("2-3.1. Mətn", "7", None),  # sub-point of dashed point 2-3
        ("a) mətn;", "114", None),
        ("Qeyd: Bu maddədə ...", "70", None),
        ("60 və daha yuxarı yaşda", "67-2", None),
    ],
)
def test_top_level_point(line: str, article_no: str, expected: str | None) -> None:
    assert top_level_point(line, article_no) == expected


# --- split_points ---------------------------------------------------------------------


def test_split_points_keeps_sub_points_and_notes_in_parent() -> None:
    article = make_article(
        "7",
        "1. Bir.\n2-3. Aşağıdakı hallarda:\n2-3.1. birinci;\n2-3.2. ikinci.\n"
        "3. Üç:\nbir ilədək;\nbir ildən çox.\nQeyd: Bu maddədə tərif.",
    )
    assert split_points(article) == [
        ("1", "1. Bir."),
        ("2-3", "2-3. Aşağıdakı hallarda:\n2-3.1. birinci;\n2-3.2. ikinci."),
        ("3", "3. Üç:\nbir ilədək;\nbir ildən çox.\nQeyd: Bu maddədə tərif."),
    ]


def test_split_points_joins_preamble_to_first_point() -> None:
    article = make_article("5", "Giriş:\n1. Bir.\n2. İki.")
    assert split_points(article) == [("1", "Giriş:\n1. Bir."), ("2", "2. İki.")]


def test_split_points_article_without_numbered_points() -> None:
    article = make_article("9", "Hüquqlar:\na) bir;\nb) iki.")
    assert split_points(article) == [(None, "Hüquqlar:\na) bir;\nb) iki.")]


# --- chunk_article on the fixture -----------------------------------------------------


def test_chunk_ids_for_fixture(chunks: dict[str, Chunk]) -> None:
    assert list(chunks) == [
        "7-1.1",
        "7-1.2",
        "114.1",
        "114.2",
        "114.3",
        "124.3",
        "305",
        "317.1",
        "317.2",
    ]


def test_article_114_splits_into_points_with_letters_inside_point_3(
    chunks: dict[str, Chunk],
) -> None:
    assert [chunks[f"114.{p}"].point for p in "123"] == ["1", "2", "3"]
    point_3 = body(chunks["114.3"]).split("\n")
    assert point_3[0].startswith("3. ")
    assert [line[:2] for line in point_3[1:]] == ["a)", "b)", "c)", "ç)", "d)", "e)"]
    for p in "12":
        assert not re.search(r"^[a-zç]\) ", body(chunks[f"114.{p}"]), re.MULTILINE)


def test_article_7_1_dotted_points(chunks: dict[str, Chunk]) -> None:
    lines = body(chunks["7-1.1"]).split("\n")
    assert [line.split(" ", 1)[0] for line in lines] == ["7-1.1.", "7-1.1.1.", "7-1.1.2."]
    assert body(chunks["7-1.2"]).startswith("7-1.2. ")
    assert chunks["7-1.1"].article_no == "7-1"
    assert chunks["7-1.1"].point == "1"


def test_article_124_repealed_points_are_skipped(chunks: dict[str, Chunk]) -> None:
    assert "124.1" not in chunks and "124.2" not in chunks
    assert "ləğv edilmişdir" not in "".join(c.text for c in chunks.values())
    assert body(chunks["124.3"]).startswith("3. ")


def test_article_without_points_is_one_chunk(chunks: dict[str, Chunk]) -> None:
    chunk = chunks["305"]
    assert chunk.point is None
    assert body(chunk).startswith("Kollektiv müqavilələrdə")


def test_every_chunk_starts_with_article_header(chunks: dict[str, Chunk]) -> None:
    assert chunks["114.2"].text.startswith(
        "Maddə 114. Əsas məzuniyyət və onun müddətləri\n2. İşçilərə ödənişli"
    )
    for chunk in chunks.values():
        assert chunk.text.split("\n", 1)[0] == f"Maddə {chunk.article_no}. {chunk.title}"


def test_chunks_carry_article_metadata(chunks: dict[str, Chunk]) -> None:
    chunk = chunks["124.3"]
    assert chunk.title == "Təhsil məzuniyyətlərinin müddətləri"
    assert chunk.part == "V bölmə. İstirahət vaxtı və işçilərin məzuniyyət hüquqları"
    assert chunk.chapter == "On səkkizinci fəsil. Yaradıcılıq və təhsil məzuniyyətləri"
    assert chunk.url == URL
    assert chunk.source_downloaded_at == DOWNLOADED_AT


def test_chunk_ids_unique(articles: list[Article]) -> None:
    result = chunk_articles(articles, DOWNLOADED_AT)
    validate_chunks(result)
    assert len({c.chunk_id for c in result}) == len(result)


def test_no_text_lost_except_repealed_points(articles: list[Article]) -> None:
    for article in articles:
        bodies = [body(c) for c in chunk_article(article, DOWNLOADED_AT)]
        kept = [
            line
            for line in article.text.split("\n")
            if not re.fullmatch(r"\d+\. ləğv edilmişdir\.?", line)
        ]
        assert "\n".join(bodies).split("\n") == kept, article.article_no


# --- validate_chunks ------------------------------------------------------------------


def test_validate_chunks_rejects_empty_list() -> None:
    with pytest.raises(ParseError):
        validate_chunks([])


def test_validate_chunks_rejects_duplicate_ids() -> None:
    article = make_article("5", "1. Bir.\n1. Yenə bir.")
    with pytest.raises(ParseError, match=r"5\.1"):
        validate_chunks(chunk_article(article, DOWNLOADED_AT))


# --- run_ingest -----------------------------------------------------------------------


def write_meta(path: Path) -> None:
    meta = SourceMeta(
        source_url="https://frameworks.e-qanun.az/46/f_46943.html",
        ui_url=URL,
        downloaded_at=DOWNLOADED_AT,
        sha256="0" * 64,
        size_bytes=1,
        last_modified=None,
        etag=None,
    )
    path.write_text(meta.model_dump_json(), encoding="utf-8")


def test_run_ingest_writes_jsonl_and_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    meta_path = tmp_path / "f.meta.json"
    out_path = tmp_path / "processed" / "chunks.jsonl"
    write_meta(meta_path)

    with caplog.at_level(logging.INFO, logger="labor_code_rag.ingest.chunk"):
        result = run_ingest(FIXTURE, meta_path, out_path)

    raw = out_path.read_text(encoding="utf-8")
    assert "Maddə 114" in raw  # ensure_ascii=False: letters are not \u-escaped
    lines = raw.splitlines()
    assert len(lines) == len(result) == 9
    first = Chunk.model_validate(json.loads(lines[0]))
    assert first == result[0]
    assert first.source_downloaded_at == DOWNLOADED_AT

    [record] = [r for r in caplog.records if r.getMessage() == "ingest.completed"]
    assert record.__dict__["articles"] == 5
    assert record.__dict__["chunks"] == 9
    assert record.__dict__["repealed_points"] == 2
    assert isinstance(record.__dict__["latency_ms"], int)


def test_run_ingest_requires_meta(tmp_path: Path) -> None:
    with pytest.raises(SourceDownloadError, match="make data"):
        run_ingest(FIXTURE, tmp_path / "missing.meta.json", tmp_path / "chunks.jsonl")


# --- full source ----------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not FULL_SOURCE.exists(), reason="run `make data` first")
def test_chunk_full_source() -> None:
    articles = parse_articles(extract_paragraphs(decode_source(FULL_SOURCE.read_bytes())), URL)
    result = chunk_articles(articles, DOWNLOADED_AT)
    validate_chunks(result)

    assert len(result) == 988
    assert len({c.chunk_id for c in result}) == len(result)
    covered = {c.article_no for c in result}
    assert covered == {a.article_no for a in articles}
    assert {n for n in covered if "-" not in n} == {str(i) for i in range(1, 318)} - {
        "241",
        "298",
    }
    assert len([n for n in covered if "-" in n]) == 12
    # Spot checks from ADR-003: dashed points, prefixed points, a struck first point.
    ids = {c.chunk_id for c in result}
    assert {"3.2-1", "7.2-3", "10-1.7", "21-1.1", "179.2"} <= ids
    assert not {"7.2-3.1", "21-1.1.1", "124.1", "124.2", "220.5", "249.2", "258.2"} & ids
