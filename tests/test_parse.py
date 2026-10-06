import re
import unicodedata
from pathlib import Path

import pytest

from labor_code_rag.errors import ParseError
from labor_code_rag.ingest.parse import (
    decode_source,
    extract_paragraphs,
    parse_articles,
    strip_footnote_markers,
    validate_articles,
)
from labor_code_rag.models import Article

FIXTURE = Path(__file__).parent / "fixtures" / "labor_code_sample.html"
FULL_SOURCE = Path(__file__).parents[1] / "data" / "raw" / "f_46943.html"
URL = "https://e-qanun.az/framework/46943"


@pytest.fixture(scope="module")
def paragraphs() -> list[str]:
    return extract_paragraphs(decode_source(FIXTURE.read_bytes()))


@pytest.fixture(scope="module")
def articles(paragraphs: list[str]) -> dict[str, Article]:
    return {a.article_no: a for a in parse_articles(paragraphs, URL)}


def make_article(article_no: str) -> Article:
    return Article(
        article_no=article_no,
        title="Başlıq",
        part="I bölmə. Ümumi normalar",
        chapter="Birinci fəsil. Əsas müddəalar",
        text="1. Mətn.",
        footnote_ids=[],
        repealed_points=[],
        url=URL,
    )


# --- decode_source --------------------------------------------------------------------


def test_decode_source_reads_windows_1251() -> None:
    html = decode_source(FIXTURE.read_bytes())
    assert "charset=windows-1251" in html
    assert "<title>Az&#601;rbaycan Respublikas&#305; Prezidentinin" in html


def test_decode_source_rejects_bytes_outside_windows_1251() -> None:
    with pytest.raises(ParseError):
        decode_source(b"<p>\x98</p>")  # 0x98 is undefined in windows-1251


# --- extract_paragraphs ---------------------------------------------------------------


def test_extract_paragraphs_resolves_entities(paragraphs: list[str]) -> None:
    assert "Maddə 114. Əsas məzuniyyət və onun müddətləri" in paragraphs
    assert "&#" not in "".join(paragraphs)
    assert not re.search(r"&[a-z]+;", "".join(paragraphs))


def test_extract_paragraphs_output_is_nfc(paragraphs: list[str]) -> None:
    assert all(unicodedata.is_normalized("NFC", p) for p in paragraphs)


def test_extract_paragraphs_normalizes_decomposed_text_to_nfc() -> None:
    # "c" + combining cedilla (U+0327) must become a single "ç".
    assert extract_paragraphs("<html><body><p>c&#807;ox</p></body></html>") == ["çox"]


def test_extract_paragraphs_collapses_whitespace(paragraphs: list[str]) -> None:
    for p in paragraphs:
        assert p == p.strip()
        assert "  " not in p
        assert "\r" not in p and "\n" not in p and "\xa0" not in p


def test_extract_paragraphs_drops_empty_paragraphs(paragraphs: list[str]) -> None:
    assert all(paragraphs)


def test_extract_paragraphs_ignores_css_and_conditional_comments(
    paragraphs: list[str],
) -> None:
    joined = "\n".join(paragraphs)
    for leak in ("@font-face", "mso-", "Palatino", "[if gte mso", "endif", "<o:p>", "{"):
        assert leak not in joined


def test_extract_paragraphs_reads_headings_as_paragraphs() -> None:
    html = "<html><body><h3>Maddə 1. Başlıq</h3><p>1. Mətn.</p></body></html>"
    assert extract_paragraphs(html) == ["Maddə 1. Başlıq", "1. Mətn."]


def test_extract_paragraphs_drops_struck_through_text() -> None:
    html = "<html><body><p>bir <s>iki</s> üç</p><p><s>hamısı xətli</s></p></body></html>"
    assert extract_paragraphs(html) == ["bir üç"]


def test_extract_paragraphs_drops_struck_through_text_in_fixture(
    paragraphs: list[str],
) -> None:
    joined = "\n".join(paragraphs)
    # Maddə 298 (an <h3>) is struck through entirely; only its endnote marker survives.
    assert "Maddə 298" not in joined
    assert "dövlət rüsumu və məhkəmə xərcləri" not in joined
    # One struck word inside Maddə 305.
    assert "Kollektiv müqavilələrdə əmək kollektivinin" in joined
    assert "müəssisənin əmək kollektivinin" not in joined


def test_extract_paragraphs_keeps_downlevel_revealed_footnote_marker(
    paragraphs: list[str],
) -> None:
    # "[25]" sits inside <![if !supportFootnotes]> ... <![endif]> in the source.
    assert any(p.startswith("Maddə 7-1.") and p.endswith("[25]") for p in paragraphs)


# --- strip_footnote_markers -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2. ləğv edilmişdir.[351]", ("2. ləğv edilmişdir.", [351])),
        ("Səsvermə günü [321]", ("Səsvermə günü", [321])),
        ("a[1] b[22] c[1]", ("a b c", [1, 22])),
        ("Marker yoxdur.", ("Marker yoxdur.", [])),
        ("bu Məcəllənin 76-1.1-ci maddəsinə", ("bu Məcəllənin 76-1.1-ci maddəsinə", [])),
    ],
)
def test_strip_footnote_markers(text: str, expected: tuple[str, list[int]]) -> None:
    assert strip_footnote_markers(text) == expected


# --- parse_articles: which articles ---------------------------------------------------


def test_parse_articles_finds_exactly_the_fixture_articles(paragraphs: list[str]) -> None:
    result = parse_articles(paragraphs, URL)
    assert [a.article_no for a in result] == ["7-1", "114", "124", "305", "317"]


def test_parse_articles_ignores_old_wording_quoted_in_endnotes(
    articles: dict[str, Article],
) -> None:
    assert "49" not in articles
    last = articles["317"]
    for leak in ("ƏLAVƏLƏR", "Əvvəlki redaksiyada", "imzaladığı gündən qüvvəyə minir"):
        assert leak not in last.text
    assert last.text.endswith("yol verilir.")


def test_parse_articles_skips_repealed_article_leftovers(
    articles: dict[str, Article],
) -> None:
    # After struck text is dropped, Maddə 298 leaves a lone "[584]" right after Maddə 124.
    assert "298" not in articles
    assert 584 not in articles["124"].footnote_ids
    assert articles["124"].text.endswith("məzuniyyət verilir.")


def test_parse_articles_sets_url(articles: dict[str, Article]) -> None:
    assert all(a.url == URL for a in articles.values())


# --- parse_articles: headings and text ------------------------------------------------


def test_parse_articles_reads_dashed_number_and_multiline_title(
    articles: dict[str, Article],
) -> None:
    # In the source this title is wrapped over three lines inside one <p>.
    assert articles["7-1"].title == (
        "İşçilərin pay (səhm) iştirakı planı haqqında müqavilənin şərtləri"
    )


def test_parse_articles_titles(articles: dict[str, Article]) -> None:
    assert articles["114"].title == "Əsas məzuniyyət və onun müddətləri"
    assert articles["124"].title == "Təhsil məzuniyyətlərinin müddətləri"
    assert articles["305"].title == "Sosial sığortanın kollektiv müqavilələrlə tənzimlənməsi"
    assert articles["317"].title == "Bu Məcəllənin mətninin hüquqi qüvvəsi"


def test_parse_articles_text_without_struck_word(articles: dict[str, Article]) -> None:
    assert articles["305"].text == (
        "Kollektiv müqavilələrdə əmək kollektivinin üzvlərinin qanunvericilikdə nəzərdə "
        "tutulan sığortalanma şərtlərindən daha üstün sosial sığorta olunmasının əlavə "
        "formaları, qaydaları, sığorta məbləğləri, sığorta mənbələri müəyyən edilə bilər."
    )
    assert articles["305"].footnote_ids == [591]


def test_parse_articles_text_is_body_only(articles: dict[str, Article]) -> None:
    text = articles["114"].text
    assert not text.startswith("Maddə")
    assert text.startswith("1. Əsas məzuniyyət")
    assert "\n2. İşçilərə ödənişli əsas məzuniyyət 21 təqvim günündən" in text
    assert all(unicodedata.is_normalized("NFC", a.text) for a in articles.values())


def test_parse_articles_moves_footnote_markers_to_ids(articles: dict[str, Article]) -> None:
    assert articles["7-1"].footnote_ids == [25]
    assert "[" not in articles["7-1"].title
    assert {351, 352} <= set(articles["124"].footnote_ids)
    for article in articles.values():
        assert not re.search(r"\[\d+\]", article.title + article.text)


def test_parse_articles_records_repealed_points(articles: dict[str, Article]) -> None:
    assert articles["124"].repealed_points == ["1", "2"]
    assert articles["124"].text.startswith("1. ləğv edilmişdir.\n2. ləğv edilmişdir.\n3. ")
    assert articles["114"].repealed_points == []


def test_parse_articles_keeps_lettered_sub_points_inside_article(
    articles: dict[str, Article],
) -> None:
    lines = articles["114"].text.split("\n")
    for letter in ("a", "b", "c", "ç", "d", "e"):
        assert any(line.startswith(f"{letter}) ") for line in lines), letter
    assert lines[-1] == "e) həkimlərə, orta tibb işçilərinə və əczaçılara."


def test_parse_articles_keeps_dotted_sub_points_inside_article(
    articles: dict[str, Article],
) -> None:
    lines = articles["7-1"].text.split("\n")
    assert [line.split(" ", 1)[0] for line in lines] == ["7-1.1.", "7-1.1.1.", "7-1.1.2.", "7-1.2."]


@pytest.mark.parametrize(
    ("article_no", "part", "chapter"),
    [
        ("7-1", "I bölmə. Ümumi normalar", "Birinci fəsil. Əsas müddəalar"),
        (
            "114",
            "V bölmə. İstirahət vaxtı və işçilərin məzuniyyət hüquqları",
            "On yeddinci fəsil. Əmək məzuniyyətlərinin müddətləri",
        ),
        # New chapter, same part.
        (
            "124",
            "V bölmə. İstirahət vaxtı və işçilərin məzuniyyət hüquqları",
            "On səkkizinci fəsil. Yaradıcılıq və təhsil məzuniyyətləri",
        ),
        # Source titles: "İşçilərin <s>sosial</s> sığortası[589]" (struck word + marker).
        (
            "305",
            "XII bölmə. İşçilərin sığortası",
            "Qırx altıncı fəsil. İşçilərin sığorta olunmasının tənzimlənməsi",
        ),
        (
            "317",
            "XIII bölmə. Yekun normalar",
            "Qırx səkkizinci fəsil. Azərbaycan Respublikasının Əmək Məcəlləsinin tətbiqi "
            "ilə bağlı hüquqi tənzimləmə məsələləri",
        ),
    ],
)
def test_parse_articles_assigns_part_and_chapter(
    articles: dict[str, Article], article_no: str, part: str, chapter: str
) -> None:
    assert articles[article_no].part == part
    assert articles[article_no].chapter == chapter


# --- validate_articles ----------------------------------------------------------------


def test_validate_articles_accepts_fixture_articles(articles: dict[str, Article]) -> None:
    validate_articles(list(articles.values()))


def test_validate_articles_rejects_empty_list() -> None:
    with pytest.raises(ParseError):
        validate_articles([])


def test_validate_articles_rejects_duplicate_numbers() -> None:
    with pytest.raises(ParseError, match="114"):
        validate_articles([make_article("114"), make_article("115"), make_article("114")])


# --- full source ----------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not FULL_SOURCE.exists(), reason="run `make data` first")
def test_parse_full_source() -> None:
    result = parse_articles(extract_paragraphs(decode_source(FULL_SOURCE.read_bytes())), URL)
    validate_articles(result)
    numbers = [a.article_no for a in result]

    # 1-317 plus 12 dashed, minus 241 and 298: repealed, struck through entirely (ADR-001).
    assert len(numbers) == 327
    assert len(set(numbers)) == len(numbers)
    assert numbers[-1] == "317"
    assert {n for n in numbers if "-" not in n} == {str(i) for i in range(1, 318)} - {
        "241",
        "298",
    }
    assert len([n for n in numbers if "-" in n]) == 12

    repealed = sorted((a.article_no, p) for a in result for p in a.repealed_points)
    assert repealed == [("124", "1"), ("124", "2"), ("220", "5"), ("249", "2"), ("258", "2")]

    assert len({a.part for a in result}) == 13
    assert len({a.chapter for a in result}) == 48
    for article in result:
        assert article.title and article.text, article.article_no
        assert not re.search(r"\[\d+\]", article.title + article.text), article.article_no
