import html
import re
from pathlib import Path

import pytest

from labor_code_rag.text import ArticleRef, az_lower, normalize, parse_article_refs, tokenize

FIXTURE = Path(__file__).parent / "fixtures" / "labor_code_sample.html"


def ref(article: str, point: str | None = None) -> ArticleRef:
    return ArticleRef(article=article, point=point)


# --- normalize / az_lower -------------------------------------------------------------


def test_normalize_composes_decomposed_characters() -> None:
    decomposed = "ç"  # "c" + combining cedilla
    assert normalize(decomposed) == "ç"
    assert len(normalize(decomposed)) == 1


def test_normalize_keeps_schwa() -> None:
    assert normalize("Əmək") == "Əmək"


@pytest.mark.parametrize(
    ("upper", "lower"),
    [
        ("İŞÇİ", "işçi"),
        ("IŞIQ", "ışıq"),
        ("MADDƏ", "maddə"),
        ("İstirahət Işığı", "istirahət ışığı"),
        ("ĞÖÜÇŞ", "ğöüçş"),
    ],
)
def test_az_lower(upper: str, lower: str) -> None:
    assert az_lower(upper) == lower


def test_plain_lower_is_wrong_for_azerbaijani() -> None:
    # Documents why az_lower exists.
    assert "IŞIQ".lower() == "işiq"
    assert "İ".lower() == "i̇"


def test_az_lower_handles_decomposed_capital_i_with_dot() -> None:
    assert az_lower("İŞÇİ") == "işçi"  # NFC first turns "I" + U+0307 into "İ"


# --- tokenize -----------------------------------------------------------------------


def test_tokenize_keeps_article_numbers_whole() -> None:
    assert tokenize("Maddə 7-1.1.1. İşçilərin pay (səhm) iştirakı;") == [
        "maddə",
        "7-1.1.1",
        "işçilərin",
        "pay",
        "səhm",
        "iştirakı",
    ]


def test_tokenize_splits_ordinal_suffix() -> None:
    assert tokenize("114-cü maddə") == ["114", "cü", "maddə"]


# --- parse_article_refs -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Maddə 114", [ref("114")]),
        ("maddə 114.2 deyir ki", [ref("114", "2")]),
        ("MADDƏ 7-1", [ref("7-1")]),
        ("m. 114", [ref("114")]),
        ("M. 114.3", [ref("114", "3")]),
        ("Maddə 7-1.1.1", [ref("7-1", "1.1")]),
        ("114-cü maddə nə deyir?", [ref("114")]),
        ("Bu Məcəllənin 7-ci maddəsinin", [ref("7")]),
        ("7-1-ci maddə", [ref("7-1")]),
        ("bu Məcəllənin 76-1.1-ci maddəsinə uyğun", [ref("76-1", "1")]),
        ("125-inci maddədə", [ref("125")]),
        ("70-ci maddənin 1-ci hissəsi", [ref("70", "1")]),
        ("114-cü maddənin 2-ci hissəsində", [ref("114", "2")]),
        ("7-ci maddəsinin 2-1-ci hissəsi", [ref("7", "2-1")]),
        ("İŞƏGÖTÜRƏN 114-CÜ MADDƏNİ", [ref("114")]),
        (
            "bu Məcəllənin 118, 119, 120 və 121-ci maddələrində",
            [ref("118"), ref("119"), ref("120"), ref("121")],
        ),
        ("12, 16 və 35-ci maddələrinin", [ref("12"), ref("16"), ref("35")]),
    ],
)
def test_parse_article_refs(text: str, expected: list[ArticleRef]) -> None:
    assert parse_article_refs(text) == expected


def test_heading_trailing_dot_is_not_a_point() -> None:
    assert parse_article_refs("Maddə 7-1. İşçilərin pay (səhm) iştirakı") == [ref("7-1")]


@pytest.mark.parametrize(
    "text",
    [
        "bu maddənin 2-ci hissəsi",  # relative: no article number
        "27 dekabr 2013-cü il tarixli",
        "№ 24",
        "1. Əsas məzuniyyət 21 təqvim günü",
        "maddələr haqqında",
    ],
)
def test_no_refs(text: str) -> None:
    assert parse_article_refs(text) == []


def test_refs_are_ordered_and_deduplicated() -> None:
    text = "Maddə 115 və 114-cü maddə; həmçinin Maddə 115, m. 114.2"
    assert parse_article_refs(text) == [ref("115"), ref("114"), ref("114", "2")]


def test_article_ref_str() -> None:
    assert str(ref("114", "2")) == "Maddə 114.2"
    assert str(ref("7-1")) == "Maddə 7-1"


def test_refs_in_real_fixture() -> None:
    # Crude tag stripping, only to get text for this test; real parsing lives in parse.py.
    raw = FIXTURE.read_bytes().decode("windows-1251")
    no_markup = re.sub(r"<style>.*?</style>|<!--.*?-->|<[^>]+>", " ", raw, flags=re.S)
    refs = parse_article_refs(html.unescape(no_markup))

    for article in ("7-1", "114", "124", "317", "49"):
        assert ref(article) in refs
    assert ref("76-1", "1") in refs  # "76-1.1-ci maddəsinə" inside Maddə 7-1.2
