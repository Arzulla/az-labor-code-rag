"""Azerbaijani text helpers: NFC normalization, case folding, tokenization, article refs.

Use these everywhere instead of ``str.lower()`` (see CLAUDE.md §6): Python's default
mapping is Turkish/Azerbaijani-unaware (``"I".lower() == "i"``, but it must be ``"ı"``).
"""

import re
import unicodedata

from pydantic import BaseModel, ConfigDict

# Applied before str.lower(): "İ".lower() would give "i" + U+0307 (combining dot above).
_AZ_UPPER_I = str.maketrans({"I": "ı", "İ": "i"})

# Numbers with dots/hyphens stay one token ("7-1", "114.2"); words are letter runs.
_TOKEN_RE = re.compile(r"\d+(?:[.-]\d+)*|[^\W\d_]+")

# Building blocks for article references; matched against az_lower(text).
_NUM = r"\d+(?:-\d+)?"  # article number: "114", "7-1"
_POINT = r"(?:\.\d+(?:-\d+)?)+"  # dotted point: ".2", ".1.1"
_SUFFIX = r"-(?:ci|cı|cu|cü|inci|ıncı|uncu|üncü)"  # ordinal suffix: "-cü", "-ci"

_REF_RE = re.compile(
    rf"""
    # "Maddə 114", "maddə 114.2", "m. 7-1"
    \b(?:maddə|m\.)\s*(?P<a_num>{_NUM})(?P<a_point>{_POINT})?
    |
    # optional list head: "118, 119 və " in "118, 119 və 121-ci maddələrində"
    (?P<list>\b(?:{_NUM}(?:{_POINT})?\s*,\s*)*{_NUM}(?:{_POINT})?\s+və\s+)?
    # "114-cü maddə", "76-1.1-ci maddəsinə", "70-ci maddənin 1-ci hissəsi"
    \b(?P<s_num>{_NUM})(?P<s_point>{_POINT})?{_SUFFIX}\s+maddə\w*
    (?:\s+(?P<part>{_NUM}){_SUFFIX}\s+hissə\w*)?
    """,
    re.VERBOSE,
)
_LIST_ITEM_RE = re.compile(rf"(?P<num>{_NUM})(?P<point>{_POINT})?")


class ArticleRef(BaseModel):
    """A reference to an article and, optionally, a point inside it."""

    model_config = ConfigDict(frozen=True)

    article: str  # str, not int: article numbers like "7-1" exist
    point: str | None = None  # "2", "1.1" (Maddə 7-1.1.1 -> article "7-1", point "1.1")

    def __str__(self) -> str:
        return f"Maddə {self.article}.{self.point}" if self.point else f"Maddə {self.article}"


def normalize(text: str) -> str:
    """Unicode NFC; apply at ingest and query time."""
    return unicodedata.normalize("NFC", text)


def az_lower(text: str) -> str:
    """Lowercase with Azerbaijani rules: ``I`` -> ``ı``, ``İ`` -> ``i``."""
    return normalize(text).translate(_AZ_UPPER_I).lower()


def tokenize(text: str) -> list[str]:
    """Lowercased tokens for BM25; punctuation dropped, no stemming or stopwords."""
    return _TOKEN_RE.findall(az_lower(text))


def _point(dotted: str | None) -> str | None:
    return dotted[1:] if dotted else None  # ".1.1" -> "1.1"


def parse_article_refs(text: str) -> list[ArticleRef]:
    """Find article references in order of appearance, without duplicates.

    Handles ``Maddə 114``, ``m. 114.2``, ``114-cü maddə``, ``76-1.1-ci maddəsinə``,
    ``70-ci maddənin 1-ci hissəsi`` (-> 70.1) and lists like ``118, 119 və 121-ci
    maddələrində``. Relative references ("bu maddənin 2-ci hissəsi") are ignored.
    """
    refs: dict[ArticleRef, None] = {}  # dict keeps insertion order and dedups
    for match in _REF_RE.finditer(az_lower(text)):
        if match["a_num"]:
            refs[ArticleRef(article=match["a_num"], point=_point(match["a_point"]))] = None
            continue
        for item in _LIST_ITEM_RE.finditer(match["list"] or ""):
            refs[ArticleRef(article=item["num"], point=_point(item["point"]))] = None
        # A dotted point wins over "N-ci hissəsi"; both at once does not occur in the law.
        point = _point(match["s_point"]) or match["part"]
        refs[ArticleRef(article=match["s_num"], point=point)] = None
    return list(refs)
