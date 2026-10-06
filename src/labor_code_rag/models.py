"""Pydantic models shared across module boundaries (CLAUDE.md §6)."""

from pydantic import BaseModel, ConfigDict


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
