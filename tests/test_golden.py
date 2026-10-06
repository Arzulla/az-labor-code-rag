"""The golden set itself: schema, split hygiene, and gold chunk ids that exist (ADR-006)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval.run_eval import load_golden
from eval.split import assign_split
from labor_code_rag.models import CATEGORIES, GoldenItem

GOLDEN_DIR = Path("eval/golden")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")


@pytest.fixture(scope="module")
def splits() -> dict[str, list[GoldenItem]]:
    return {name: load_golden(GOLDEN_DIR / f"{name}.jsonl") for name in ("dev", "test")}


def test_size_and_categories(splits: dict[str, list[GoldenItem]]) -> None:
    items = splits["dev"] + splits["test"]
    assert 60 <= len(items) <= 80
    for name, part in splits.items():
        assert {i.category for i in part} == set(CATEGORIES), name


def test_ids_unique_and_questions_disjoint(splits: dict[str, list[GoldenItem]]) -> None:
    items = splits["dev"] + splits["test"]
    assert len({i.id for i in items}) == len(items)
    assert not {i.question for i in splits["dev"]} & {i.question for i in splits["test"]}


def test_out_of_scope_has_no_relevant_chunks(splits: dict[str, list[GoldenItem]]) -> None:
    for item in splits["dev"] + splits["test"]:
        assert (item.category == "out_of_scope") == (item.relevant_chunks == []), item.id


def test_split_matches_seeded_assignment(splits: dict[str, list[GoldenItem]]) -> None:
    expected = assign_split(splits["dev"] + splits["test"])
    for name, part in splits.items():
        assert {expected[i.id] for i in part} == {name}


@pytest.mark.skipif(not CHUNKS_FILE.exists(), reason="data/processed missing: run make ingest")
def test_every_relevant_chunk_exists(splits: dict[str, list[GoldenItem]]) -> None:
    lines = CHUNKS_FILE.read_text(encoding="utf-8").splitlines()
    chunk_ids = {json.loads(line)["chunk_id"] for line in lines}
    for item in splits["dev"] + splits["test"]:
        assert set(item.relevant_chunks) <= chunk_ids, item.id


def test_golden_item_validation() -> None:
    with pytest.raises(ValidationError, match="out_of_scope"):
        GoldenItem(
            id="x",
            question="q",
            expected_answer="a",
            relevant_chunks=["1"],
            category="out_of_scope",
        )
    with pytest.raises(ValidationError, match="out_of_scope"):
        GoldenItem(
            id="x", question="q", expected_answer="a", relevant_chunks=[], category="factual"
        )
    item = GoldenItem(
        id="x",
        question="q",
        expected_answer="a",
        category="multi_article",
        relevant_chunks=["114.2", "114.3", "7-1.1", "70"],
    )
    assert item.relevant_articles == ["114", "7-1", "70"]
    assert item.verified is False


def test_assign_split_is_deterministic_and_stratified() -> None:
    items = [
        GoldenItem(
            id=f"f{n}",
            question=f"q{n}",
            expected_answer="a",
            relevant_chunks=["1"],
            category="factual",
        )
        for n in range(10)
    ]
    first, again = assign_split(items), assign_split(list(reversed(items)))
    assert first == again  # file order does not matter
    assert sum(v == "dev" for v in first.values()) == 6  # round(0.55 * 10) = 6 (half to even)
