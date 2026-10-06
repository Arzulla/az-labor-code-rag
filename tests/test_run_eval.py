"""Harness end to end on a 3-item golden fixture with a fake LLM (no network)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import openai
import pytest
from chromadb.api.models.Collection import Collection

from eval.run_eval import (
    build_result,
    evaluate,
    load_golden,
    result_label,
    summary_table,
    write_result,
)
from labor_code_rag.config import Settings
from labor_code_rag.embedding_cache import EmbeddingCache
from labor_code_rag.ingest.index import index_chunks
from labor_code_rag.llm import LLMClient
from labor_code_rag.models import Chunk, EvalResult, GoldenItem

from conftest import FakeOpenAI, api_error, make_settings

GOLDEN = [
    GoldenItem(
        id="fact-x",
        question="Əmək məzuniyyəti neçə gündür?",
        expected_answer="21 təqvim günü (Maddə 114.1).",
        relevant_chunks=["114.1"],
        category="factual",
    ),
    GoldenItem(
        id="coll-x",
        question="Hamilə qadını işdən çıxarmaq olar?",
        expected_answer="Xeyr (Maddə 79).",
        relevant_chunks=["79"],
        category="colloquial",
    ),
    GoldenItem(
        id="oos-x",
        question="Gəlir vergisi neçə faizdir?",
        expected_answer="Cavab yoxdur.",
        relevant_chunks=[],
        category="out_of_scope",
    ),
]

ANSWER = json.dumps(
    {
        "text": "21 gün (Maddə 114.1).",
        "citations": [{"article_no": "114", "point": "1"}],
        "refused": False,
    },
    ensure_ascii=False,
)
REFUSAL = json.dumps({"text": "Bilmirəm.", "citations": [], "refused": True})
JUDGE = json.dumps({"accuracy": 5, "completeness": 4, "relevance": 5, "rationale": "ok"})

# Fake usage per chat call: 1200 in / 300 out tokens.
ANSWER_COST = 1200 * 0.40e-6 + 300 * 1.60e-6  # 0.00096
JUDGE_COST = 1200 * 2.00e-6 + 300 * 8.00e-6  # 0.0048


@pytest.fixture
def indexed(llm: LLMClient, collection: Collection, chunks: list[Chunk]) -> Collection:
    index_chunks(chunks, llm, EmbeddingCache(":memory:"), collection)
    return collection


@pytest.fixture
def golden_path(tmp_path: Path) -> Path:
    path = tmp_path / "dev.jsonl"
    path.write_text("".join(g.model_dump_json() + "\n" for g in GOLDEN), encoding="utf-8")
    return path


def _result(run_items: list[GoldenItem], path: Path, run: object, **kw: object) -> EvalResult:
    return build_result(
        run,  # type: ignore[arg-type]
        run_items,
        label="fixture",
        split="dev",
        settings=Settings(llm=make_settings()),
        golden_path=path,
        retrieval_only=bool(kw.get("retrieval_only", False)),
        created_at=datetime(2026, 10, 6, 12, 0, tzinfo=UTC),
        wall_time_s=1.234,
    )


def test_end_to_end_writes_complete_result(
    llm: LLMClient,
    fake_openai: FakeOpenAI,
    indexed: Collection,
    golden_path: Path,
    tmp_path: Path,
) -> None:
    golden = load_golden(golden_path)
    fake_openai.chat_replies = [ANSWER, JUDGE, REFUSAL, REFUSAL]

    run = evaluate(golden, llm, indexed, k=3, max_cost_usd=1.0)

    assert len(fake_openai.chat_calls) == 4  # 3 answers + 1 judge (refusals not judged)
    fact, coll, oos = run.items
    assert fact.judge is not None and fact.judge.accuracy == 5
    assert fact.cost_usd == pytest.approx(ANSWER_COST + JUDGE_COST, rel=1e-3)
    assert coll.judge_skipped == "refused" and coll.judge is not None
    assert coll.judge.accuracy == 1
    assert oos.judge_skipped == "out_of_scope" and oos.judge is None
    assert fact.retrieved_article_nos[0] == "114"

    result = _result(golden, golden_path, run)
    path = write_result(result, tmp_path / "results")

    assert path.name == "20261006T120000Z_fixture_unverified_dev.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "git_sha",
        "git_dirty",
        "config",
        "prompt_version",
        "judge_prompt_version",
        "embedding_model",
        "answer_model",
        "judge_model",
        "golden_sha256",
        "golden_verified",
        "metrics",
        "metrics_by_category",
        "items",
        "total_cost_usd",
        "wall_time_s",
        "budget_exceeded",
    ):
        assert key in saved, key
    assert saved["golden_verified"] == {"verified_n": 0, "total_n": 3}
    assert "openai_api_key" not in saved["config"]
    assert saved["metrics"]["false_refusal_rate"] == 0.5
    assert saved["metrics"]["oos_refusal_rate"] == 1.0
    assert set(saved["metrics_by_category"]) == {"factual", "colloquial", "out_of_scope"}
    assert saved["items"][0]["cited"] == [{"article": "114", "point": "1"}]
    assert EvalResult.model_validate(saved) == result
    assert "| mrr |" in summary_table(result)


def test_label_suffix_only_when_unverified() -> None:
    assert result_label("baseline", 3, 3) == "baseline"
    assert result_label("baseline", 2, 3) == "baseline_unverified"


def test_budget_guard_stops_after_judge_call(
    llm: LLMClient, fake_openai: FakeOpenAI, indexed: Collection
) -> None:
    fake_openai.chat_replies = [ANSWER, JUDGE, REFUSAL, REFUSAL]
    run = evaluate(GOLDEN, llm, indexed, k=3, max_cost_usd=0.001)  # answer fits, judge not
    assert run.budget_exceeded
    assert run.stopped_reason == "budget_exceeded"
    assert [i.id for i in run.items] == ["fact-x"]
    assert len(fake_openai.chat_calls) == 2


def test_budget_guard_stops_before_judge(
    llm: LLMClient, fake_openai: FakeOpenAI, indexed: Collection
) -> None:
    fake_openai.chat_replies = [ANSWER, JUDGE]
    run = evaluate(GOLDEN, llm, indexed, k=3, max_cost_usd=0.0005)
    assert run.budget_exceeded
    assert len(fake_openai.chat_calls) == 1  # the judge was never paid for
    assert run.items[0].judge is None


def test_retrieval_only_makes_no_chat_calls(
    llm: LLMClient, fake_openai: FakeOpenAI, indexed: Collection, golden_path: Path
) -> None:
    run = evaluate(GOLDEN, llm, indexed, k=3, max_cost_usd=1.0, retrieval_only=True)
    assert fake_openai.chat_calls == []
    assert len(run.items) == 3 and all(i.answer is None for i in run.items)
    result = _result(GOLDEN, golden_path, run, retrieval_only=True)
    assert result.retrieval_only
    assert result.metrics["mrr"] is not None
    assert "judge_accuracy" not in result.metrics


def test_pipeline_error_stops_the_run(
    llm: LLMClient, fake_openai: FakeOpenAI, indexed: Collection
) -> None:
    fake_openai.chat_errors = [api_error(openai.BadRequestError, 400)]
    run = evaluate(GOLDEN, llm, indexed, k=3, max_cost_usd=1.0)
    assert run.items == []
    assert run.stopped_reason is not None and run.stopped_reason.startswith("error on fact-x")
    assert not run.budget_exceeded
