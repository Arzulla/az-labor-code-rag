import pytest

from eval.calibrate import CalibrationItem, calibration_report
from eval.judge import REFUSED_SCORE
from eval.report import (
    README_END,
    README_START,
    build_calibration,
    mean_metric,
    replace_block,
    select_calibration,
    spread,
)
from labor_code_rag.models import EvalItemResult, EvalResult, GoldenItem, JudgeScore


def _cal(id_: str, judge: int, human: int | None) -> CalibrationItem:
    return CalibrationItem.model_validate(
        {
            "id": id_,
            "category": "factual",
            "question": "q",
            "expected_answer": "e",
            "answer": "a",
            "judge_accuracy": judge,
            "judge_completeness": 5,
            "judge_relevance": 5,
            "judge_rationale": "r",
            "human_accuracy": human,
            "source_result": "x.json",
        }
    )


def test_report_pending_while_human_scores_empty() -> None:
    report = calibration_report([_cal("a", 5, None), _cal("b", 3, None)])
    assert report.startswith("calibration: pending (0/2")


def test_report_agreement_and_spearman() -> None:
    # judge [5, 4, 2], human [5, 3, 2]: exact 2/3, within ±1 3/3, ranks identical -> rho 1.
    items = [_cal("a", 5, 5), _cal("b", 4, 3), _cal("c", 2, 2), _cal("d", 3, None)]
    report = calibration_report(items)
    assert "3/4 human scores" in report
    assert "exact agreement: 0.67" in report
    assert "within ±1:       1.00" in report
    assert "Spearman rho:    1.00" in report
    assert "partial" in report


def _item(id_: str, category: str, *, refused: bool = False, answer: str = "a") -> EvalItemResult:
    judged = category != "out_of_scope" and not refused
    return EvalItemResult.model_validate(
        {
            "id": id_,
            "category": category,
            "question": "q",
            "relevant_chunks": [] if category == "out_of_scope" else ["1"],
            "request_id": "r",
            "retrieved_chunk_ids": [],
            "retrieved_article_nos": [],
            "answer": answer,
            "refused": refused,
            "judge": JudgeScore(accuracy=4, completeness=4, relevance=4, rationale="r")
            if judged
            else (REFUSED_SCORE if refused else None),
            "judge_skipped": None if judged else ("refused" if refused else "out_of_scope"),
            "latency_ms": 1,
            "cost_usd": 0.0,
        }
    )


def test_select_calibration_round_robin_over_judged_items() -> None:
    items = [
        *[_item(f"f{n}", "factual") for n in range(5)],
        _item("c1", "colloquial"),
        _item("c2", "colloquial", refused=True),
        _item("o1", "out_of_scope"),
    ]
    chosen = select_calibration(items, n=3, seed=1)
    ids = [i.id for i in chosen]
    assert len(ids) == 3
    assert "c1" in ids  # categories alternate, so the only judged colloquial item is picked
    assert "c2" not in ids and "o1" not in ids  # refused / out_of_scope have no judge score
    assert select_calibration(items, n=3, seed=1) == chosen  # seeded


def test_build_calibration_keeps_human_scores_for_unchanged_answers() -> None:
    golden = {
        i: GoldenItem(
            id=i, question="q", expected_answer="e", relevant_chunks=["1"], category="factual"
        )
        for i in ("f0", "f1")
    }
    previous = [_cal("f0", 4, 2), _cal("f1", 4, 5)]
    previous[1] = previous[1].model_copy(update={"answer": "old answer"})
    items = [_item("f0", "factual"), _item("f1", "factual")]

    rows, dropped = build_calibration(items, golden, "run.json", previous, n=2)

    by_id = {r.id: r for r in rows}
    assert by_id["f0"].human_accuracy == 2  # same id and answer: kept
    assert by_id["f1"].human_accuracy is None  # answer changed: must be re-scored
    assert dropped == 1


def test_readme_block_replacement() -> None:
    text = f"intro\n{README_START}\nold\n{README_END}\noutro\n"
    assert replace_block(text, f"{README_START}\nnew\n{README_END}") == (
        f"intro\n{README_START}\nnew\n{README_END}\noutro\n"
    )
    with pytest.raises(ValueError):
        replace_block("no markers", "x")


def test_mean_and_spread_over_runs() -> None:
    runs = [
        EvalResult.model_construct(metrics={"mrr": 0.4, "x": None}),
        EvalResult.model_construct(metrics={"mrr": 0.5, "x": 1.0}),
    ]
    assert mean_metric(runs, "mrr") == pytest.approx(0.45)
    assert spread(runs, "mrr") == pytest.approx(0.1)
    assert mean_metric(runs, "x") is None  # undefined in one run
