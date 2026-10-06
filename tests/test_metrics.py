import math

import pytest

from eval.judge import REFUSED_SCORE
from eval.metrics import (
    article_only_share,
    average_ranks,
    chunk_recall,
    citation_correct_point,
    citation_precision,
    dedup_ranked,
    exact_agreement,
    invalid_kind,
    mean,
    percentile,
    precision_at_k,
    rate,
    recall_at_k,
    reciprocal_rank,
    spearman,
    summarize,
    within_agreement,
)
from labor_code_rag.models import EvalItemResult, JudgeScore
from labor_code_rag.text import ArticleRef

# 114 appears twice: after dedup the ranking is [114, 70, 79].
RANKED = ["114", "114", "70", "79"]


def ref(article: str, point: str | None = None) -> ArticleRef:
    return ArticleRef(article=article, point=point)


def test_dedup_keeps_first_occurrence() -> None:
    assert dedup_ranked(["114", "114", "70", "114", "79"]) == ["114", "70", "79"]


def test_reciprocal_rank_counts_duplicates_once() -> None:
    assert reciprocal_rank(RANKED, {"70"}) == 0.5  # rank 2 after dedup, not 3
    assert reciprocal_rank(RANKED, {"114"}) == 1.0
    assert reciprocal_rank(RANKED, {"5"}) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k(RANKED, {"79", "5"}, 1) == 0.0
    assert recall_at_k(RANKED, {"79", "5"}, 3) == 0.5  # 79 found, 5 missing
    assert recall_at_k(RANKED, {"114", "70"}, 2) == 1.0
    assert recall_at_k(RANKED, {"79", "5"}, 8) == 0.5  # k larger than the list


def test_precision_at_k_uses_k_as_denominator() -> None:
    assert precision_at_k(RANKED, {"114"}, 1) == 1.0
    assert precision_at_k(RANKED, {"79", "5"}, 3) == pytest.approx(1 / 3)
    # Only 3 unique articles but k=8: missing slots count as non-relevant (trec_eval).
    assert precision_at_k(RANKED, {"79", "5"}, 8) == 0.125


def test_chunk_recall() -> None:
    assert chunk_recall(["114.2", "114.3", "70"], ["114.2", "79.1"]) == 0.5
    assert chunk_recall(["114.2", "70"], ["70"]) == 1.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: reciprocal_rank(RANKED, []),
        lambda: recall_at_k(RANKED, [], 3),
        lambda: precision_at_k(RANKED, [], 3),
        lambda: chunk_recall(["114.2"], []),
    ],
)
def test_empty_relevant_set_raises(call: object) -> None:
    with pytest.raises(ValueError, match="out_of_scope"):
        call()  # type: ignore[operator]


def test_citation_point_rules() -> None:
    gold = ["114.2", "70"]
    assert citation_correct_point(ref("114", "2"), gold)
    assert citation_correct_point(ref("114", "2.1"), gold)  # Maddə 114.2.1 is in chunk 114.2
    assert citation_correct_point(ref("70", "1"), gold)  # 70 is gold as a whole chunk
    assert citation_correct_point(ref("70"), gold)
    assert not citation_correct_point(ref("114"), gold)  # article-only cannot hit 114.2
    assert not citation_correct_point(ref("114", "3"), gold)


def test_citation_precision_pooled_and_article_only_share() -> None:
    refs = [ref("114", "2"), ref("114"), ref("70", "1"), ref("79")]
    pairs = [(refs[:2], ["114.2", "70"]), (refs[2:], ["114.2", "70"])]
    assert citation_precision(pairs, "article") == 0.75  # 114, 114, 70 hit; 79 misses
    assert citation_precision(pairs, "point") == 0.5  # 114.2 and 70.1 hit
    assert article_only_share(refs) == 0.5
    assert citation_precision([([], ["114.2"])], "point") is None
    assert article_only_share([]) is None


def test_rate_mean_percentile() -> None:
    assert rate(1, 4) == 0.25
    assert rate(0, 0) is None
    assert mean([1.0, 2.0, 6.0]) == 3.0
    assert mean([]) is None
    assert percentile([40, 10, 30, 20], 50) == 20  # rank ceil(2.0) = 2
    assert percentile([40, 10, 30, 20], 95) == 40  # rank ceil(3.8) = 4
    assert percentile([7], 95) == 7
    assert percentile([], 50) is None


def test_agreement() -> None:
    assert exact_agreement([3, 4, 5], [3, 5, 5]) == pytest.approx(2 / 3)
    assert within_agreement([3, 4, 5], [3, 5, 5]) == 1.0
    assert within_agreement([1, 4], [3, 4]) == 0.5
    assert exact_agreement([], []) is None


def test_spearman() -> None:
    assert average_ranks([5, 3, 5]) == [2.5, 1.0, 2.5]
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    # Ties: ranks a=[1, 2.5, 2.5, 4], b=[1, 3, 2, 4]; cov=4.5, var_a=4.5, var_b=5.
    assert spearman([1, 2, 2, 3], [1, 3, 2, 4]) == pytest.approx(4.5 / math.sqrt(4.5 * 5))
    assert spearman([3, 3, 3], [1, 2, 3]) is None  # constant side: undefined
    assert spearman([1], [1]) is None


def _item(**fields: object) -> EvalItemResult:
    base: dict[str, object] = {"question": "q", "request_id": "r", "latency_ms": 0, "cost_usd": 0}
    return EvalItemResult.model_validate({**base, **fields})


def test_summarize_hand_computed() -> None:
    items = [
        _item(
            id="a",
            category="factual",
            relevant_chunks=["114.2"],
            retrieved_chunk_ids=["114.2", "70"],
            retrieved_article_nos=["114", "70"],
            answer="21 gün (Maddə 114.2).",
            refused=False,
            cited=[ref("114", "2")],
            judge=JudgeScore(accuracy=5, completeness=4, relevance=5, rationale="ok"),
            latency_ms=100,
            cost_usd=0.010,
        ),
        _item(
            id="b",
            category="colloquial",
            relevant_chunks=["79.1"],
            retrieved_chunk_ids=["70", "79.1"],
            retrieved_article_nos=["70", "79"],
            answer="refusal",
            refused=True,
            judge=REFUSED_SCORE,
            judge_skipped="refused",
            latency_ms=300,
            cost_usd=0.002,
        ),
        _item(
            id="c",
            category="out_of_scope",
            relevant_chunks=[],
            retrieved_chunk_ids=["5"],
            retrieved_article_nos=["5"],
            answer="refusal",
            refused=True,
            judge_skipped="out_of_scope",
            latency_ms=200,
            cost_usd=0.001,
        ),
    ]
    m = summarize(items, k_chunks=8)
    assert m["retrieval_n"] == 2  # out_of_scope skipped
    assert m["mrr"] == 0.75  # (1 + 1/2) / 2
    assert m["recall@1"] == 0.5
    assert m["recall@3"] == 1.0
    assert m["precision@1"] == 0.5
    assert m["chunk_recall@8"] == 1.0
    assert m["oos_refusal_rate"] == 1.0
    assert m["false_refusal_rate"] == 0.5
    assert m["cited_n"] == 1
    assert m["invalid_citation_rate"] == 0.0
    assert m["citation_precision_point"] == 1.0
    assert m["article_only_citation_share"] == 0.0
    assert m["judge_n"] == 1
    assert m["judge_skipped_refused_n"] == 1
    assert m["judge_accuracy"] == 3.0  # (5 + refused 1) / 2
    assert m["judge_completeness"] == 2.5
    assert m["judge_accuracy_answered"] == 5.0
    assert m["latency_p50_ms"] == 200
    assert m["latency_p95_ms"] == 300
    assert m["cost_total_usd"] == pytest.approx(0.013)
    assert m["cost_mean_usd"] == pytest.approx(0.013 / 3)


def test_summarize_retrieval_only_has_no_answer_metrics() -> None:
    item = _item(
        id="a",
        category="factual",
        relevant_chunks=["114.2"],
        retrieved_chunk_ids=["114.2"],
        retrieved_article_nos=["114"],
    )
    m = summarize([item], k_chunks=8)
    assert m["mrr"] == 1.0
    assert "false_refusal_rate" not in m
    assert "judge_accuracy" not in m


def test_invalid_citations_split_into_format_and_not_retrieved() -> None:
    # The answer model put the chunk id into article_no: article "254.1", point "1".
    format_error = ref("254.1", "1")
    assert invalid_kind(format_error) == "format"
    assert invalid_kind(ref("250")) == "not_retrieved"
    assert invalid_kind(ref("254", "1")) == "not_retrieved"

    def answered(id_: str, cited: list[ArticleRef], invalid: list[ArticleRef]) -> EvalItemResult:
        return _item(
            id=id_,
            category="factual",
            relevant_chunks=["114.1"],
            retrieved_chunk_ids=["114.1"],
            retrieved_article_nos=["114"],
            answer="x",
            refused=False,
            cited=cited,
            invalid_citations=invalid,
            judge=JudgeScore(accuracy=5, completeness=5, relevance=5, rationale="ok"),
        )

    items = [
        answered("a", [ref("114", "1"), format_error, ref("250")], [format_error, ref("250")]),
        answered("b", [ref("114", "1")], []),
    ]
    m = summarize(items, k_chunks=8)
    # 4 cited refs in total: 1 format error, 1 well-formed but not retrieved.
    assert m["cited_n"] == 4
    assert m["invalid_citation_rate"] == 0.5
    assert m["invalid_format_rate"] == 0.25
    assert m["invalid_not_retrieved_rate"] == 0.25
    assert m["answers_with_format_error_n"] == 1
