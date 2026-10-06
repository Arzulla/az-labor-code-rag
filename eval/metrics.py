"""Eval metrics: small pure functions, each with its formula (CLAUDE.md §8, ADR-006/007).

Retrieval metrics work at **article level** on a ranked, deduplicated list of article
numbers: the retriever returns chunks (``114.2``, ``114.3``, ``70``), and an article counts
once, at the rank of its first chunk. Out_of_scope items have no relevant articles, so
retrieval metrics skip them (the per-query functions raise on an empty relevant set, so a
caller that forgets to filter fails loudly instead of averaging in zeros).

``summarize`` turns per-question results into the aggregate dict saved in every result
file. Undefined aggregates (nothing to average) are ``None``, never 0.
"""

import math
from collections.abc import Iterable, Sequence
from typing import Literal

from labor_code_rag.models import EvalItemResult, article_of
from labor_code_rag.text import ArticleRef

KS: tuple[int, ...] = (1, 3, 5, 8)


# --- Retrieval ------------------------------------------------------------------------------


def dedup_ranked(article_nos: Iterable[str]) -> list[str]:
    """Rank-ordered unique articles: ``[114, 114, 70, 114, 79]`` -> ``[114, 70, 79]``."""
    return list(dict.fromkeys(article_nos))


def _require_relevant(relevant: Iterable[str]) -> set[str]:
    rel = set(relevant)
    if not rel:
        raise ValueError("empty relevant set: skip out_of_scope items before calling")
    return rel


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """RR = 1 / rank of the first relevant article (1-based); 0 if none is retrieved.

    MRR is the mean of RR over queries. ``ranked`` is deduplicated first, so a repeated
    article does not push the next one down.
    """
    rel = _require_relevant(relevant)
    for rank, article in enumerate(dedup_ranked(ranked), start=1):
        if article in rel:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Recall@k = |top-k articles ∩ relevant| / |relevant|.

    Top-k is taken from the deduplicated list; if it is shorter than k, all of it counts.
    """
    rel = _require_relevant(relevant)
    return len(set(dedup_ranked(ranked)[:k]) & rel) / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Precision@k = |top-k articles ∩ relevant| / k (trec_eval convention).

    The denominator is always k: if the deduplicated list is shorter than k, the missing
    slots count as non-relevant. With 1-2 relevant articles, Precision@8 is therefore capped
    at 0.125-0.25; read it as a relative number, MRR and Recall@k are the headline metrics.
    """
    rel = _require_relevant(relevant)
    return len(set(dedup_ranked(ranked)[:k]) & rel) / k


def chunk_recall(retrieved_chunk_ids: Sequence[str], relevant_chunks: Iterable[str]) -> float:
    """Chunk-level recall = |retrieved chunk ids ∩ relevant chunks| / |relevant chunks|.

    Citations are at point level (ADR-003), so finding article 114 but not chunk 114.2 is a
    miss here. With the full retrieved list (k=8 chunks) this is ChunkRecall@8.
    """
    rel = _require_relevant(relevant_chunks)
    return len(set(retrieved_chunk_ids) & rel) / len(rel)


# --- Citations ------------------------------------------------------------------------------


def citation_correct_article(ref: ArticleRef, relevant_chunks: Iterable[str]) -> bool:
    """Article-level hit: the cited article is one of the gold articles."""
    return ref.article in {article_of(c) for c in relevant_chunks}


def citation_correct_point(ref: ArticleRef, relevant_chunks: Iterable[str]) -> bool:
    """Point-level hit: the cited ref resolves to a gold chunk.

    ``Maddə N.P`` hits if chunk ``N.<top-level part of P>`` is gold (``114.2.1`` lives in
    chunk ``114.2``) or if article N is gold as one whole chunk ``N`` (no points to be more
    precise about). ``Maddə N`` without a point hits only if chunk ``N`` itself is gold.
    """
    gold = set(relevant_chunks)
    if ref.point is None:
        return ref.article in gold
    top_point = ref.point.split(".")[0]
    return f"{ref.article}.{top_point}" in gold or ref.article in gold


def citation_precision(
    pairs: Iterable[tuple[Sequence[ArticleRef], Sequence[str]]],
    level: Literal["article", "point"],
) -> float | None:
    """Pooled (micro) precision = correct cited refs / all cited refs, over all answers.

    ``pairs`` holds (refs cited in one answer, that item's relevant_chunks). ``level`` is
    ``"article"`` or ``"point"``. None when nothing was cited.
    """
    check = citation_correct_article if level == "article" else citation_correct_point
    cited = correct = 0
    for refs, relevant in pairs:
        cited += len(refs)
        correct += sum(check(ref, relevant) for ref in refs)
    return correct / cited if cited else None


def article_only_share(refs: Sequence[ArticleRef]) -> float | None:
    """Share of citations without a point = refs with point None / all refs.

    Explains a low point-level precision: ``Maddə 114`` cannot hit gold chunk ``114.2``.
    """
    return sum(ref.point is None for ref in refs) / len(refs) if refs else None


def rate(hits: int, total: int) -> float | None:
    """hits / total; None when total is 0 (e.g. invalid citations / all citations)."""
    return hits / total if total else None


# --- Ops ------------------------------------------------------------------------------------


def mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean; None for an empty sequence."""
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile: the value at 1-based rank ceil(p/100 * n) in sorted order.

    p50 of [10, 20, 30, 40] is 20 (rank 2), p95 is 40 (rank 4). Always an observed value,
    no interpolation. None for an empty sequence.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


# --- Calibration (judge vs human) -----------------------------------------------------------


def exact_agreement(a: Sequence[int], b: Sequence[int]) -> float | None:
    """Share of pairs with identical scores = |{i: a_i = b_i}| / n."""
    return within_agreement(a, b, tolerance=0)


def within_agreement(a: Sequence[int], b: Sequence[int], tolerance: int = 1) -> float | None:
    """Share of pairs whose scores differ by at most ``tolerance`` = |{i: |a_i-b_i| <= t}| / n."""
    if len(a) != len(b):
        raise ValueError("score lists differ in length")
    return mean([float(abs(x - y) <= tolerance) for x, y in zip(a, b, strict=True)])


def average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks; tied values share the mean of their ranks ([5, 3, 5] -> [2.5, 1, 2.5])."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for i in order[start : end + 1]:
            ranks[i] = (start + end) / 2 + 1  # mean of ranks start+1 .. end+1
        start = end + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman's rho = Pearson correlation of the average ranks of a and b.

    rho = cov(ra, rb) / (sd(ra) * sd(rb)). Handles ties (unlike the 1 - 6Σd²/(n(n²-1))
    shortcut). None if n < 2 or either side is constant (correlation undefined).
    """
    if len(a) != len(b):
        raise ValueError("score lists differ in length")
    if len(a) < 2:
        return None
    ra, rb = average_ranks(a), average_ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    var_a = sum((x - ma) ** 2 for x in ra)
    var_b = sum((y - mb) ** 2 for y in rb)
    if var_a == 0 or var_b == 0:
        return None
    return cov / math.sqrt(var_a * var_b)


# --- Aggregation ----------------------------------------------------------------------------


def summarize(items: Sequence[EvalItemResult], k_chunks: int) -> dict[str, float | None]:
    """Aggregate metrics for one set of per-question results (overall or one category).

    - Retrieval: in-scope items only (MRR, Recall@k, Precision@k, ChunkRecall@k_chunks).
    - Refusals: oos_refusal_rate over out_of_scope items (want high); false_refusal_rate
      over in-scope items (want low).
    - Citations: invalid rate over every answered item; gold precision and article-only
      share over answered in-scope items.
    - Judge: means over in-scope items (refusals scored 1 without a call) and over judged
      (answered) items only.
    - Ops: latency p50/p95, mean and total cost.
    Answer-side metrics are None in --retrieval-only runs.
    """
    in_scope = [i for i in items if i.category != "out_of_scope"]
    oos = [i for i in items if i.category == "out_of_scope"]
    m: dict[str, float | None] = {
        "n": float(len(items)),
        "retrieval_n": float(len(in_scope)),
    }

    ranked = [(i.retrieved_article_nos, _articles(i)) for i in in_scope]
    m["mrr"] = mean([reciprocal_rank(r, rel) for r, rel in ranked])
    for k in KS:
        m[f"recall@{k}"] = mean([recall_at_k(r, rel, k) for r, rel in ranked])
    for k in KS:
        m[f"precision@{k}"] = mean([precision_at_k(r, rel, k) for r, rel in ranked])
    m[f"chunk_recall@{k_chunks}"] = mean(
        [chunk_recall(i.retrieved_chunk_ids, i.relevant_chunks) for i in in_scope]
    )

    answered_items = [i for i in items if i.refused is not None]
    if not answered_items:  # --retrieval-only: no answer-side metrics
        _add_ops(m, items)
        return m

    m["oos_refusal_rate"] = rate(sum(bool(i.refused) for i in oos), len(oos))
    m["false_refusal_rate"] = rate(sum(bool(i.refused) for i in in_scope), len(in_scope))

    answered = [i for i in items if i.refused is False]
    answered_in_scope = [i for i in answered if i.category != "out_of_scope"]
    all_refs = [ref for i in answered for ref in i.cited]
    m["cited_n"] = float(len(all_refs))
    m["invalid_citation_rate"] = rate(
        sum(len(i.invalid_citations) for i in answered), len(all_refs)
    )
    pairs = [(i.cited, i.relevant_chunks) for i in answered_in_scope]
    m["citation_precision_article"] = citation_precision(pairs, "article")
    m["citation_precision_point"] = citation_precision(pairs, "point")
    m["article_only_citation_share"] = article_only_share(
        [ref for i in answered_in_scope for ref in i.cited]
    )

    scored = [i for i in in_scope if i.judge is not None]
    judged = [i for i in scored if i.judge_skipped is None]
    m["judge_n"] = float(len(judged))
    m["judge_skipped_refused_n"] = float(sum(i.judge_skipped == "refused" for i in scored))
    for field in ("accuracy", "completeness", "relevance"):
        m[f"judge_{field}"] = mean([float(getattr(i.judge, field)) for i in scored])
        m[f"judge_{field}_answered"] = mean([float(getattr(i.judge, field)) for i in judged])

    _add_ops(m, items)
    return m


def _articles(item: EvalItemResult) -> list[str]:
    return [article_of(c) for c in item.relevant_chunks]


def _add_ops(m: dict[str, float | None], items: Sequence[EvalItemResult]) -> None:
    latencies = [float(i.latency_ms) for i in items]
    costs = [i.cost_usd for i in items]
    m["latency_p50_ms"] = percentile(latencies, 50)
    m["latency_p95_ms"] = percentile(latencies, 95)
    m["cost_mean_usd"] = mean(costs)
    m["cost_total_usd"] = sum(costs) if costs else None
