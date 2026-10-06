"""Regenerate the README baseline table and ``eval/calibration.jsonl`` (last step of
``make baseline``).

Inputs: the two newest ``baseline`` dev results and the newest ``baseline`` test result
in ``eval/results/`` (``_unverified`` labels included). The README table shows the dev mean
of the 2 runs, their spread, and test. Calibration takes 15 judged dev answers from the
first of the two dev runs, stratified by category; a ``human_accuracy`` the owner already
filled is kept when the item id and the answer text are unchanged.
"""

import random
import sys
from collections.abc import Sequence
from pathlib import Path

from eval.calibrate import CalibrationItem, load_calibration, write_calibration
from eval.compare import load_result
from eval.run_eval import fmt, load_golden
from labor_code_rag.config import load_settings
from labor_code_rag.models import CATEGORIES, EvalItemResult, EvalResult, GoldenItem

README_PATH = Path("README.md")
README_START = "<!-- baseline-results:start -->"
README_END = "<!-- baseline-results:end -->"
BASELINE_LABELS = ("baseline", "baseline_unverified")
CALIBRATION_SEED = 20261006

# (metric key, README row name)
README_ROWS = (
    ("mrr", "MRR (article)"),
    ("recall@1", "Recall@1 (article)"),
    ("recall@5", "Recall@5 (article)"),
    ("recall@8", "Recall@8 (article)"),
    ("chunk_recall@8", "ChunkRecall@8 (point)"),
    ("precision@8", "Precision@8 (article, capped, see note)"),
    ("false_refusal_rate", "False-refusal rate (in-scope, ↓)"),
    ("oos_refusal_rate", "Out-of-scope refusal rate (↑)"),
    ("invalid_citation_rate", "Invalid citation rate (not retrieved, ↓)"),
    ("citation_precision_article", "Citation precision vs gold (article)"),
    ("citation_precision_point", "Citation precision vs gold (point)"),
    ("article_only_citation_share", "Article-only citation share"),
    ("judge_accuracy", "Judge accuracy (1-5, refusals = 1)"),
    ("judge_completeness", "Judge completeness (1-5, refusals = 1)"),
    ("judge_relevance", "Judge relevance (1-5, refusals = 1)"),
    ("judge_accuracy_answered", "Judge accuracy, answered only"),
    ("latency_p50_ms", "Latency p50 (ms)"),
    ("latency_p95_ms", "Latency p95 (ms)"),
    ("cost_mean_usd", "Cost per query (USD, answer + judge)"),
)


def find_baseline(
    results_dir: Path,
) -> tuple[list[tuple[Path, EvalResult]], tuple[Path, EvalResult]]:
    """The two newest complete baseline dev runs (oldest first) and the newest test run."""
    runs = [(p, load_result(p)) for p in sorted(results_dir.glob("*.json"))]
    base = [
        (p, r)
        for p, r in runs
        if r.label in BASELINE_LABELS and not r.retrieval_only and r.stopped_reason is None
    ]
    dev = [(p, r) for p, r in base if r.split == "dev"][-2:]
    test = [(p, r) for p, r in base if r.split == "test"][-1:]
    if len(dev) < 2 or not test:
        raise SystemExit("need 2 baseline dev runs and 1 baseline test run: run `make baseline`")
    return dev, test[0]


def mean_metric(results: Sequence[EvalResult], metric: str) -> float | None:
    """Mean over runs; None if any run lacks the metric."""
    values = [r.metrics.get(metric) for r in results]
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean and len(clean) == len(values) else None


def spread(results: Sequence[EvalResult], metric: str) -> float | None:
    """max - min over runs: the run-to-run noise of this metric."""
    values = [r.metrics.get(metric) for r in results]
    clean = [v for v in values if v is not None]
    return max(clean) - min(clean) if clean and len(clean) == len(values) else None


def readme_block(dev: Sequence[tuple[Path, EvalResult]], test: tuple[Path, EvalResult]) -> str:
    dev_results = [r for _, r in dev]
    test_path, test_result = test
    unverified = any(
        r.golden_verified.verified_n < r.golden_verified.total_n
        for r in [*dev_results, test_result]
    )
    first = dev_results[0]
    lines = [README_START]
    if unverified:
        lines += [
            "",
            "> **Provisional (golden set not yet verified).** The golden items were drafted "
            "from the law text and await the owner's check; numbers change after verification.",
        ]
    lines += [
        "",
        f"Pipeline: `{first.embedding_model}` top-{first.k} → `{first.answer_model}` "
        f"(prompt `{first.prompt_version}`); judge `{first.judge_model}` "
        f"(`{first.judge_prompt_version}`); git `{first.git_sha[:7]}`. "
        f"Dev: {first.golden_verified.total_n} items × 2 runs; "
        f"test: {test_result.golden_verified.total_n} items × 1 run.",
        "",
        "| Metric | Dev (mean of 2) | Dev spread | Test |",
        "|---|---|---|---|",
    ]
    for key, name in README_ROWS:
        lines.append(
            f"| {name} | {fmt(mean_metric(dev_results, key), key)} "
            f"| {fmt(spread(dev_results, key), key)} "
            f"| {fmt(test_result.metrics.get(key), key)} |"
        )
    files = ", ".join(f"`{p.name}`" for p, _ in dev) + f"; test `{test_path.name}`"
    lines += ["", f"Source: `eval/results/` dev {files}.", "", README_END]
    return "\n".join(lines)


def replace_block(text: str, block: str) -> str:
    """Replace the marked README section; the markers must already exist."""
    start, end = text.find(README_START), text.find(README_END)
    if start == -1 or end == -1:
        raise ValueError(f"README markers {README_START} / {README_END} not found")
    return text[:start] + block + text[end + len(README_END) :]


def select_calibration(items: Sequence[EvalItemResult], n: int, seed: int) -> list[EvalItemResult]:
    """Up to ``n`` judged in-scope items, round-robin over categories, seeded per category."""
    pools = []
    for category in CATEGORIES:
        pool = [
            i
            for i in items
            if i.category == category and i.judge is not None and i.judge_skipped is None
        ]
        pool.sort(key=lambda i: i.id)
        random.Random(f"{seed}:{category}").shuffle(pool)
        pools.append(pool)
    chosen: list[EvalItemResult] = []
    while len(chosen) < n and any(pools):
        for pool in pools:
            if pool and len(chosen) < n:
                chosen.append(pool.pop(0))
    return chosen


def build_calibration(
    items: Sequence[EvalItemResult],
    golden: dict[str, GoldenItem],
    source: str,
    previous: Sequence[CalibrationItem],
    n: int,
) -> tuple[list[CalibrationItem], int]:
    """New calibration rows; returns them and how many filled human scores were dropped."""
    kept = {(p.id, p.answer): p.human_accuracy for p in previous if p.human_accuracy}
    rows = []
    for item in select_calibration(items, n, CALIBRATION_SEED):
        if item.judge is None or item.answer is None:  # excluded by selection; for mypy
            continue
        rows.append(
            CalibrationItem(
                id=item.id,
                category=item.category,
                question=item.question,
                expected_answer=golden[item.id].expected_answer,
                answer=item.answer,
                judge_accuracy=item.judge.accuracy,
                judge_completeness=item.judge.completeness,
                judge_relevance=item.judge.relevance,
                judge_rationale=item.judge.rationale,
                human_accuracy=kept.get((item.id, item.answer)),
                source_result=source,
            )
        )
    dropped = len(kept) - sum(r.human_accuracy is not None for r in rows)
    return rows, dropped


def main() -> int:
    settings = load_settings()
    dev, test = find_baseline(settings.eval.results_dir)

    README_PATH.write_text(
        replace_block(README_PATH.read_text(encoding="utf-8"), readme_block(dev, test)),
        encoding="utf-8",
    )
    sys.stdout.write(f"updated {README_PATH} baseline table\n")

    golden = {g.id: g for g in load_golden(settings.eval.golden_dir / "dev.jsonl")}
    source_path, source = dev[0]
    path = settings.eval.calibration_path
    rows, dropped = build_calibration(
        source.items, golden, source_path.name, load_calibration(path), settings.eval.calibration_n
    )
    write_calibration(path, rows)
    sys.stdout.write(f"wrote {path}: {len(rows)} items from {source_path.name}\n")
    if dropped:
        sys.stdout.write(
            f"WARNING: {dropped} filled human_accuracy score(s) dropped: item no longer "
            "selected or its answer changed; re-score those rows\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
