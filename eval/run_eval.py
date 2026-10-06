"""Eval harness: golden split -> pipeline -> metrics + judge -> result JSON (``make eval``).

    uv run python -m eval.run_eval --label baseline --split dev [--retrieval-only]

Runs ``answer_question`` (the unchanged Phase 2 pipeline) once per golden item,
sequentially, judges the answers, and writes ``eval/results/<UTC ts>_<label>_<split>.json``
with everything needed to reproduce and compare the run (CLAUDE.md §8). A run on a golden
set that is not fully verified gets the ``_unverified`` label suffix.

Budget guard: the cumulative cost is checked after every answer and every judge call; past
``eval.max_cost_usd`` the run stops, saves a partial result (``budget_exceeded: true``)
and exits non-zero.
"""

import argparse
import hashlib
import logging
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import chromadb
from chromadb.api.models.Collection import Collection

from eval.judge import REFUSED_SCORE, judge_answer, skip_reason
from eval.metrics import summarize
from labor_code_rag.config import Settings, load_settings
from labor_code_rag.errors import LaborCodeRagError
from labor_code_rag.generation.answer import answer_question
from labor_code_rag.generation.citations import answer_refs
from labor_code_rag.generation.prompts import JUDGE_PROMPT_VERSION, PROMPT_VERSION
from labor_code_rag.ingest.index import collection_name, open_collection
from labor_code_rag.llm import LLMClient, create_openai_client
from labor_code_rag.logging_setup import configure_logging, set_request_id
from labor_code_rag.models import (
    CATEGORIES,
    EvalItemResult,
    EvalResult,
    GoldenItem,
    GoldenVerified,
)
from labor_code_rag.retrieval.vector import search

# Explicit name: under `python -m` __name__ would be "__main__".
logger = logging.getLogger("eval.run_eval")

Split = Literal["dev", "test"]
UNVERIFIED_SUFFIX = "_unverified"
_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class EvalRun:
    items: list[EvalItemResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    budget_exceeded: bool = False
    stopped_reason: str | None = None


def load_golden(path: Path) -> list[GoldenItem]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [GoldenItem.model_validate_json(line) for line in lines if line.strip()]


def _retrieval_only(
    item: GoldenItem, llm: LLMClient, collection: Collection, k: int
) -> EvalItemResult:
    request_id = set_request_id()
    started = time.perf_counter()
    chunks, cost = search(item.question, llm, collection, k)
    return EvalItemResult(
        id=item.id,
        category=item.category,
        question=item.question,
        relevant_chunks=item.relevant_chunks,
        request_id=request_id,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
        retrieved_article_nos=list(dict.fromkeys(c.article_no for c in chunks)),
        latency_ms=round((time.perf_counter() - started) * 1000),
        cost_usd=cost,
    )


def _answer(item: GoldenItem, llm: LLMClient, collection: Collection, k: int) -> EvalItemResult:
    result = answer_question(item.question, llm, collection, k)
    return EvalItemResult(
        id=item.id,
        category=item.category,
        question=item.question,
        relevant_chunks=item.relevant_chunks,
        request_id=result.request_id,
        retrieved_chunk_ids=[c.chunk_id for c in result.retrieved],
        retrieved_article_nos=list(dict.fromkeys(c.article_no for c in result.retrieved)),
        answer=result.answer.text,
        refused=result.answer.refused,
        cited=answer_refs(result.answer),
        invalid_citations=result.invalid_citations,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
    )


def evaluate(
    items: Sequence[GoldenItem],
    llm: LLMClient,
    collection: Collection,
    k: int,
    max_cost_usd: float,
    retrieval_only: bool = False,
) -> EvalRun:
    """Run every item in order; stop early on the budget guard or a pipeline error."""
    run = EvalRun()

    def over_budget() -> bool:
        if run.total_cost_usd > max_cost_usd:
            run.budget_exceeded = True
            run.stopped_reason = "budget_exceeded"
            logger.warning(
                "eval.budget_exceeded",
                extra={"cost_usd": round(run.total_cost_usd, 6), "max_cost_usd": max_cost_usd},
            )
        return run.budget_exceeded

    for item in items:
        try:
            if retrieval_only:
                result = _retrieval_only(item, llm, collection, k)
                run.items.append(result)
                run.total_cost_usd += result.cost_usd
                if over_budget():
                    break
                continue

            result = _answer(item, llm, collection, k)
            run.total_cost_usd += result.cost_usd
            if over_budget():  # checked after the answer, before paying for the judge
                run.items.append(result)
                break
            reason = skip_reason(item, refused=bool(result.refused))
            if reason == "refused":
                result.judge = REFUSED_SCORE
            if reason is None:
                judged = judge_answer(item, result.answer or "", llm)
                result.judge = judged.value
                result.cost_usd += judged.cost_usd
                run.total_cost_usd += judged.cost_usd
            result.judge_skipped = reason
            run.items.append(result)
            if over_budget():
                break
        except LaborCodeRagError as exc:  # already logged as request.failed / llm errors
            run.stopped_reason = f"error on {item.id}: {type(exc).__name__}: {exc}"
            logger.error("eval.item_failed", extra={"item_id": item.id, "error": str(exc)})
            break
    return run


def result_label(label: str, verified_n: int, total_n: int) -> str:
    """Append ``_unverified`` unless every golden item is owner-verified."""
    return label if verified_n == total_n else label + UNVERIFIED_SUFFIX


def git_info() -> tuple[str, bool]:
    """HEAD sha and whether tracked files differ from it (untracked result files ignored)."""
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return sha or "unknown", bool(status)


def build_result(
    run: EvalRun,
    golden: Sequence[GoldenItem],
    *,
    label: str,
    split: Split,
    settings: Settings,
    golden_path: Path,
    retrieval_only: bool,
    created_at: datetime,
    wall_time_s: float,
) -> EvalResult:
    if settings.llm is None:
        raise ValueError("missing 'llm' section in config.yaml")
    k = settings.retrieval.k
    verified_n = sum(i.verified for i in golden)
    sha, dirty = git_info()
    by_category = {
        cat: summarize([i for i in run.items if i.category == cat], k)
        for cat in CATEGORIES
        if any(i.category == cat for i in run.items)
    }
    return EvalResult(
        label=result_label(label, verified_n, len(golden)),
        split=split,
        created_at=created_at,
        git_sha=sha,
        git_dirty=dirty,
        config=settings.model_dump(mode="json", exclude={"openai_api_key"}),
        prompt_version=PROMPT_VERSION,
        judge_prompt_version=JUDGE_PROMPT_VERSION,
        embedding_model=settings.llm.embedding_model,
        answer_model=settings.llm.answer_model,
        judge_model=settings.llm.judge_model,
        k=k,
        retrieval_only=retrieval_only,
        golden_file=golden_path.as_posix(),
        golden_sha256=hashlib.sha256(golden_path.read_bytes()).hexdigest(),
        golden_verified=GoldenVerified(verified_n=verified_n, total_n=len(golden)),
        metrics=summarize(run.items, k),
        metrics_by_category=by_category,
        items=run.items,
        total_cost_usd=run.total_cost_usd,
        wall_time_s=round(wall_time_s, 2),
        budget_exceeded=run.budget_exceeded,
        stopped_reason=run.stopped_reason,
    )


def write_result(result: EvalResult, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.created_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"{stamp}_{result.label}_{result.split}.json"
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


# Rows of the stdout summary; the result file has every metric.
SUMMARY_METRICS = (
    "n",
    "mrr",
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@8",
    "precision@8",
    "chunk_recall@8",
    "false_refusal_rate",
    "oos_refusal_rate",
    "invalid_citation_rate",
    "invalid_format_rate",
    "invalid_not_retrieved_rate",
    "answers_with_format_error_n",
    "citation_precision_article",
    "citation_precision_point",
    "article_only_citation_share",
    "judge_accuracy",
    "judge_completeness",
    "judge_relevance",
    "judge_accuracy_answered",
    "latency_p50_ms",
    "latency_p95_ms",
    "cost_mean_usd",
)


def fmt(value: float | None, metric: str = "") -> str:
    if value is None:
        return "—"
    if metric == "n" or metric.endswith(("_n", "_ms")):
        return f"{value:.0f}"
    if metric.endswith("_usd"):
        return f"{value:.5f}"
    return f"{value:.3f}"


def summary_table(result: EvalResult) -> str:
    cats = list(result.metrics_by_category)
    lines = [
        f"| metric | overall | {' | '.join(cats)} |",
        f"|---|---|{'---|' * len(cats)}",
    ]
    for metric in SUMMARY_METRICS:
        if metric not in result.metrics:
            continue
        cells = [fmt(result.metrics_by_category[c].get(metric), metric) for c in cats]
        lines.append(f"| {metric} | {fmt(result.metrics[metric], metric)} | {' | '.join(cells)} |")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--label", required=True, help="run label, e.g. baseline")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--retrieval-only", action="store_true", help="no answer/judge calls")
    parser.add_argument("--log-level", default="WARNING", help="INFO shows every stage")
    args = parser.parse_args(argv)
    if not _LABEL_RE.match(args.label):
        parser.error("label: letters, digits, '_' and '-' only")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings()
    # Logs go to stdout like everywhere else; WARNING by default keeps the summary readable.
    configure_logging(args.log_level, "console", settings.logging.log_payloads)
    if settings.llm is None:
        raise SystemExit("missing 'llm' section in config.yaml")

    golden_path = settings.eval.golden_dir / f"{args.split}.jsonl"
    golden = load_golden(golden_path)
    llm = LLMClient(create_openai_client(settings.llm, settings.openai_api_key), settings.llm)
    chroma = chromadb.PersistentClient(path=str(settings.index.chroma_dir))
    name = collection_name(settings.index.collection_prefix, settings.llm.embedding_model)
    if name not in [c.name for c in chroma.list_collections()]:
        raise SystemExit(f"collection {name} missing; run `make index`")
    collection = open_collection(chroma, name)

    created_at = datetime.now(UTC)
    started = time.perf_counter()
    run = evaluate(
        golden,
        llm,
        collection,
        settings.retrieval.k,
        settings.eval.max_cost_usd,
        retrieval_only=args.retrieval_only,
    )
    result = build_result(
        run,
        golden,
        label=args.label,
        split=args.split,
        settings=settings,
        golden_path=golden_path,
        retrieval_only=args.retrieval_only,
        created_at=created_at,
        wall_time_s=time.perf_counter() - started,
    )
    path = write_result(result, settings.eval.results_dir)

    out = sys.stdout
    v = result.golden_verified
    if v.verified_n < v.total_n:
        out.write(
            f"\nWARNING: golden set not verified ({v.verified_n}/{v.total_n} items); "
            f"label is '{result.label}', numbers are provisional.\n"
        )
    out.write(
        f"\n{result.label} · {result.split} · git {result.git_sha[:7]}"
        f"{' (dirty)' if result.git_dirty else ''} · prompt {result.prompt_version} · "
        f"judge {result.judge_prompt_version} · k={result.k}\n\n"
    )
    out.write(summary_table(result) + "\n\n")
    out.write(
        f"items {len(result.items)}/{len(golden)} · cost ${result.total_cost_usd:.4f} · "
        f"wall {result.wall_time_s:.0f}s\nsaved {path}\n"
    )
    if run.stopped_reason:
        out.write(f"STOPPED: {run.stopped_reason}\n")
        return 2 if run.budget_exceeded else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
