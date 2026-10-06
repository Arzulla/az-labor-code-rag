"""Metric deltas between two eval result files (``make compare A=<file> B=<file>``).

Delta = B - A. Differences that small are noise: run each configuration at least twice
(CLAUDE.md §8) and compare against the run-to-run spread before calling a change a win.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from eval.run_eval import fmt
from labor_code_rag.models import EvalResult

# Per-category deltas only for the headline metrics; the overall table has everything.
HEADLINE = ("mrr", "recall@5", "recall@8", "false_refusal_rate", "judge_accuracy")


def load_result(path: Path) -> EvalResult:
    return EvalResult.model_validate_json(path.read_text(encoding="utf-8"))


def delta(a: float | None, b: float | None) -> float | None:
    """b - a; None if either side is undefined."""
    return None if a is None or b is None else b - a


def _signed(value: float | None, metric: str) -> str:
    if value is None:
        return "—"
    text = fmt(abs(value), metric)
    return ("+" if value > 0 else "-" if value < 0 else "±") + text


def warnings(a: EvalResult, b: EvalResult) -> list[str]:
    """Differences that make the comparison invalid or need a caveat."""
    out = []
    if a.split != b.split:
        out.append(f"different splits: {a.split} vs {b.split}")
    if a.golden_sha256 != b.golden_sha256:
        out.append("different golden files (sha256): numbers are not comparable")
    if a.judge_model != b.judge_model or a.judge_prompt_version != b.judge_prompt_version:
        out.append("different judge model/prompt: judge scores are not comparable")
    if a.retrieval_only != b.retrieval_only:
        out.append("one run is --retrieval-only: answer metrics missing on one side")
    for run in (a, b):
        if run.stopped_reason:
            out.append(f"{run.label}: incomplete run ({run.stopped_reason})")
    return out


def compare_table(a: EvalResult, b: EvalResult) -> str:
    lines = ["| metric | A | B | B - A |", "|---|---|---|---|"]
    for metric in a.metrics:
        va, vb = a.metrics.get(metric), b.metrics.get(metric)
        lines.append(
            f"| {metric} | {fmt(va, metric)} | {fmt(vb, metric)} "
            f"| {_signed(delta(va, vb), metric)} |"
        )
    cats = [c for c in a.metrics_by_category if c in b.metrics_by_category]
    if cats:
        lines += [
            "",
            f"| category | {' | '.join(f'Δ {m}' for m in HEADLINE)} |",
            f"|---|{'---|' * len(HEADLINE)}",
        ]
        for cat in cats:
            ma, mb = a.metrics_by_category[cat], b.metrics_by_category[cat]
            cells = [_signed(delta(ma.get(m), mb.get(m)), m) for m in HEADLINE]
            lines.append(f"| {cat} | {' | '.join(cells)} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B - A metric deltas of two eval results")
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args(argv)
    a, b = load_result(args.a), load_result(args.b)
    out = sys.stdout
    out.write(f"A: {args.a.name} ({a.label}, {a.split}, git {a.git_sha[:7]})\n")
    out.write(f"B: {args.b.name} ({b.label}, {b.split}, git {b.git_sha[:7]})\n")
    for warning in warnings(a, b):
        out.write(f"WARNING: {warning}\n")
    out.write("\n" + compare_table(a, b) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
