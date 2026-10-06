"""Judge-vs-human calibration report (``make calibrate``), ADR-007.

``eval/calibration.jsonl`` holds 15 dev answers with the judge's scores and an empty
``human_accuracy`` that only the owner fills (1-5, same rubric as the judge). This script
reports agreement on accuracy: exact match, within ±1, and Spearman's rho. It prints
"pending" while no human score is filled. The answer and judge model are both GPT-4.1
(ADR-004 risk), so this is the check on self-preference bias.
"""

import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from eval.metrics import exact_agreement, spearman, within_agreement
from labor_code_rag.config import load_settings
from labor_code_rag.models import Category, Score


class CalibrationItem(BaseModel):
    id: str
    category: Category
    question: str
    expected_answer: str
    answer: str
    judge_accuracy: Score
    judge_completeness: Score
    judge_relevance: Score
    judge_rationale: str
    human_accuracy: Score | None = None  # filled by the owner only
    source_result: str  # result file the answer and judge scores come from


def load_calibration(path: Path) -> list[CalibrationItem]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [CalibrationItem.model_validate_json(line) for line in lines if line.strip()]


def write_calibration(path: Path, items: Sequence[CalibrationItem]) -> None:
    path.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")


def calibration_report(items: Sequence[CalibrationItem]) -> str:
    scored = [i for i in items if i.human_accuracy is not None]
    if not scored:
        return f"calibration: pending (0/{len(items)} human_accuracy scores filled)"
    judge = [i.judge_accuracy for i in scored]
    human = [i.human_accuracy for i in scored if i.human_accuracy is not None]
    rho = spearman(judge, human)
    exact = exact_agreement(judge, human) or 0.0
    within = within_agreement(judge, human) or 0.0
    lines = [
        f"calibration on accuracy: {len(scored)}/{len(items)} human scores",
        f"- exact agreement: {exact:.2f}",
        f"- within ±1:       {within:.2f}",
        f"- Spearman rho:    {'undefined (constant scores)' if rho is None else f'{rho:.2f}'}",
        f"- mean judge {sum(judge) / len(judge):.2f} vs human {sum(human) / len(human):.2f}",
    ]
    if len(scored) < len(items):
        lines.append("(partial: fill every human_accuracy before quoting these numbers)")
    return "\n".join(lines)


def main() -> int:
    settings = load_settings()
    items = load_calibration(settings.eval.calibration_path)
    if not items:
        sys.stdout.write(f"{settings.eval.calibration_path} missing: run `make baseline`\n")
        return 1
    sys.stdout.write(calibration_report(items) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
