"""LLM judge: scores an answer against the golden reference (ADR-007).

The judge sees the question, the reference answer and the system answer, never the
retrieved chunks: it grades what the user would read, not how it was found. Calls go
through ``llm.py`` with the pinned judge model (ADR-004) and ``temperature=0``.

Skipped (no call): out_of_scope items (the refusal metric scores them) and refused in-scope
answers (scored 1/1/1 directly: a refusal answers nothing, and paying the judge to say so
adds cost and noise).
"""

from typing import Literal

from labor_code_rag.generation.prompts import JUDGE_PROMPT_VERSION, build_judge_messages
from labor_code_rag.llm import ChatResult, LLMClient
from labor_code_rag.models import GoldenItem, JudgeScore

JUDGE_TEMPERATURE = 0.0  # CLAUDE.md §5: judge calls are deterministic

REFUSED_SCORE = JudgeScore(
    accuracy=1,
    completeness=1,
    relevance=1,
    rationale="Refused an in-scope question: scored 1 without a judge call.",
)

SkipReason = Literal["out_of_scope", "refused"]


def skip_reason(item: GoldenItem, refused: bool) -> SkipReason | None:
    """Why this item gets no judge call, or None if it must be judged."""
    if item.category == "out_of_scope":
        return "out_of_scope"
    if refused:
        return "refused"
    return None


def judge_answer(item: GoldenItem, answer_text: str, llm: LLMClient) -> ChatResult[JudgeScore]:
    """One judge call. Raises ``LLMError`` like every ``llm.py`` call."""
    return llm.chat_structured(
        build_judge_messages(item.question, item.expected_answer, answer_text),
        JudgeScore,
        purpose="judge",
        model=llm.settings.judge_model,
        temperature=JUDGE_TEMPERATURE,
        prompt_version=JUDGE_PROMPT_VERSION,
    )
