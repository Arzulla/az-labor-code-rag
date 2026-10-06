"""Question -> retrieve -> prompt -> structured answer -> citation check (baseline)."""

import logging
import time

from chromadb.api.models.Collection import Collection

from labor_code_rag.generation.citations import check_citations
from labor_code_rag.generation.prompts import PROMPT_VERSION, REFUSAL_TEXT, build_messages
from labor_code_rag.llm import LLMClient
from labor_code_rag.logging_setup import log_payload, set_request_id
from labor_code_rag.models import Answer, AnswerResult
from labor_code_rag.retrieval.vector import search
from labor_code_rag.text import normalize

logger = logging.getLogger(__name__)

# 0 = (near-)deterministic answers: the smoke run and Phase 3 evals must be reproducible.
ANSWER_TEMPERATURE = 0.0


def _refusal() -> Answer:
    return Answer(text=REFUSAL_TEXT, citations=[], refused=True)


def answer_question(question: str, llm: LLMClient, collection: Collection, k: int) -> AnswerResult:
    """Answer one question. Raises ``RetrievalError`` / ``LLMError`` after logging them."""
    request_id = set_request_id()  # every log line below carries this id
    started = time.perf_counter()
    question = normalize(question)
    logger.info("request.received", extra={"question_chars": len(question)})
    log_payload(logger, "request.payload", question=question)

    stage = "retrieval"  # which step failed, for request.failed
    try:
        retrieved, cost = search(question, llm, collection, k)
        if not retrieved:  # empty collection: nothing to ground an answer on
            logger.info("answer.refused", extra={"reason": "no_context"})
            answer = _refusal()
        else:
            stage = "generation"
            messages = build_messages(question, retrieved)
            log_payload(logger, "prompt.built", messages=messages)
            result = llm.chat_structured(
                messages,
                Answer,
                purpose="answer",
                model=llm.settings.answer_model,
                temperature=ANSWER_TEMPERATURE,
                prompt_version=PROMPT_VERSION,
            )
            cost += result.cost_usd
            answer = result.value
            if answer.refused:
                logger.info("answer.refused", extra={"reason": "model_refused"})
                answer = _refusal()  # one fixed wording, whatever the model wrote
        stage = "citation"
        check = check_citations(answer, retrieved)
    except Exception as exc:  # log with stage + stack trace, then re-raise (not swallowed)
        logger.error(
            "request.failed",
            exc_info=True,
            extra={"error_type": type(exc).__name__, "stage": stage},
        )
        raise

    latency_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "request.completed",
        extra={
            "total_latency_ms": latency_ms,
            "total_cost_usd": round(cost, 8),
            "cited_articles": [str(ref) for ref in check.valid],
            "invalid_citations": len(check.invalid),
            "refused": answer.refused,
        },
    )
    return AnswerResult(
        request_id=request_id,
        question=question,
        answer=answer,
        retrieved=retrieved,
        cited_articles=check.valid,
        invalid_citations=check.invalid,
        latency_ms=latency_ms,
        cost_usd=cost,
    )
