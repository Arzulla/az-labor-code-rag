"""The only module that calls LLM / embedding APIs (CLAUDE.md §5).

``LLMClient`` wraps an OpenAI SDK client that the entry point creates once and passes in
(no module-level client). The SDK's own retries are disabled (``max_retries=0``) so that
``tenacity`` alone owns the retry policy: one place to read, test and log it.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, SecretStr, ValidationError
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tenacity.wait import wait_base

from labor_code_rag.config import LLMSettings
from labor_code_rag.errors import LLMError

logger = logging.getLogger(__name__)

# Transient = worth retrying. 5xx all map to InternalServerError in the SDK, and
# APITimeoutError is a subclass of APIConnectionError. Everything else (401, 400, 404,
# 422, ...) fails fast: retrying a bad key or a bad request only burns time.
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,  # 429
    openai.InternalServerError,  # 5xx
    openai.APIConnectionError,  # network errors and timeouts
)


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]]
    prompt_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class ChatResult[M: BaseModel]:
    value: M
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


def create_openai_client(settings: LLMSettings, api_key: SecretStr | None) -> OpenAI:
    """Build the SDK client once, in the entry point."""
    if api_key is None:
        raise LLMError("OPENAI_API_KEY is not set (see .env.example)")
    return OpenAI(
        api_key=api_key.get_secret_value(),
        base_url=settings.base_url,
        timeout=settings.timeout_s,
        max_retries=0,  # tenacity owns retries (see module docstring)
    )


def cost_usd(
    settings: LLMSettings, model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Price from the config table; the settings validator guarantees pinned models exist."""
    price = settings.prices_usd_per_1m[model]
    return (prompt_tokens * price.input + completion_tokens * price.output) / 1_000_000


def _log_retry(purpose: str, state: RetryCallState) -> None:
    error = state.outcome.exception() if state.outcome else None
    logger.warning(
        "llm.retry",
        extra={
            "purpose": purpose,
            "error_type": type(error).__name__,
            "attempt": state.attempt_number,
            "wait_s": round(state.next_action.sleep, 2) if state.next_action else None,
        },
    )


class LLMClient:
    def __init__(
        self,
        client: OpenAI,
        settings: LLMSettings,
        wait: wait_base | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        # Full jitter: random wait in [0, min(max, 2^n)] s, so parallel clients don't
        # retry in lockstep. Tests inject wait_none() to run instantly.
        self._wait = wait or wait_random_exponential(multiplier=1, max=20)

    @property
    def settings(self) -> LLMSettings:
        return self._settings

    def _with_retry[R](self, purpose: str, call: Callable[[], R]) -> tuple[R, int]:
        """Run ``call`` with the retry policy; return the result and the attempt number."""
        retrying = Retrying(
            retry=retry_if_exception_type(TRANSIENT_ERRORS),
            stop=stop_after_attempt(self._settings.max_attempts),
            wait=self._wait,
            before_sleep=lambda state: _log_retry(purpose, state),
            reraise=True,  # after the last attempt, raise the API error, not RetryError
        )
        try:
            result = retrying(call)
        except openai.OpenAIError as exc:
            raise LLMError(f"{purpose}: {type(exc).__name__}: {exc}") from exc
        except ValidationError as exc:  # structured output did not match the schema
            raise LLMError(f"{purpose}: invalid structured output: {exc}") from exc
        return result, retrying.statistics["attempt_number"]

    def _log_called(
        self,
        purpose: str,
        model: str,
        prompt_version: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        started: float,
        attempt: int,
    ) -> None:
        logger.info(
            "llm.called",
            extra={
                "purpose": purpose,
                "model": model,
                "prompt_version": prompt_version,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": round(cost, 8),
                "latency_ms": round((time.perf_counter() - started) * 1000),  # incl. retries
                "attempt": attempt,
            },
        )

    def embed(self, texts: Sequence[str], purpose: str) -> EmbedResult:
        """Embed ``texts`` with the pinned embedding model, in batches; order is kept."""
        model = self._settings.embedding_model
        size = self._settings.embed_batch_size
        vectors: list[list[float]] = []
        total_tokens = 0
        for start in range(0, len(texts), size):
            batch = list(texts[start : start + size])
            started = time.perf_counter()
            call = partial(
                self._client.embeddings.create,
                model=model,
                input=batch,
                timeout=self._settings.timeout_s,
            )
            response, attempt = self._with_retry(purpose, call)
            tokens = response.usage.prompt_tokens
            cost = cost_usd(self._settings, model, tokens, 0)
            self._log_called(purpose, model, None, tokens, 0, cost, started, attempt)
            # The API returns an index per item; sort instead of trusting the order.
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
            total_tokens += tokens
        return EmbedResult(vectors, total_tokens, cost_usd(self._settings, model, total_tokens, 0))

    def chat_structured[M: BaseModel](
        self,
        messages: list[ChatCompletionMessageParam],
        schema: type[M],
        *,
        purpose: str,
        model: str,
        temperature: float,
        prompt_version: str,
    ) -> ChatResult[M]:
        """One chat call whose reply is parsed into ``schema`` (OpenAI structured outputs)."""
        started = time.perf_counter()
        completion, attempt = self._with_retry(
            purpose,
            lambda: self._client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=schema,
                temperature=temperature,
                timeout=self._settings.timeout_s,
            ),
        )
        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        cost = cost_usd(self._settings, model, prompt_tokens, completion_tokens)
        self._log_called(
            purpose, model, prompt_version, prompt_tokens, completion_tokens, cost, started, attempt
        )
        message = completion.choices[0].message
        if message.refusal or message.parsed is None:
            # The model's safety refusal (not our "not in context" refusal): not retryable.
            raise LLMError(f"{purpose}: no parsed output (refusal={message.refusal!r})")
        return ChatResult(message.parsed, prompt_tokens, completion_tokens, cost)
