"""Structured logging (see CLAUDE.md §7).

Deliberately does not import ``config``: the entry point reads settings and passes
level, format and ``log_payloads`` in.

Convention: the log message is the event name (``noun.verb_past``) and the ``extra``
dict carries the fields, e.g.
``logger.info("retrieval.completed", extra={"source": "bm25", "k": 20})``.
"""

import contextvars
import json
import logging
import sys
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_log_payloads = False

# Attributes every LogRecord has; anything else on a record came from ``extra``.
_STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}

_HANDLER_MARK = "_labor_code_rag_handler"


def set_request_id(request_id: str | None = None) -> str:
    """Bind a request id to the current context (MDC analogue); generates one if omitted."""
    value = request_id or uuid.uuid4().hex[:12]
    _request_id.set(value)
    return value


def get_request_id() -> str:
    return _request_id.get()


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; keeps Azerbaijani characters readable."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(_extra_fields(record))  # includes request_id from the filter
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable single line for local development."""

    def format(self, record: logging.LogRecord) -> str:
        fields = {k: v for k, v in _extra_fields(record).items() if k != "request_id"}
        rendered = " ".join(f"{k}={v}" for k, v in fields.items())
        line = (
            f"{datetime.fromtimestamp(record.created, UTC):%H:%M:%S} "
            f"{record.levelname:<7} [{_request_id.get()}] {record.name} "
            f"{record.getMessage()} {rendered}"
        ).rstrip()
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(level: str, fmt: Literal["json", "console"], log_payloads: bool) -> None:
    """Configure the root logger. Safe to call repeatedly (replaces our own handler)."""
    global _log_payloads
    _log_payloads = log_payloads

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    setattr(handler, _HANDLER_MARK, True)
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())


def log_payload(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log sensitive payloads (question, prompts) at DEBUG, only if ``log_payloads`` is on."""
    if _log_payloads:
        logger.debug(event, extra=fields)
