import json
import logging
from collections.abc import Iterator

import pytest

from labor_code_rag.logging_setup import (
    configure_logging,
    get_request_id,
    log_payload,
    set_request_id,
)

logger = logging.getLogger("labor_code_rag.test")


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    set_request_id("-")


def _lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return [line for line in capsys.readouterr().out.splitlines() if line]


def test_json_line_has_base_fields_and_extras(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json", log_payloads=False)
    set_request_id("req-1")

    logger.info("retrieval.completed", extra={"source": "bm25", "k": 5, "article_nos": [114, 115]})

    (line,) = _lines(capsys)
    record = json.loads(line)
    assert record["event"] == "retrieval.completed"
    assert record["level"] == "INFO"
    assert record["logger"] == "labor_code_rag.test"
    assert record["request_id"] == "req-1"
    assert record["ts"].endswith("+00:00")
    assert (record["source"], record["k"], record["article_nos"]) == ("bm25", 5, [114, 115])


def test_json_keeps_azerbaijani_characters_unescaped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO", "json", log_payloads=False)

    logger.info("answer.refused", extra={"reason": "Əmək məzuniyyəti: ığşöüç"})

    (line,) = _lines(capsys)
    assert "Əmək məzuniyyəti: ığşöüç" in line
    assert "\\u" not in line


def test_exception_includes_stack_trace(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json", log_payloads=False)

    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("request.failed", extra={"stage": "retrieval"})

    record = json.loads(_lines(capsys)[0])
    assert record["level"] == "ERROR"
    assert "ValueError: boom" in record["exc_info"]


def test_level_filters_records(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING", "json", log_payloads=False)

    logger.info("request.received")
    logger.warning("llm.retry")

    assert [json.loads(x)["event"] for x in _lines(capsys)] == ["llm.retry"]


def test_console_format_is_readable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "console", log_payloads=False)
    set_request_id("req-2")

    logger.info("rerank.completed", extra={"input_n": 20, "output_n": 5})

    (line,) = _lines(capsys)
    assert "rerank.completed" in line
    assert "input_n=20" in line
    assert "[req-2]" in line


def test_repeated_configuration_does_not_duplicate_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO", "json", log_payloads=False)
    configure_logging("INFO", "json", log_payloads=False)

    logger.info("request.received")

    assert len(_lines(capsys)) == 1


def test_set_request_id_generates_when_omitted() -> None:
    generated = set_request_id()
    assert generated != "-"
    assert get_request_id() == generated


@pytest.mark.parametrize(("enabled", "expected_lines"), [(False, 0), (True, 1)])
def test_log_payload_only_when_enabled(
    capsys: pytest.CaptureFixture[str], enabled: bool, expected_lines: int
) -> None:
    configure_logging("DEBUG", "json", log_payloads=enabled)

    log_payload(logger, "request.payload", question="İşdən qovdular")

    lines = _lines(capsys)
    assert len(lines) == expected_lines
    if enabled:
        record = json.loads(lines[0])
        assert record["level"] == "DEBUG"
        assert record["question"] == "İşdən qovdular"
