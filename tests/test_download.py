import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from labor_code_rag.errors import SourceDownloadError
from labor_code_rag.ingest.download import Fetch, FetchResult, SourceMeta, download_source

SOURCE_URL = "https://frameworks.e-qanun.az/46/f_46943.html"
UI_URL = "https://e-qanun.az/framework/46943"
FIXED_NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
# windows-1251 bytes that are invalid UTF-8, plus an entity as e-qanun.az uses for "ə".
BODY = b"<html><meta charset=windows-1251><p>\xc0\xe7\xe5\xf0 &#601;m&#601;k</p></html>"


class FakeFetch:
    """Stands in for ``urllib_fetch``; records calls instead of hitting the network."""

    def __init__(
        self,
        body: bytes = BODY,
        last_modified: str | None = "Mon, 01 Sep 2026 10:00:00 GMT",
        etag: str | None = '"abc123"',
    ) -> None:
        self.result = FetchResult(body=body, last_modified=last_modified, etag=etag)
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout_s: float) -> FetchResult:
        self.calls.append((url, timeout_s))
        return self.result


def _download(tmp_path: Path, fetch: Fetch) -> SourceMeta:
    return download_source(
        SOURCE_URL, UI_URL, tmp_path / "raw", 12.5, fetch=fetch, now=lambda: FIXED_NOW
    )


def test_writes_bytes_unchanged(tmp_path: Path) -> None:
    fetch = FakeFetch()
    _download(tmp_path, fetch)

    assert (tmp_path / "raw" / "f_46943.html").read_bytes() == BODY
    assert fetch.calls == [(SOURCE_URL, 12.5)]


def test_meta_json_fields(tmp_path: Path) -> None:
    returned = _download(tmp_path, FakeFetch())

    raw = json.loads((tmp_path / "raw" / "f_46943.meta.json").read_text(encoding="utf-8"))
    assert raw == {
        "source_url": SOURCE_URL,
        "ui_url": UI_URL,
        "downloaded_at": "2026-10-05T12:00:00Z",
        "sha256": hashlib.sha256(BODY).hexdigest(),
        "size_bytes": len(BODY),
        "last_modified": "Mon, 01 Sep 2026 10:00:00 GMT",
        "etag": '"abc123"',
    }
    assert SourceMeta.model_validate(raw) == returned


def test_missing_headers_are_null(tmp_path: Path) -> None:
    _download(tmp_path, FakeFetch(last_modified=None, etag=None))

    raw = json.loads((tmp_path / "raw" / "f_46943.meta.json").read_text(encoding="utf-8"))
    assert raw["last_modified"] is None
    assert raw["etag"] is None


def test_fetch_failure_raises_and_writes_nothing(tmp_path: Path) -> None:
    def failing_fetch(url: str, timeout_s: float) -> FetchResult:
        raise SourceDownloadError("HTTP 503")

    with pytest.raises(SourceDownloadError, match="503"):
        _download(tmp_path, failing_fetch)
    assert not (tmp_path / "raw").exists()


def test_empty_body_raises(tmp_path: Path) -> None:
    with pytest.raises(SourceDownloadError, match="empty"):
        _download(tmp_path, FakeFetch(body=b""))
    assert not (tmp_path / "raw").exists()


def test_logs_source_downloaded(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="labor_code_rag.ingest.download"):
        _download(tmp_path, FakeFetch())

    (record,) = [r for r in caplog.records if r.getMessage() == "source.downloaded"]
    assert record.levelno == logging.INFO
    assert record.size_bytes == len(BODY)
    assert record.sha256 == hashlib.sha256(BODY).hexdigest()
    assert record.last_modified == "Mon, 01 Sep 2026 10:00:00 GMT"
    assert isinstance(record.latency_ms, int)
