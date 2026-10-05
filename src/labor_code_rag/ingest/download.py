"""Download the official Labor Code HTML (``make data``).

The response is stored as raw bytes (windows-1251, Word-exported HTML); decoding is the
parser's job. A ``<name>.meta.json`` sidecar records provenance. It is written *after*
the HTML, so an HTML file without its meta.json means an incomplete download.
"""

import hashlib
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from pydantic import BaseModel

from labor_code_rag.config import load_settings
from labor_code_rag.errors import SourceDownloadError
from labor_code_rag.logging_setup import configure_logging, set_request_id

logger = logging.getLogger(__name__)

_USER_AGENT = "labor-code-rag/0.1 (+https://github.com/Arzulla/az-labor-code-rag)"


@dataclass(frozen=True)
class FetchResult:
    body: bytes
    last_modified: str | None
    etag: str | None


Fetch = Callable[[str, float], FetchResult]
"""``(url, timeout_s) -> FetchResult``; raises ``SourceDownloadError`` on failure."""


class SourceMeta(BaseModel):
    source_url: str
    ui_url: str
    downloaded_at: datetime
    sha256: str
    size_bytes: int
    last_modified: str | None
    etag: str | None


def urllib_fetch(url: str, timeout_s: float) -> FetchResult:
    """Real HTTP GET via stdlib; the only network code in this module."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return FetchResult(
                body=response.read(),
                last_modified=response.headers.get("Last-Modified"),
                etag=response.headers.get("ETag"),
            )
    except urllib.error.HTTPError as exc:  # subclass of URLError, so it goes first
        raise SourceDownloadError(f"HTTP {exc.code} from {url}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SourceDownloadError(f"cannot reach {url}: {exc}") from exc


def _write_atomic(path: Path, data: bytes) -> None:
    """Write to a temp file, then rename: readers never see a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def download_source(
    source_url: str,
    ui_url: str,
    raw_dir: Path,
    timeout_s: float,
    fetch: Fetch = urllib_fetch,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> SourceMeta:
    """Fetch ``source_url`` into ``raw_dir`` as raw bytes plus a ``.meta.json`` sidecar."""
    name = PurePosixPath(urlparse(source_url).path).name
    if not name:
        raise SourceDownloadError(f"cannot derive a file name from {source_url}")

    started = time.perf_counter()
    result = fetch(source_url, timeout_s)
    latency_ms = round((time.perf_counter() - started) * 1000)
    if not result.body:
        raise SourceDownloadError(f"empty response body from {source_url}")

    meta = SourceMeta(
        source_url=source_url,
        ui_url=ui_url,
        downloaded_at=now(),
        sha256=hashlib.sha256(result.body).hexdigest(),
        size_bytes=len(result.body),
        last_modified=result.last_modified,
        etag=result.etag,
    )

    raw_dir.mkdir(parents=True, exist_ok=True)
    html_path = raw_dir / name
    _write_atomic(html_path, result.body)
    # Second on purpose: meta.json is the "download complete" marker.
    _write_atomic(
        html_path.with_suffix(".meta.json"),
        (meta.model_dump_json(indent=2) + "\n").encode("utf-8"),
    )

    logger.info(
        "source.downloaded",
        extra={
            "size_bytes": meta.size_bytes,
            "sha256": meta.sha256,
            "last_modified": meta.last_modified,
            "latency_ms": latency_ms,
        },
    )
    return meta


def main() -> None:
    settings = load_settings()
    configure_logging(
        settings.logging.level, settings.logging.format, settings.logging.log_payloads
    )
    set_request_id()
    if settings.data is None:
        raise SourceDownloadError("missing 'data' section in config.yaml")
    data = settings.data
    download_source(data.source_url, data.ui_url, data.raw_dir, data.download_timeout_s)


if __name__ == "__main__":
    # Under `python -m` this file runs as `__main__`; importing it again by its real name
    # makes `logger` report "labor_code_rag.ingest.download" instead of "__main__".
    from labor_code_rag.ingest.download import main as _main

    _main()
