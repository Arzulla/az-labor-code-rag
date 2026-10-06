# az-labor-code-rag

RAG assistant for the Labor Code of the Republic of Azerbaijan (Əmək Məcəlləsi), answering
in Azerbaijani with article citations.

> **Not legal advice.** Answers are generated from the law text for information only and
> may be incomplete or wrong. Consult a qualified lawyer for decisions.

## Data source

- Official consolidated text from the Ministry of Justice legal database:
  <https://e-qanun.az/framework/46943> (file: <https://frameworks.e-qanun.az/46/f_46943.html>).
- Downloaded 2026-10-05 (`Last-Modified: Thu, 17 Sep 2026`); `make data` records the date,
  sha256 and headers in `data/raw/f_46943.meta.json`. The law changes; re-run `make data`
  and `make ingest` to refresh.
- The site states no terms of use, so the full text is not redistributed here: only a small
  test fixture is committed (`tests/fixtures/`).
- `make ingest` parses the HTML into 327 articles and 988 chunks (one per top-level point)
  in `data/processed/chunks.jsonl`. See ADR-001…003 in `PROJECT_BRIEF.md`.
