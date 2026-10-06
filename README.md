# az-labor-code-rag

RAG assistant for the Labor Code of the Republic of Azerbaijan (Əmək Məcəlləsi), answering
in Azerbaijani with article citations.

> **Not legal advice.** Answers are generated from the law text for information only and
> may be incomplete or wrong. Consult a qualified lawyer for decisions.

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and an OpenAI API key.

```bash
cp .env.example .env          # then set OPENAI_API_KEY=... (never commit .env)
make setup                    # install dependencies
make data                     # download the official law text
make ingest                   # parse into 988 chunks (data/processed/chunks.jsonl)
make index                    # embed + store in Chroma (~$0.005, cached on re-run)
make app                      # Gradio UI on http://127.0.0.1:7860
```

`make test` / `make lint` run offline; `make smoke` runs 5 sample questions against the real
API (~$0.005).

## Architecture (Phase 2 baseline)

question → NFC normalize → embed (`text-embedding-3-small`) → Chroma cosine top-8 →
`gpt-4.1-mini` with chunks in `<article>` delimiters, structured output
`{text, citations, refused}` → citation check against the retrieved set → answer with
`Maddə 114.2` citations, or a fixed refusal. All API calls go through `llm.py` (timeouts,
tenacity retries, cost logging). Models are pinned in `config.yaml`; decisions in
`PROJECT_BRIEF.md` (ADR-004, ADR-005).

Known baseline limitation: dense retrieval alone often misses the right article (exact
article numbers, colloquial wording), so the assistant refuses rather than guesses. Hybrid
search, query rewriting and reranking come in Phase 4, measured against a golden set.

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
