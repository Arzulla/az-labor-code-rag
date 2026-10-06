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
API (~$0.005); `make eval` measures the pipeline on the golden set (see Evaluation).

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

## Evaluation (Phase 3)

A golden set of 67 Azerbaijani questions (`eval/golden/`, dev 38 / test 29) in five
categories: factual, exact_article, colloquial, multi_article, out_of_scope. Each item has a
reference answer and gold chunk ids (`114.2`). Retrieval is scored at article level (MRR,
Recall@k, Precision@k) plus chunk-level Recall@8; answers by an LLM judge (`gpt-4.1`,
temperature 0, rubric in `generation/prompts.py`) on accuracy, completeness and relevance;
citations against the retrieved set and against the gold chunks; refusals on both sides
(out-of-scope refused vs. in-scope wrongly refused); latency and cost. Decisions: ADR-006
(golden set) and ADR-007 (judge) in `PROJECT_BRIEF.md`.

```bash
make eval LABEL=my-change SPLIT=dev      # full run: answers + judge (~$0.12 on dev)
make eval-retrieval LABEL=my-change      # retrieval metrics only, no answer/judge calls
make compare A=eval/results/<a>.json B=eval/results/<b>.json   # B - A deltas
make baseline                            # dev ×2 + test ×1, then this table + calibration
make calibrate                           # judge vs. human scores (eval/calibration.jsonl)
```

Every run writes `eval/results/<UTC timestamp>_<label>_<split>.json` (git SHA, config,
prompt versions, models, golden sha256, metrics overall and per category, per-question
details, cost). A run stops when its cost exceeds `eval.max_cost_usd` (default $1.00).

### Baseline results

<!-- baseline-results:start -->

> **Provisional (golden set not yet verified).** The golden items were drafted from the law text and await the owner's check; numbers change after verification.

Pipeline: `text-embedding-3-small` top-8 → `gpt-4.1-mini-2025-04-14` (prompt `answer-v1`); judge `gpt-4.1-2025-04-14` (`judge-v1`); git `148e5d9`. Dev: 38 items × 2 runs; test: 29 items × 1 run.

| Metric | Dev (mean of 2) | Dev spread | Test |
|---|---|---|---|
| MRR (article) | 0.525 | 0.000 | 0.518 |
| Recall@1 (article) | 0.344 | 0.000 | 0.380 |
| Recall@5 (article) | 0.635 | 0.000 | 0.600 |
| Recall@8 (article) | 0.667 | 0.000 | 0.620 |
| ChunkRecall@8 (point) | 0.536 | 0.000 | 0.540 |
| Precision@8 (article, capped, see note) | 0.098 | 0.000 | 0.085 |
| False-refusal rate (in-scope, ↓) | 0.281 | 0.062 | 0.280 |
| Out-of-scope refusal rate (↑) | 1.000 | 0.000 | 1.000 |
| Invalid citation rate (total, ↓) | 0.172 | 0.020 | 0.179 |
| … of which format errors (`article_no` = "254.1") | 0.172 | 0.020 | 0.179 |
| … of which well-formed but not retrieved | 0.000 | 0.000 | 0.000 |
| Answers with a citation format error (count) | 4 | 1 | 5 |
| Citation precision vs gold (article) | 0.729 | 0.002 | 0.536 |
| Citation precision vs gold (point) | 0.501 | 0.029 | 0.464 |
| Article-only citation share | 0.174 | 0.077 | 0.107 |
| Judge accuracy (1-5, refusals = 1) | 3.203 | 0.094 | 3.360 |
| Judge completeness (1-5, refusals = 1) | 3.188 | 0.125 | 3.240 |
| Judge relevance (1-5, refusals = 1) | 3.641 | 0.156 | 3.840 |
| Judge accuracy, answered only | 4.068 | 0.136 | 4.278 |
| Latency p50 (ms) | 1615 | 78 | 1761 |
| Latency p95 (ms) | 3356 | 627 | 2796 |
| Cost per query (USD, answer + judge) | 0.00207 | 0.00015 | 0.00216 |

Source: `eval/results/` dev `20261006T120014Z_baseline_unverified_dev.json`, `20261006T120202Z_baseline_unverified_dev.json`; test `20261006T120359Z_baseline_unverified_test.json`.

<!-- baseline-results:end -->

How to read it: MRR and Recall@k are the headline retrieval numbers. Precision@8 is capped
by design: with 1-2 gold articles per question it cannot exceed 0.125-0.25, so a low value
is not a bad result. "Refusals = 1" means a refused in-scope question scores 1 on every
judge axis without a judge call; "answered only" excludes those. The judge is the same model
family as the answer model (ADR-004), so judge scores are calibrated against owner scores
on 15 dev answers (`make calibrate`). Every invalid citation in the baseline is a format error (the
model wrote the chunk id `254.1` into `article_no`), not a hallucinated article; it is an
open issue for Phase 4/5 (ADR-007).

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
