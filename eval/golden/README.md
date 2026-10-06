# Golden set

67 questions about the Azerbaijan Labor Code with reference answers and gold chunks, used
by `make eval` (decision record: ADR-006 in `PROJECT_BRIEF.md`).

| File | Items | factual | exact_article | colloquial | multi_article | out_of_scope |
|---|---|---|---|---|---|---|
| `dev.jsonl` | 38 | 12 | 6 | 8 | 6 | 6 |
| `test.jsonl` | 29 | 9 | 4 | 7 | 5 | 4 |
| total | 67 | 21 | 10 | 15 | 11 | 10 |

Iterate on `dev`; `test` is only for final numbers (CLAUDE.md §8).

## Schema (`GoldenItem` in `src/labor_code_rag/models.py`)

One JSON object per line:

| Field | Meaning |
|---|---|
| `id` | `fact-07`, `coll-03`, `exact-01`, `multi-05`, `oos-02`; unique across both files |
| `question` | Azerbaijani, phrased as a user would ask |
| `expected_answer` | Short Azerbaijani reference answer with `Maddə N.P` refs; what the judge compares against |
| `relevant_chunks` | Gold chunk ids from `data/processed/chunks.jsonl` (`"114.2"`, or `"70"` for an article without numbered points); empty for `out_of_scope` |
| `category` | See below |
| `verified` | `false` until the owner has checked the item against the law; only the owner sets `true` |
| `notes` | Related chunks that are not gold, assumptions, doubts |

Article-level relevance is derived from `relevant_chunks` (`"114.2"` → article `114`), never
stored twice. Gold chunks are the **minimal set needed for a complete answer**; related but
optional chunks are named in `notes`, not in `relevant_chunks`.

## Categories

| Category | Definition | Example |
|---|---|---|
| `factual` | A fact stated in one place, asked in roughly the law's own terms | "Sınaq müddəti ən çoxu nə qədər ola bilər?" |
| `exact_article` | The question names the article (and maybe the point) | "114-cü maddənin 3-cü hissəsinə görə kimlərə 30 gün…" |
| `colloquial` | Everyday wording that differs from the law's terms | "Maaşımı vaxtında vermirlər…", "Dekretə çıxıram…" |
| `multi_article` | A complete answer needs chunks from 2+ different articles | pregnant worker + redundancy + liquidation (79.1, 79.2, 70) |
| `out_of_scope` | Plausible HR/legal questions the Labor Code does not answer (taxes, pensions, other laws) plus 2 clearly unrelated ones; the expected behaviour is a refusal | "Pensiya yaşı neçədir?" |

`follow_up` from the original eval plan is skipped: the system is single-turn (ADR-006).

## How the items were written

1. Read the law text in `data/processed/chunks.jsonl`, picked chunks spread over the code:
   12 of the 13 parts (bölmə) have gold chunks, not only leave and dismissal (part XIII,
   final provisions on supervision and liability, has none). The mix is weighted toward
   parts III (employment contract) and V (leave), where most real questions are.
2. Wrote the question **from the chunk**, then the reference answer from the same text.
3. The retriever was **never** run to choose or check gold chunks: that would bias the set
   toward what the baseline already finds.
4. `tests/test_golden.py` checks: both files parse, ids are unique, no question appears in
   both splits, every gold chunk id exists in `chunks.jsonl`, out_of_scope items have no
   chunks, and the files match the seeded split.

The draft was written by an LLM (Claude) and is **unverified** until the owner checks every
item against the law. Results on an unverified set carry the `_unverified` label suffix and
the README marks them provisional.

## Split

Stratified by category, assigned once before any eval run: per category the ids are sorted,
shuffled with `random.Random("20261006:<category>")`, and the first `round(0.55 · n)` go to
dev (`eval/split.py`, **seed `20261006`**).

## Frozen rule

After verification the golden set is **frozen**. Never edit it to improve a score
(CLAUDE.md §8). Any change (fixing a wrong reference, adding items) needs an ADR in
`PROJECT_BRIEF.md` and a baseline re-run (`make baseline`), because results on different
golden files are not comparable (`golden_sha256` in every result file; `make compare` warns).
