"""Phase 2 smoke run: 5 hand-picked questions -> markdown table on stdout (``make smoke``).

Not an eval (no golden set, no metrics): it shows the baseline end-to-end for the PR.
Phase 3's ``eval/run_eval.py`` replaces it. Needs OPENAI_API_KEY and ``make index``.
"""

import subprocess

import chromadb

from labor_code_rag.config import load_settings
from labor_code_rag.generation.answer import answer_question
from labor_code_rag.generation.prompts import PROMPT_VERSION
from labor_code_rag.ingest.index import collection_name, open_collection
from labor_code_rag.llm import LLMClient, create_openai_client
from labor_code_rag.logging_setup import configure_logging

QUESTIONS = [
    "Əmək məzuniyyəti minimum neçə gündür?",
    "114-cü maddə nə deyir?",
    "Məni işdən qovdular, mənə pul verməlidirlər?",
    "Hamilə qadını işdən çıxarmaq olar?",
    "Azərbaycanda gəlir vergisi neçə faizdir?",  # out of scope: must be refused
]


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def main() -> None:
    settings = load_settings()
    configure_logging("WARNING", "console", log_payloads=False)  # keep stdout a table
    if settings.llm is None:
        raise SystemExit("missing 'llm' section in config.yaml")
    llm = LLMClient(create_openai_client(settings.llm, settings.openai_api_key), settings.llm)
    chroma = chromadb.PersistentClient(path=str(settings.index.chroma_dir))
    name = collection_name(settings.index.collection_prefix, settings.llm.embedding_model)
    collection = open_collection(chroma, name)
    k = settings.retrieval.k

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    print(
        f"git `{sha}` · embedding `{settings.llm.embedding_model}` · "
        f"answer `{settings.llm.answer_model}` · prompt `{PROMPT_VERSION}` · k={k}\n"
    )
    print("| # | Question | Answer | Cited (valid) | Invalid | Retrieved article_nos | ms | USD |")
    print("|---|---|---|---|---|---|---|---|")
    total_cost = 0.0
    for i, question in enumerate(QUESTIONS, start=1):
        r = answer_question(question, llm, collection, k)
        total_cost += r.cost_usd
        retrieved = ", ".join(dict.fromkeys(c.article_no for c in r.retrieved))
        answer = ("**[refused]** " if r.answer.refused else "") + r.answer.text
        print(
            f"| {i} | {_cell(question)} | {_cell(answer)} "
            f"| {', '.join(map(str, r.cited_articles)) or '—'} "
            f"| {', '.join(map(str, r.invalid_citations)) or '—'} "
            f"| {retrieved} | {r.latency_ms} | {r.cost_usd:.5f} |"
        )
    print(f"\nTotal cost: ${total_cost:.5f}")


if __name__ == "__main__":
    main()
