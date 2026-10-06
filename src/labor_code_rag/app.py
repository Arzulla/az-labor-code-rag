"""Gradio UI (``make app``): one question in, one grounded answer out. Single-turn."""

import logging

import chromadb
import gradio as gr
from chromadb.api.models.Collection import Collection

from labor_code_rag.config import load_settings
from labor_code_rag.errors import LaborCodeRagError, LLMError, RetrievalError
from labor_code_rag.generation.answer import answer_question
from labor_code_rag.ingest.index import collection_name, open_collection
from labor_code_rag.llm import LLMClient, create_openai_client
from labor_code_rag.logging_setup import configure_logging, get_request_id
from labor_code_rag.models import AnswerResult

# Explicit name: under `python -m` __name__ would be "__main__".
logger = logging.getLogger("labor_code_rag.app")

DISCLAIMER = (
    "⚠️ **Bu, hüquqi məsləhət deyil.** Cavablar Əmək Məcəlləsinin mətnindən avtomatik "
    "yaradılır, natamam və ya səhv ola bilər. Qərar vermək üçün hüquqşünasa müraciət edin."
)

EXAMPLES = [
    "Əmək məzuniyyəti minimum neçə gündür?",
    "114-cü maddə nə deyir?",
    "Hamilə qadını işdən çıxarmaq olar?",
]


def format_answer(result: AnswerResult) -> str:
    """Answer text + cited articles as links (+ a warning for invalid citations)."""
    urls = {c.article_no: c.url for c in result.retrieved}
    parts = [result.answer.text]
    if result.cited_articles:
        # e-qanun.az has no per-article anchors: the link opens the law, the label names it.
        links = ", ".join(f"[{ref}]({urls[ref.article]})" for ref in result.cited_articles)
        parts.append(f"**İstinad olunan maddələr:** {links}")
    if result.invalid_citations:
        bad = ", ".join(str(ref) for ref in result.invalid_citations)
        parts.append(f"⚠️ Tapılan maddələr arasında olmayan istinadlar: {bad}")
    return "\n\n".join(parts)


def format_sources(result: AnswerResult) -> str:
    """Retrieved chunks with rank and cosine score, for the collapsed panel."""
    blocks = []
    for c in result.retrieved:
        body = c.text.replace("\n", "\n> ")
        blocks.append(f"**{c.rank}. {c.chunk_id}** · score {c.score:.3f}\n\n> {body}")
    footer = f"`request_id={result.request_id}` · {result.latency_ms} ms · ${result.cost_usd:.5f}"
    return "\n\n".join([*blocks, footer])


def build_app(llm: LLMClient, collection: Collection, k: int) -> gr.Blocks:
    def ask(question: str) -> tuple[str, str]:
        if not question.strip():
            return "Zəhmət olmasa sual yazın.", ""
        try:
            result = answer_question(question, llm, collection, k)
        except LaborCodeRagError as exc:  # already logged as request.failed
            raise gr.Error(f"Xəta baş verdi, yenidən cəhd edin (id: {get_request_id()}).") from exc
        return format_answer(result), format_sources(result)

    app = gr.Blocks(title="Əmək Məcəlləsi köməkçisi")
    with app:
        gr.Markdown(
            "# Əmək Məcəlləsi köməkçisi\nAzərbaycan Respublikasının Əmək Məcəlləsi üzrə sual verin."
        )
        gr.Markdown(DISCLAIMER)
        question = gr.Textbox(label="Sual", lines=2, placeholder="Məs.: " + EXAMPLES[0])
        button = gr.Button("Soruş", variant="primary")
        answer = gr.Markdown()
        with gr.Accordion("Tapılan maddələr və score-lar", open=False):
            sources = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=question)
        button.click(ask, inputs=question, outputs=[answer, sources])
        question.submit(ask, inputs=question, outputs=[answer, sources])
    return app


def main() -> None:
    settings = load_settings()
    configure_logging(
        settings.logging.level, settings.logging.format, settings.logging.log_payloads
    )
    if settings.llm is None:
        raise LLMError("missing 'llm' section in config.yaml")
    # Clients are created once here and passed down (no module-level clients).
    llm = LLMClient(create_openai_client(settings.llm, settings.openai_api_key), settings.llm)
    chroma = chromadb.PersistentClient(path=str(settings.index.chroma_dir))
    name = collection_name(settings.index.collection_prefix, settings.llm.embedding_model)
    collection = open_collection(chroma, name)
    if collection.count() == 0:
        raise RetrievalError(f"collection {name} is empty; run `make index` first")
    build_app(llm, collection, settings.retrieval.k).launch()


if __name__ == "__main__":
    main()
