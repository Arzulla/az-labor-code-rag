.PHONY: setup data ingest app test lint eval

LABEL ?=
SPLIT ?= dev

setup:
	uv sync

data:
	uv run python -m labor_code_rag.ingest.download

ingest:
	uv run python -m labor_code_rag.ingest.chunk

app:
	@echo "make app: not implemented yet (Phase 2)"; exit 1

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

eval:
	@test -n "$(LABEL)" || (echo "usage: make eval LABEL=<name> SPLIT=dev|test"; exit 1)
	@echo "make eval: not implemented yet (Phase 3)"; exit 1
