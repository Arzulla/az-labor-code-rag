.PHONY: setup data ingest index app smoke test lint eval

LABEL ?=
SPLIT ?= dev

setup:
	uv sync

data:
	uv run python -m labor_code_rag.ingest.download

ingest:
	uv run python -m labor_code_rag.ingest.chunk

index:
	uv run python -m labor_code_rag.ingest.index

app:
	uv run python -m labor_code_rag.app

# 5 hand-picked questions against the real API (~$0.01); not an eval (Phase 3).
smoke:
	uv run python scripts/smoke_run.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

eval:
	@test -n "$(LABEL)" || (echo "usage: make eval LABEL=<name> SPLIT=dev|test"; exit 1)
	@echo "make eval: not implemented yet (Phase 3)"; exit 1
