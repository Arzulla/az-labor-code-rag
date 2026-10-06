.PHONY: setup data ingest index app smoke test lint eval eval-retrieval compare baseline calibrate

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

# Golden-set eval (Phase 3): result JSON in eval/results/, summary table on stdout.
eval:
	@test -n "$(LABEL)" || (echo "usage: make eval LABEL=<name> SPLIT=dev|test"; exit 1)
	uv run python -m eval.run_eval --label $(LABEL) --split $(SPLIT)

# Retrieval metrics only: no answer or judge calls (~$0.0001 per run).
eval-retrieval:
	@test -n "$(LABEL)" || (echo "usage: make eval-retrieval LABEL=<name> SPLIT=dev|test"; exit 1)
	uv run python -m eval.run_eval --label $(LABEL) --split $(SPLIT) --retrieval-only

compare:
	@test -n "$(A)" -a -n "$(B)" || (echo "usage: make compare A=<result.json> B=<result.json>"; exit 1)
	uv run python -m eval.compare $(A) $(B)

# Dev twice + test once, then README table + eval/calibration.jsonl (~$0.40).
baseline:
	uv run python -m eval.run_eval --label baseline --split dev
	uv run python -m eval.run_eval --label baseline --split dev
	uv run python -m eval.run_eval --label baseline --split test
	uv run python -m eval.report

calibrate:
	uv run python -m eval.calibrate
