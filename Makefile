.PHONY: setup setup-egodex test test-quickstart test-examples test-no-creds lint format precommit install-hooks strip-notebooks

# Example notebooks are committed source-only (no baked-in run outputs). Scoped
# to datasets/ so conference/demo notebooks under notebooks/ keep their outputs.
DATASET_NOTEBOOKS := $(shell find datasets -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*')

setup: install-hooks
	uv sync --extra test --extra lint
	@test -f .env || cp .env.example .env

setup-egodex: install-hooks
	uv sync --extra egodex --extra notebook
	@test -f .env || cp .env.example .env

install-hooks:
	@mkdir -p .git/hooks
	@echo '#!/bin/sh\nmake precommit' > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit

# ── Lint & Format ──────────────────────────────────────────────────────

lint:
	uv run --extra lint ruff check .

format:
	uv run --extra lint ruff format .
	uv run --extra lint ruff check --fix .

strip-notebooks:
	uv run --extra lint nbstripout $(DATASET_NOTEBOOKS)

precommit: lint
	uv run --extra lint ruff format --check .
	uv run --extra lint nbstripout --verify $(DATASET_NOTEBOOKS)
	@echo "All checks passed."

# ── Tests ──────────────────────────────────────────────────────────────

test:
	uv run -m pytest tests -q -n auto

test-quickstart:
	uv run -m pytest tests/test_examples.py -q -n auto -k quickstart

test-examples:
	uv run -m pytest tests/test_examples.py -q -n auto -k example

test-no-creds:
	uv run -m pytest tests/test_examples.py -q -n auto -m "not credentials"
