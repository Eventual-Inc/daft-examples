.PHONY: setup test test-quickstart test-examples test-no-creds lint format precommit install-hooks

setup: install-hooks
	uv sync --extra test --extra lint
	@test -f .env || cp .env.example .env

install-hooks:
	@mkdir -p .git/hooks
	@echo '#!/bin/sh\nmake precommit' > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit

# ── Lint & Format ──────────────────────────────────────────────────────

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

precommit: lint
	uv run ruff format --check .
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
