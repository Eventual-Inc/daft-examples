.PHONY: setup test test-quickstart test-examples test-no-creds lint format check

setup:
	uv sync --extra test --extra lint
	@test -f .env || cp .env.example .env

# ── Lint & Format ──────────────────────────────────────────────────────

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

check: lint
	uv run ruff format --check .
	@echo "All checks passed."

# ── Tests ──────────────────────────────────────────────────────────────

# Run all tests (scripts without credentials are auto-skipped)
test:
	uv run -m pytest tests -q

# Run only quickstart tier
test-quickstart:
	uv run -m pytest tests/test_examples.py -q -k quickstart

# Run only example tier
test-examples:
	uv run -m pytest tests/test_examples.py -q -k example

# Run only scripts that need no credentials (good for CI without secrets)
test-no-creds:
	uv run -m pytest tests/test_examples.py -q -m "not credentials"
