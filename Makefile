.PHONY: setup prepare serve test eval lint

setup:
	uv sync

prepare:
	uv run churn-mcp prepare

serve:
	uv run churn-mcp serve

test:
	uv run pytest -q

eval:
	uv run python evals/run_evals.py

lint:
	uv run ruff check src tests evals
	uv run ruff format --check src tests evals
	uv run mypy src
