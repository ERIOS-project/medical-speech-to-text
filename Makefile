RUFF ?= env UV_PROJECT_ENVIRONMENT=.venv uv run --extra train ruff

.PHONY: lint format

lint:
	$(RUFF) check src tests
	$(RUFF) format --check src tests

format:
	$(RUFF) check --fix src tests
	$(RUFF) format src tests
