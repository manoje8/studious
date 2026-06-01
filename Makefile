.PHONY: lint, format, run-server

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

run-server:
	uvicorn src.api.main:app --reload
