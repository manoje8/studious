.PHONY: lint, format, run-server

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

run-server:
	python src/api/main.py

test:
	pytest tests/ -v --tb=short
