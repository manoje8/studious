.PHONY: lint, format

lint:
	ruff check .
	black --check .

 format:
	ruff check --fix .
	black .
