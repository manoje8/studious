.PHONY: lint, format, run-server, run-ui

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

run-server:
	python src/api/main.py

run-ui:
	streamlit run web_ui/main.py

test:
	pytest tests/ -v --tb=short
