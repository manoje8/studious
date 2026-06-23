.PHONY: lint, format, server-run, ui-run

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

server-run:
	python src/api/main.py

ui-run:
	streamlit run web_ui/main.py

test:
	pytest tests/ -v --tb=short
