.PHONY: help lint format isort server-run ui-run test clean


help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

isort:
	ruff check --select I --fix .

server-run:
	python src/api/main.py

ui-run:
	streamlit run web_ui/main.py

test:
	pytest tests/ -v --tb=short

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
