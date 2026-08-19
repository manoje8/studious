.PHONY: help lint format isort server-run ui-run test eval clean \
        docker-build docker-up docker-down docker-logs docker-restart docker-clean


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
	python -m medici.api.main

ui-run:
	streamlit run web_ui/main.py

test:
	pytest tests/ -v --tb=short

eval:
	python tests/eval/eval_runner.py --exit-code

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete


docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-restart:
	docker compose restart

docker-clean:
	docker compose down -v --remove-orphans
