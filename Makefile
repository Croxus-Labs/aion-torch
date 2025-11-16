.PHONY: install format lint type-check test clean help run

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install package in editable mode with dev dependencies
	pip install -e ".[dev]"

format:  ## Format code with black and ruff
	black src/ tests/ examples/
	ruff format src/ tests/ examples/

lint:  ## Run linting checks
	ruff check src/ tests/ examples/
	black --check src/ tests/ examples/

type-check:  ## Run type checking with mypy
	mypy src/

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=src/aion_torch --cov-report=term-missing

run:  ## Run demo examples
	python examples/demo.py

all: format lint type-check test  ## Run all checks (format, lint, type-check, test)

clean:  ## Clean generated files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

pre-commit-install:  ## Install pre-commit hooks
	pre-commit install
