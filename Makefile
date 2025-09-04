.PHONY: help install install-dev format lint typecheck test check clean docs

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install the package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

format:  ## Format code with black and isort
	black src/ tests/ --line-length 100
	isort src/ tests/ --profile black --line-length 100

lint:  ## Run linting with ruff
	ruff check src/ tests/

lint-fix:  ## Run linting with ruff and fix issues
	ruff check src/ tests/ --fix


test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=src/jraphx --cov-report=html --cov-report=term

check:  ## Run all checks (lint)
	$(MAKE) lint

check-all:  ## Run all checks and tests
	$(MAKE) check
	$(MAKE) test

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

clean:  ## Clean up build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete

docs:  ## Build documentation
	$(MAKE) -C docs html

docs-clean:  ## Clean documentation build
	$(MAKE) -C docs clean

docs-live:  ## Build documentation with live reload
	sphinx-autobuild docs/source docs/build/html

# Shortcuts
fmt: format  ## Alias for format
ci: check-all  ## Run CI checks (lint, test)
