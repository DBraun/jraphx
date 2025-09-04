# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About JraphX

JraphX is a Graph Neural Network library for JAX/Flax NNX. It serves as an unofficial successor to DeepMind's archived jraph library, providing graph neural network layers and utilities for JAX.

## Development Commands

### Testing
- `make test` - Run all tests with pytest
- `make test-cov` - Run tests with coverage reporting
- `pytest tests/` - Direct pytest command
- `python -m pytest tests/path/to/test.py::test_name -v` - Run specific test

### Code Quality
- `make lint` - Run ruff linting
- `make lint-fix` - Run ruff linting with auto-fix
- `make typecheck` - Run mypy type checking
- `make format` - Format code with black and isort
- `make check` - Run lint + typecheck
- `make check-all` - Run lint + typecheck + tests
- `pre-commit run --all-files` - Run all pre-commit hooks

### Installation
- `make install-dev` - Install package with dev dependencies and pre-commit hooks
- `pip install -e ".[dev]"` - Install in development mode with dev dependencies

## Architecture

### Core Structure
- **src/jraphx/** - Main package with three primary modules:
  - **data/** - Core data structures (`Data`, `Batch`) for graph representation
  - **nn/** - Neural network layers organized by type:
    - `conv/` - Graph convolution layers (GCN, GAT, SAGE, GIN, etc.)
    - `models/` - Higher-level models (BasicGNN, MLP, JumpingKnowledge)
    - `norm/` - Normalization layers (BatchNorm)
    - `pool/` - Pooling operations
  - **utils/** - Utility functions and helpers

### Testing Structure
- **tests/** - Main test suite mirroring src structure
- Tests use pytest and are organized by module hierarchy

### Development Notes
- Package follows strict typing with mypy
- Code formatting with black (100 char line length)
- Linting with ruff
- Pre-commit hooks enforce code quality
- Python 3.11+ required
