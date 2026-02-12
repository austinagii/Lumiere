# Contributing

Thank you for your interest in contributing to Lumière!

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/austinagii/Lumiere.git
cd Lumiere
```

2. Install dependencies:

```bash
pipenv install --dev
```

3. Run tests:

```bash
pipenv run pytest
```

## Code Style

We use:

- **Ruff**: For linting and formatting
- **Type hints**: For better code clarity
- **Google-style docstrings**: For documentation

Format your code before committing:

```bash
pipenv run ruff check --fix .
pipenv run ruff format .
```

## Testing

Write tests for new features:

- Place tests in `tests/` matching the source structure
- Use pytest fixtures for common setup
- Aim for high test coverage

Run tests:

```bash
# All tests
pipenv run pytest

# Specific test file
pipenv run pytest tests/nn/test_builder.py

# With coverage
pipenv run pytest --cov=lumiere
```

## Documentation

Update documentation for:

- New features
- API changes
- Configuration options

Build docs locally:

```bash
pipenv run mkdocs serve
```

## Pull Requests

1. Create a feature branch
2. Make your changes
3. Add tests
4. Update documentation
5. Run tests and linting
6. Submit PR with clear description

## Code Organization

- `lumiere/` - Source code
  - `nn/` - Neural network components
  - `data/` - Data loading and processing
  - `training/` - Training infrastructure
  - `persistence/` - Storage backends
  - `tokenizers/` - Tokenization
  - `internal/` - Internal utilities
- `tests/` - Test suite
- `scripts/` - Utility scripts
- `docs/` - Documentation

## Questions?

Open an issue for:

- Bug reports
- Feature requests
- Questions
