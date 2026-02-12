# Testing

Lumière uses pytest for testing.

## Running Tests

```bash
# Run all tests
pipenv run pytest

# Run specific test file
pipenv run pytest tests/nn/test_builder.py

# Run specific test
pipenv run pytest tests/nn/test_builder.py::TestModelSpec::test_from_yaml

# With coverage
pipenv run pytest --cov=lumiere

# Parallel execution
pipenv run pytest -n auto
```

## Test Organization

Tests mirror the source structure:

```
lumiere/
  nn/
    builder.py
tests/
  nn/
    test_builder.py
```

## Test Markers

Available markers:

- `@pytest.mark.slow` - Slow tests
- `@pytest.mark.integration` - Integration tests

Run specific markers:

```bash
# Skip slow tests
pipenv run pytest -m "not slow"

# Only integration tests
pipenv run pytest -m integration
```

## Writing Tests

### Test Structure

```python
class TestMyComponent:
    def test_feature_works(self):
        # Arrange
        component = MyComponent(config)

        # Act
        result = component.process()

        # Assert
        assert result == expected
```

### Fixtures

Use fixtures for common setup:

```python
@pytest.fixture
def model():
    return Transformer(
        vocab_size=1000,
        context_size=64,
        # ...
    )

def test_forward_pass(model):
    output = model(input_ids)
    assert output.shape == expected_shape
```

### Parametrize

Test multiple inputs:

```python
@pytest.mark.parametrize("vocab_size", [100, 1000, 10000])
@pytest.mark.parametrize("context_size", [64, 128, 256])
def test_model_shapes(vocab_size, context_size):
    model = create_model(vocab_size, context_size)
    # ...
```

## Coverage

Maintain high test coverage:

```bash
# Generate coverage report
pipenv run pytest --cov=lumiere --cov-report=html

# View report
open htmlcov/index.html
```

## Pre-commit Hooks

Tests run automatically on commit:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: pytest
      name: Run tests with coverage
      entry: pipenv run pytest --cov=lumiere
      language: system
      pass_filenames: false
```

## See Also

- [Contributing](contributing.md)
- [Documentation](documentation.md)
