# Documentation

Lumière uses MkDocs with Material theme and Google-style docstrings.

## Building Documentation

### Install Dependencies

```bash
pipenv install mkdocs mkdocs-material mkdocstrings[python] mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index
```

### Serve Locally

```bash
pipenv run mkdocs serve
```

Visit [http://localhost:8000](http://localhost:8000)

### Build Static Site

```bash
pipenv run mkdocs build
```

Output in `site/` directory.

## Writing Documentation

### Page Structure

Create markdown files in `docs/`:

```
docs/
  index.md
  getting-started/
    installation.md
    quickstart.md
  user-guide/
    configuration.md
    training.md
  development/
    contributing.md
```

### Google-Style Docstrings

Use Google-style docstrings in Python code:

```python
def train_model(config: dict, device: str) -> Trainer:
    """Train a transformer model with the given configuration.

    This function initializes all training components, creates a new
    training run, and executes the training loop with automatic
    checkpointing.

    Args:
        config: Training configuration dictionary containing model,
            data, optimizer, and training parameters.
        device: Device to use for training ('cpu', 'cuda', 'mps').

    Returns:
        Configured Trainer instance ready for training.

    Raises:
        ValueError: If configuration is invalid.
        RuntimeError: If training initialization fails.

    Example:
        >>> config = load_config("config.yaml")
        >>> trainer = train_model(config, device="cuda")
        >>> metrics = trainer.train()
    """
    pass
```

### Code Blocks

Use fenced code blocks with syntax highlighting:

````markdown
```python
from lumiere.nn.builder import load

model = load(config)
```
````

### Admonitions

Highlight important information:

```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.

!!! tip
    This is a helpful tip.
```

### Links

Link to other pages:

```markdown
See [Configuration](../user-guide/configuration.md) for details.
```

## API Reference

API docs are auto-generated from docstrings:

1. Write Google-style docstrings in code
2. `docs/gen_ref_pages.py` generates reference pages
3. Access at `/reference/` in docs

## Navigation

Update `mkdocs.yml` to add pages to navigation:

```yaml
nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
  - User Guide:
    - Configuration: user-guide/configuration.md
```

## See Also

- [Contributing](contributing.md)
- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
