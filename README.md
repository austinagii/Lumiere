# Lumiére: A machine learning workspace built for hands-on experimentation.

![Lumiére Logo](assets/logo.png)

Lumiére is an extensible deep learning workspace built for rapid, reproducible experimentation. It provides highly configurable, modular implementations of modern neural network components and architectures, alongside wrappers for datasets, tokenizers, and preprocessing pipelines, that compose seamlessly via a declarative YAML specification. A built-in component registry and dependency injection system allow new architectures and data sources to be plugged in with minimal boilerplate. Training runs are launched from a single config file, making ablations as simple as changing a value, while automatic checkpointing and

## Installation

For Linux based systems, the `init.sh` script can be used to quickly initialize the repository, determing whether the required components are installed, and symlinking the main `lumi` script for managing training runs.

Dependencies must still be installed manually, using either the `pip install` or `pipenv install` commands, depending on what's available in your current environment.

```bash
# Using pip
pip install -r requirements.txt

# Or using pipenv
pipenv install
```

## Getting Started

### Training a Model

Training runs can be managed using the `lumi` command (see `lumi --help` for the full list of available options).

Runs can be initiated by specifying a configuration file which defines the training parameters, while existing training runs can be resumed by specifying the training run along with the specific checkpoint to resume training from.

```bash
# Train a new model
lumi train --config-path configs/transformer.yaml

# Resume training from a checkpoint
lumi train --run-id <run_id> --checkpoint-tag best

# Evaluate a trained model
lumi test --run-id <run_id> --checkpoint-tag best
```

## Development

Tests will be executed on commit and push to ensure that changes do not break existing functionality. To execute tests manually, you can execute the following.

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=lumiere --cov-report=html
```

To see documenation, you can spin up the documentation server using `pipenv run docs` or `mkdocs serve --watch lumiere/`.

## Contributions

All contributions are welcome!

## License

[MIT License](LICENSE)

