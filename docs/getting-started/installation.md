# Installation

## Prerequisites

- Python 3.11 or higher
- pip or pipenv

## Install from Source

Clone the repository and install dependencies:

```bash
git clone https://github.com/austinagii/Lumiere.git
cd Lumiere
pipenv install --dev
```

## Dependencies

Lumière requires the following main dependencies:

- **PyTorch**: Deep learning framework
- **PyYAML**: Configuration file parsing
- **python-dotenv**: Environment variable management

### Optional Dependencies

- **Azure Blob Storage**: For cloud-based checkpoint storage
- **Weights & Biases**: For experiment tracking (coming soon)

## Verify Installation

Test your installation:

```bash
pipenv run python -c "import lumiere; print('Lumière installed successfully!')"
```

## Next Steps

Continue to the [Quick Start](quickstart.md) guide to train your first model.
