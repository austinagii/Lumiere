# Lumière

A powerful and flexible deep learning framework for building and training transformer models.

## Features

- **Flexible Model Building**: Component-based architecture with registry system for easy customization
- **Multiple Data Sources**: Support for various datasets with extensible pipeline system
- **Training Infrastructure**: Complete training loop with checkpointing, metrics, and callbacks
- **Tokenization**: Built-in BPE tokenizer with training support
- **Storage Backends**: Multiple storage options including filesystem and Azure Blob Storage
- **Dependency Injection**: Clean configuration management with automatic dependency resolution

## Quick Example

```python
from lumiere.nn.builder import load as load_model
from lumiere.training import Trainer

# Load model from configuration
model = load_model(config["model"])

# Initialize trainer
trainer = Trainer(
    model=model,
    dataloader=dataloader,
    pipeline=pipeline,
    optimizer=optimizer,
    device=device,
)

# Start training
trainer.train()
```

## Getting Started

- [Installation](getting-started/installation.md) - Install Lumière and its dependencies
- [Quick Start](getting-started/quickstart.md) - Your first training run
- [Configuration](user-guide/configuration.md) - Configure your models and training

## Documentation

- [User Guide](user-guide/configuration.md) - Learn how to use Lumière
- [API Reference](reference/) - Detailed API documentation
- [Development](development/contributing.md) - Contributing to Lumière
