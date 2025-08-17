# Lumiére: Text-Based Transformer for Interpretability
![Lumiére Logo](assets/logo.png)

Lumiére is a text-based transformer built for interpretability research and easy experimentation. 

The project aims to provide a lightweight canvas for exploring and interpreting the internals of transformer models. It provides a beginner friendly, modular implementation of a modern transformer architecture along with many of the common building blocks.

**⚠️ Under Active Development**: This project is currently under active development. APIs and interfaces may change.

## Features
- **Text-Only Focus**: Streamlined for text-based language modeling without multimodal complexity
- **PyTorch-Based**: Built on top of PyTorch for efficient tensor operations and GPU acceleration  
- **Modular Architecture**: Each component is implemented as a separate module for clarity and reusability
- **Interpretability-First**: Returns attention weights and intermediate representations for analysis
- **Modern Architecture**: Implements RMSNorm, SwiGLU, and other contemporary improvements
- **Comprehensive Tooling**: Complete training pipeline with checkpointing, evaluation, and monitoring

## Installation

```bash
# Using pip
pip install -r requirements.txt

# Or using pipenv
pipenv install
```

## Getting Started

### Training a Model

```bash
# Train a new model with default settings
python scripts/train.py

# Resume training from a checkpoint
python scripts/train.py --run-id <run_id> --checkpoint-name best.pth

# Train without logging to wandb
python scripts/train.py --disable-wandb-logging
```

The model will be trained on WikiText-2 dataset and checkpoints will be saved to `artifacts/checkpoints/`.

## Architecture

The transformer implementation includes:

- **Transformer Blocks**: Multi-head attention with RMSNorm and SwiGLU feed-forward networks
- **Embedding Layer**: Token and positional embeddings
- **BPE Tokenizer**: Byte-pair encoding for text preprocessing
- **Attention Visualization**: Full attention weight extraction for interpretability analysis

## Model Configuration

Models are configured via `configs/transformer.yaml`. Key parameters:
- `embedding_size`: Dimensionality of token embeddings (default: 128)
- `context_size`: Maximum sequence length (default: 64)  
- `num_layers`: Number of transformer blocks (default: 4)
- `num_heads`: Number of attention heads per block (default: 4)

## Development

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=lumiere --cov-report=html
```

## License

[MIT License](LICENSE)