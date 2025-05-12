# Prism: A Transformer Implementation
![Prism Logo](assets/logo.png)

Prism is a simple generative transformer based on the original Transformer architecture described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. It provides a reference implementation designed to be used for experimenting with language models and their internals.

This is intended as a 
## Features

- **Modular Architecture**: Each component is implemented as a separate module for clarity and reusability
- **PyTorch-Based**: Built on top of PyTorch for efficient tensor operations and GPU acceleration
- **Byte-Pair Encoding Tokenization**: Uses the Hugging Face tokenizers library for efficient text tokenization
- **Data Processing Utilities**: Includes utilities for batch processing of datasets
- **Positional Encoding**: Implementation of sinusoidal positional encoding
- **Multi-Head Attention**: Core attention mechanism of the Transformer architecture

## Installation

```bash
# Using pip
pip install -r requirements.txt

# Or using pipenv
pipenv install
```

## Getting Started

```python
from prism.tokenizer import Tokenizer
from prism.model import Model
from prism.data import to_batches

# Initialize a tokenizer and train it on your dataset
tokenizer = Tokenizer().train(dataset, "text", batch_size=64, vocab_size=16384)

# Create a transformer model
model = Model(
    vocab_size=16384, 
    embedding_size=256, 
    context_size=512,
    num_heads=12
)

# Process data in batches
for batch in to_batches(tokenizer, dataset, batch_size=64, context_size=512):
    output = model(batch)
    # Your training code here
```

## Model Components

- **Embedding**: Token embedding layer
- **PositionalEncoding**: Adds position information to token embeddings
- **MultiHeadAttention**: Performs the self-attention mechanism
- **Tokenizer**: BPE tokenization for text processing
- **Data Utilities**: Tools for dataset batching and processing

## Development

```bash
# Run tests
pytest tests/
```

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:

```
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```