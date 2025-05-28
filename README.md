# Lumiére: Transformer-Based Multimodal Learning
![Lumiére Logo](assets/logo.png)

Lumiére is a multimodal transformer built for interpretability research and easy experimentation. 

The project aims to provide a lightweight canvas for exploring and interpreting the internals of transformer models 
by It provides a beginner friendly, modular implementation of a modern(ish) multimodal transformer architecture
along with many of the common building blocks. 

## Features
- **PyTorch-Based**: Built on top of PyTorch for efficient tensor operations and GPU acceleration
- **Modular Architecture**: Each component is implemented as a separate module for clarity and reusability

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