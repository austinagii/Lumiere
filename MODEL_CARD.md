# Lumiére Transformer Model Card

## Model Overview

**Model Name:** Lumiére Transformer  
**Model Type:** Autoregressive Language Model  
**Architecture:** Transformer (GPT-style)  
**Framework:** PyTorch  
**Purpose:** Text generation and language modeling with interpretability focus

## Model Architecture

### Core Architecture
- **Base Architecture:** Transformer decoder with modern improvements
- **Layers:** 4 transformer blocks
- **Attention Heads:** 4 per layer
- **Embedding Dimension:** 128
- **Context Length:** 64 tokens
- **Feed-forward Dimension:** 256
- **Key/Value Dimensions:** 32 each
- **Dropout:** 0.1

### Modern Improvements
- **Normalization:** RMSNorm (pre-normalization) instead of LayerNorm
- **Activation:** SwiGLU in feed-forward networks instead of ReLU
- **Interpretability:** Returns attention weights and intermediate representations

### Parameter Count
- **Vocabulary Size:** 4,096 tokens
- **Total Parameters:** Approximately 1M parameters

## Training Details

### Dataset
- **Primary Dataset:** WikiText-2 Raw (wikitext-2-raw-v1)
- **Sliding Window Size:** 8 tokens

### Tokenization
- **Tokenizer Type:** Byte-Pair Encoding (BPE)
- **Vocabulary Size:** 4,096 tokens

### Training Configuration
- **Batch Size:** 32
- **Learning Rate:** 3e-4
- **Weight Decay:** 0.01
- **Gradient Clipping:** 1.0
- **Warmup Steps:** 500
- **Max Epochs:** 250
- **Early Stopping Patience:** 5 epochs
- **Stopping Threshold:** 0.0001


## Performance Metrics

### Validation Results
- **Average Loss:** 4.4180
- **Average Perplexity:** 85.01
- **Evaluation Batches:** 178


## Model Capabilities

### Strengths
- **Lightweight Design:** Small parameter count suitable for research and experimentation
- **Interpretability Focus:** Provides attention weights for analysis
- **Modern Architecture:** Incorporates contemporary improvements (RMSNorm, SwiGLU)
- **Complete Pipeline:** Includes training, evaluation, and inference tools

### Limitations
- **Small Scale:** Limited capacity due to small parameter count
- **Context Length:** Short 64-token context window
- **Single Language:** Trained only on English text
- **Domain:** Limited to WikiText-2 domain knowledge

## Technical Implementation

### Key Components
- **Attention Mechanism:** Multi-head self-attention with causal masking
- **Position Encoding:** Learned positional embeddings
- **Padding Handling:** Special padding token (ID managed by tokenizer)
- **Output Layer:** Linear projection to vocabulary logits

### Code Structure
```
lumiere/
├── components/         # Individual model components
├── models/             # Complete transformer implementation  
├── data/               # Data loading and preprocessing
├── training/           # Training and evaluation loops
└── persistence/        # Model checkpointing and management
```

## Usage Examples

### Loading the Model
```python
from lumiere.models.transformer import Transformer
from lumiere.persistence.checkpoint_manager import CheckpointManager

# Load model from checkpoint
checkpoint = checkpoint_manager.load_checkpoint("run_hiq6ihsb_20250819_001808", "best")
model = Transformer(vocab_size=4096, embedding_size=128, context_size=64, 
                   num_layers=4, num_heads=4, d_key=32, d_value=32, d_ff=256)
model.load_state_dict(checkpoint["model_state_dict"])
```

### Evaluation
```bash
# Evaluate model performance
python scripts/eval.py --run-id hiq6ihsb --checkpoint-name best
```

## License

MIT License - See project LICENSE file for details.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{lumiere-transformer,
  title={Lumiére: Text-Based Transformer for Interpretability},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/lumiere}
}
```

## Model Card Authors

This model card was generated on August 19, 2025.

---

*This model is designed for research and educational purposes, with a focus on interpretability and understanding transformer architectures.*