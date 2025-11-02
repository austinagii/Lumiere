# Lumiére Transformer Model Card

## Model Overview

**Model Name:** Lumiére
**Model Type:** Autoregressive Language Model  
**Architecture:** Transformer (Decoder Only)
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

- **Vocabulary Size:** 4096 tokens
- **Total Parameters:** Approximately 1180800 parameters

## Training Details

### Dataset

- **Primary Dataset:** WikiText-2 Raw (wikitext-2-raw-v1)

### Tokenization

- **Tokenizer Type:** Byte-Pair Encoding (BPE)
- **Vocabulary Size:** 4096 tokens

## Performance Metrics

### Validation Results

- **Test Loss:** 4.235
- **Test Perplexity:** 71.806

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

## License

MIT License - See project LICENSE file for details.

## Model Card Authors

This model card was generated on Oct 17, 2025 using model checkpoint 'best' from run 'GyefN9L6'.

---

_This model is designed for research and educational purposes, with a focus on interpretability and understanding transformer architectures._

