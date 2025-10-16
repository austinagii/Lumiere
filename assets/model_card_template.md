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
- **Layers:** {num_layers} transformer blocks
- **Attention Heads:** {num_attention_heads} per layer
- **Embedding Dimension:** {embedding_size}
- **Context Length:** {context_size} tokens
- **Feed-forward Dimension:** {d_ff}
- **Key/Value Dimensions:** {d_key} each
- **Dropout:** {dropout}

### Modern Improvements

- **Normalization:** RMSNorm (pre-normalization) instead of LayerNorm
- **Activation:** SwiGLU in feed-forward networks instead of ReLU
- **Interpretability:** Returns attention weights and intermediate representations

### Parameter Count

- **Vocabulary Size:** {vocab_size} tokens
- **Total Parameters:** Approximately {num_params} parameters

## Training Details

### Dataset

- **Primary Dataset:** WikiText-2 Raw (wikitext-2-raw-v1)

### Tokenization

- **Tokenizer Type:** Byte-Pair Encoding (BPE)
- **Vocabulary Size:** {vocab_size} tokens

## Performance Metrics

### Validation Results

- **Test Loss:** {loss:.3f}
- **Test Perplexity:** {perplexity:.3f}

## Model Capabilities

### Strengths

- **Lightweight Design:** Small parameter count suitable for research and experimentation
- **Interpretability Focus:** Provides attention weights for analysis
- **Modern Architecture:** Incorporates contemporary improvements (RMSNorm, SwiGLU)
- **Complete Pipeline:** Includes training, evaluation, and inference tools

### Limitations

- **Small Scale:** Limited capacity due to small parameter count
- **Context Length:** Short {context_size}-token context window
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

This model card was generated on {generated_date} using model checkpoint '{checkpoint_tag}' from run '{run_id}'.

---

_This model is designed for research and educational purposes, with a focus on interpretability and understanding transformer architectures._

