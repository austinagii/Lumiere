# Lumiére Documentation Standard Definitions

This file contains the canonical definitions for common terms used throughout the Lumiére codebase. **Always use these exact phrasings** to maintain consistency.

## Core Parameters

### Model Architecture

**`vocab_size`**
```
The number of unique tokens in the vocabulary.
```

**`embedding_size`**
```
The dimensionality of the token embeddings.
```

**`context_size`**
```
The maximum number of tokens in a sequence.
```

**`num_layers`**
```
The number of transformer blocks in the network.
```

**`num_heads`**
```
The number of attention heads.
```

### Attention Mechanism

**`d_key`**
```
The dimensionality of the key vectors.
```

**`d_value`**
```
The dimensionality of the value vectors.
```

### Feed-Forward Network

**`d_ff`**
```
The hidden dimension of the feed-forward network.
```

### Regularization

**`dropout`**
```
The dropout probability.
```

### Tokenization & Padding

**`padding_id`**
```
The ID of the padding token.
```

**`padding_mask`**
```
A boolean mask indicating which of the tokens in the batch are padding tokens, with `True` indicating the presence of a padding token and `False` for non-padding tokens. Expected to have the shape: `(batch_size, context_size)`.
```

**`padding_mask` (simplified for helper functions)**
```
A boolean mask indicating which of the tokens are padding tokens, with `True` indicating a padding token and `False` for non-padding tokens.
```

## Common Input/Output Descriptions

### Token Inputs

**Batch of input tokens**
```
A batch of input tokens of shape `(batch_size, context_size)`.
```

**Batch of token IDs**
```
A batch of token IDs of shape `(batch_size, context_size)`.
```

**Batch of token embeddings**
```
A batch of token embeddings of shape `(batch_size, context_size, embedding_size)`.
```

### Embeddings

**Position-encoded token embeddings (output)**
```
A batch of sinusoidal position-encoded token embeddings of shape `(batch_size, context_size, embedding_size)`.
```

**Transformed embeddings (output)**
```
A batch of transformed token embeddings of shape `(batch_size, context_size, embedding_size)`.
```

**Output embeddings (generic)**
```
A batch of output embeddings of shape `(batch_size, context_size, embedding_size)`.
```

### Attention Outputs

**Attention weights**
```
Attention weights of shape `(batch_size, num_heads, context_size, context_size)`.
```

**Attention values**
```
Attention values of shape `(batch_size, context_size, embedding_size)`.
```

**Tuple of embeddings and attention weights**
```
A tuple of output embeddings and attention weights. The output embeddings have shape `(batch_size, context_size, embedding_size)` and the attention weights have shape `(batch_size, num_heads, context_size, context_size)`.
```

**Tuple of attention values and weights**
```
A tuple of attention values and attention weights. The attention values have shape `(batch_size, context_size, embedding_size)` and the attention weights have shape `(batch_size, num_heads, context_size, context_size)`.
```

**Multi-layer attention weights**
```
Attention weights of shape `(batch_size, num_layers, num_heads, context_size, context_size)`.
```

### Model Outputs

**Logits**
```
A batch of logits of shape `(batch_size, context_size, vocab_size)`.
```

**Tuple of logits and multi-layer attention weights**
```
A tuple of logits and attention weights. The logits have shape `(batch_size, context_size, vocab_size)` and the attention weights have shape `(batch_size, num_layers, num_heads, context_size, context_size)`.
```

### Masks

**Causal mask**
```
A mask of shape `(1, 1, context_size, context_size)`.
```

## Usage Guidelines

### For Humans

1. When documenting a parameter, search this file for its definition first
2. Copy the exact wording - don't paraphrase
3. If a parameter isn't listed here, add it following the established patterns
4. When updating a definition, update it everywhere it appears in the codebase

### For AI Assistants

1. Before documenting any parameter, check if it exists in this file
2. Use the exact text provided - do not modify or paraphrase
3. If context requires slight variation (e.g., helper function vs main method), use the appropriate variant provided
4. When adding new definitions:
   - Follow the existing format
   - Use imperative mood for actions, declarative for properties
   - Include shape information in backticks for tensors
   - Test that the definition works in multiple contexts before adding

## Shape Notation Standards

All shapes follow this format:
- Use backticks: `` `(dimension1, dimension2, ...)` ``
- Use descriptive names: `batch_size`, `context_size`, `embedding_size`, `num_heads`
- Use `*` for multiplication: `num_heads * num_dimensions`
- Common dimensions:
  - `batch_size`: Number of sequences in a batch
  - `context_size`: Maximum sequence length
  - `embedding_size`: Dimensionality of embeddings
  - `num_heads`: Number of attention heads
  - `num_dimensions`: Generic dimension size
  - `d_ff`: Hidden dimension in feed-forward network

## Pattern Templates

### New Parameter Definition

**`parameter_name`**: The [description of what this parameter represents].

### New Tensor Description

A batch of [tensor_description] of shape `(dim1, dim2, ...)`.

### New Tuple Return

A tuple of [first_element] and [second_element]. The [first_element] have shape `(...)` and the [second_element] have shape `(...)`.
