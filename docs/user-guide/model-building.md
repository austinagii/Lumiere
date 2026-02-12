# Model Building

Lumière provides a flexible component-based system for building transformer models.

## Component Registry

Models are built from registered components:

- **Embeddings**: `sinusoidal`, etc.
- **Attention**: `multihead`, etc.
- **Feedforward**: `linear`, `swiglu`, etc.
- **Blocks**: `standard`, etc.
- **Normalization**: `rms`, `layer`, etc.

## Configuration-Based Building

Define models in YAML:

```yaml
model:
  vocab_size: 4096
  context_size: 64
  embedding_size: 128
  num_blocks: 4
  embedding:
    type: embedding
    name: sinusoidal
    padding_id: 0
  block:
    type: block
    name: standard
    attention:
      type: attention
      name: multihead
      num_heads: 4
      d_key: 32
      d_value: 32
    feedforward:
      type: feedforward
      name: linear
      d_ff: 256
    normalization:
      type: normalization
      name: rms
```

## Programmatic Building

Build models in code:

```python
from lumiere.nn.builder import load as load_model

config = {
    "vocab_size": 4096,
    "context_size": 64,
    "embedding": {
        "type": "embedding",
        "name": "sinusoidal",
    },
    # ... more configuration
}

model = load_model(config)
```

## Custom Components

Register custom components with the `@discover` decorator:

```python
from lumiere.internal.registry import discover

@discover("feedforward", "custom")
class CustomFeedForward(nn.Module):
    """Custom feedforward implementation."""
    pass
```

## See Also

- [Configuration](configuration.md)
- [API Reference](../reference/)
