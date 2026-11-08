# Model Building

Lumière provides a flexible component-based system for building transformer models.

## Component Registry

Models are built from registered components:

- **Architectures**: `transformer`, `vision_transformer`
- **Embeddings**: `sinusoidal`, `patch`
- **Attention**: `multihead`
- **Feedforward**: `linear`, `swiglu`
- **Blocks**: `standard`
- **Normalization**: `rms`, `layer`

Each component is selected with its `type` key. Scalar values defined on a component
are inherited by all of its nested components, so shared values such as
`embedding_size` only need to be defined once.

## Configuration-Based Building

Define models in the `architecture` section of a training configuration:

```yaml
architecture:
  type: transformer
  vocab_size: 4096
  context_size: 64
  embedding_size: 128
  num_blocks: 4
  embedding:
    type: sinusoidal
    padding_id: 0
  block:
    type: standard
    attention:
      type: multihead
      num_heads: 4
      d_key: 32
      d_value: 32
    feedforward:
      type: linear
      d_ff: 256
    normalization:
      type: rms
      normalized_shape: 128
  normalization:
    type: rms
    normalized_shape: 128
```

## Vision Transformer

The `vision_transformer` architecture implements the vision transformer (ViT) from
[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) for image
classification. It reuses the standard transformer components, replacing the token
embedding with a `patch` embedding and adding a classification head.

- **`patch` embedding**: Splits each image into non-overlapping
  `patch_size × patch_size` patches, projects each patch to `embedding_size`,
  prepends a learnable class token and adds positional embeddings. Set
  `learned_positions: false` to use fixed sinusoidal encodings instead.
- **Bidirectional attention**: Set `causal: false` on the `multihead` attention so
  every patch can attend to every other patch.
- **Classification head**: The final embedding of the class token is normalized and
  projected to `num_classes` logits.

```yaml
architecture:
  type: vision_transformer
  num_classes: 10
  embedding_size: 192
  num_blocks: 6
  embedding:
    type: patch
    image_size: 32
    patch_size: 4
    num_channels: 3
    dropout: 0.0
  block:
    type: standard
    attention:
      type: multihead
      num_heads: 3
      d_key: 64
      d_value: 64
      causal: false
    feedforward:
      type: linear
      d_ff: 768
    normalization:
      type: layer
      normalized_shape: 192
    dropout: 0.1
    pre_norm: true
  normalization:
    type: layer
    normalized_shape: 192
```

To train a vision transformer, pair it with an image dataset and the `image`
pipeline (see [Data Pipeline](data-pipeline.md)). A complete configuration is
available in
[`examples/vit-cifar10.yaml`](https://github.com/austinagii/Lumiere/blob/main/examples/vit-cifar10.yaml).

The model takes a batch of images of shape
`(batch_size, num_channels, image_size, image_size)` and returns a tuple of logits of
shape `(batch_size, num_classes)` and attention weights of shape
`(batch_size, num_blocks, num_heads, num_patches + 1, num_patches + 1)`.

## Programmatic Building

Build models in code from a specification:

```python
from lumiere.nn import ModelBuilder

spec = {
    "type": "transformer",
    "vocab_size": 4096,
    "context_size": 64,
    "embedding_size": 128,
    "embedding": {
        "type": "sinusoidal",
    },
    # ... more configuration
}

model = ModelBuilder.build(spec)
```

## Custom Components

Register custom components with the `@discover` decorator, using the
`<component>.<type>` naming convention:

```python
from torch import nn

from lumiere.internal.registry import discover

@discover(nn.Module, "feedforward.custom")
class CustomFeedForward(nn.Module):
    """Custom feedforward implementation."""
    pass
```

## See Also

- [Configuration](configuration.md)
- [API Reference](../reference/)
