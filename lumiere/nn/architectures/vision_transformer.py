"""Provides the `VisionTransformer` model for image classification.

Example:
    ```python
    import torch
    from torch.nn import LayerNorm
    from lumiere.nn.architectures import VisionTransformer
    from lumiere.nn.components.attention import MultiHeadAttention
    from lumiere.nn.components.blocks import StandardTransformerBlock
    from lumiere.nn.components.embedding import PatchEmbedding
    from lumiere.nn.components.feedforward import LinearFeedForward

    model = VisionTransformer(
        num_classes=10,
        embedding_size=64,
        num_blocks=2,
        embedding=lambda: PatchEmbedding(32, 8, 3, 64),
        block=lambda: StandardTransformerBlock(
            attention=lambda: MultiHeadAttention(4, 64, 16, 16, causal=False),
            feedforward=lambda: LinearFeedForward(64, 128),
            normalization=lambda: LayerNorm(64),
        ),
        normalization=lambda: LayerNorm(64),
    )
    logits, attention_weights = model(torch.randn(2, 3, 32, 32))
    print(logits.shape)
    # Output: torch.Size([2, 10])
    print(attention_weights.shape)
    # Output: torch.Size([2, 2, 4, 17, 17])
    ```

"""

from collections.abc import Callable

import torch
from torch import nn

from lumiere.internal.registry import discover
from lumiere.utils import validation


@discover(nn.Module, "model.vision_transformer")
class VisionTransformer(nn.Module):
    """A vision transformer model for image classification.

    This class implements the vision transformer (ViT) as described in the paper `An
    Image is Worth 16x16 Words <https://arxiv.org/abs/2010.11929>`_. Images are
    converted to a sequence of patch embeddings, passed through a stack of
    transformer blocks, and classified using the final embedding of the class token.

    The transformer blocks are expected to use bidirectional attention, e.g. a
    `MultiHeadAttention` module with `causal=False`.

    Attributes:
        num_classes (int): The number of output classes.
        embedding_size (int): The dimensionality of the patch embeddings.
        num_blocks (int): The number of transformer blocks in the model.

    """

    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        num_blocks: int,
        embedding: Callable,
        block: Callable,
        normalization: Callable,
    ):
        """Initialize a vision transformer model.

        Args:
            num_classes: The number of output classes.
            embedding_size: The dimensionality of the patch embeddings.
            num_blocks: The number of transformer blocks in the network.
            embedding: A callable factory that produces patch embedding modules.
            block: A callable factory that produces transformer block modules.
            normalization: A callable factory that produces normalization layers.

        Raises:
            ValueError: If any of the following is not a positive integer:
                - `num_classes`
                - `embedding_size`
                - `num_blocks`

        """
        super().__init__()

        validation.validate_integer(num_classes, "num_classes", min_value=1)
        validation.validate_integer(embedding_size, "embedding_size", min_value=1)
        validation.validate_integer(num_blocks, "num_blocks", min_value=1)

        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.num_blocks = num_blocks
        self.embedding = embedding()
        self.blocks = nn.ModuleList([block() for _ in range(self.num_blocks)])
        self.final_norm = normalization()
        self.head = nn.Linear(embedding_size, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass a batch of images through the vision transformer model.

        Args:
            x: A batch of images of shape
                `(batch_size, num_channels, image_size, image_size)`.

        Returns:
            A tuple of logits and attention weights. The logits have shape
            `(batch_size, num_classes)` and the attention weights have shape
            `(batch_size, num_blocks, num_heads, num_patches + 1, num_patches + 1)`.

        """
        x = self.embedding(x)
        attention_weights = []
        for block in self.blocks:
            x, block_attention_weights = block(x)
            attention_weights.append(block_attention_weights)
        attention_weights = torch.stack(attention_weights, dim=1)

        # Classify using the final embedding of the class token.
        x = self.final_norm(x[:, 0])
        return self.head(x), attention_weights
