"""Provides the `PatchEmbedding` module for converting images to patch embeddings.

Example:
    ```python
    import torch
    from lumiere.nn.components.embedding import PatchEmbedding

    x = torch.randn(2, 3, 32, 32)
    embedding = PatchEmbedding(image_size=32, patch_size=8, num_channels=3,
                               embedding_size=64)
    output = embedding(x)
    print(output.shape)
    # Output: torch.Size([2, 17, 64])
    ```

"""

import torch
from torch import nn

from lumiere.internal.registry import discover
from lumiere.utils import validation

from .sinusoidal import sinusoidal_positional_encoding


@discover(nn.Module, "embedding.patch")
class PatchEmbedding(nn.Module):
    """Converts images to position-encoded patch embeddings.

    This class implements the patch embedding as described in the paper `An Image is
    Worth 16x16 Words <https://arxiv.org/abs/2010.11929>`_. Each image is split into
    non-overlapping square patches, which are linearly projected to the embedding
    dimension. A learnable class token is prepended to the sequence of patch
    embeddings, and positional encodings are added to every position.

    Attributes:
        image_size (int): The height and width of the input images in pixels.
        patch_size (int): The height and width of each image patch in pixels.
        num_channels (int): The number of channels in the input images.
        embedding_size (int): The dimensionality of the patch embeddings.
        num_patches (int): The number of patches extracted from each image.

    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_channels: int,
        embedding_size: int,
        dropout: float = 0.0,
        learned_positions: bool = True,
    ) -> None:
        """Initialize a patch embedding layer.

        Args:
            image_size: The height and width of the input images in pixels.
            patch_size: The height and width of each image patch in pixels.
            num_channels: The number of channels in the input images.
            embedding_size: The dimensionality of the patch embeddings.
            dropout: The dropout probability. Defaults to 0.0.
            learned_positions: Whether to use learnable positional embeddings. If
                `False`, fixed sinusoidal positional encodings are used instead.
                Defaults to True.

        Raises:
            ValueError: If any of the following is not a positive integer:
                - `image_size`
                - `patch_size`
                - `num_channels`
                - `embedding_size`
            ValueError: If `image_size` is not divisible by `patch_size`.
            ValueError: If `learned_positions` is `False` and `embedding_size` is
                not even.

        """
        super().__init__()

        validation.validate_integer(image_size, "image_size", min_value=1)
        validation.validate_integer(patch_size, "patch_size", min_value=1)
        validation.validate_integer(num_channels, "num_channels", min_value=1)
        validation.validate_integer(embedding_size, "embedding_size", min_value=1)
        validation.validate_probability(dropout, "dropout")
        validation.validate_boolean(learned_positions, "learned_positions")

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.num_patches = (image_size // patch_size) ** 2

        # A convolution with kernel size and stride equal to the patch size is
        # equivalent to splitting the image into patches and applying a shared
        # linear projection to each flattened patch.
        self._projection = nn.Conv2d(
            num_channels, embedding_size, kernel_size=patch_size, stride=patch_size
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_size))

        num_positions = self.num_patches + 1
        if learned_positions:
            self.positional_embedding = nn.Parameter(
                torch.zeros(1, num_positions, embedding_size)
            )
            nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        else:
            self.register_buffer(
                "positional_embedding",
                sinusoidal_positional_encoding(num_positions, embedding_size).unsqueeze(
                    0
                ),
            )

        nn.init.trunc_normal_(self.class_token, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a batch of images to position-encoded patch embeddings.

        Args:
            x: A batch of images of shape
                `(batch_size, num_channels, image_size, image_size)`.

        Returns:
            A batch of position-encoded patch embeddings of shape
            `(batch_size, num_patches + 1, embedding_size)`, where the first
            embedding in each sequence is the class token.

        Raises:
            ValueError: If the specified images have the incorrect shape.

        """
        expected_shape = (self.num_channels, self.image_size, self.image_size)
        if x.dim() != 4 or tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                "The input tensor must have the shape (batch_size, "
                f"{self.num_channels}, {self.image_size}, {self.image_size}), "
                f"got {tuple(x.shape)}"
            )

        # (batch_size, embedding_size, h, w) -> (batch_size, num_patches,
        # embedding_size)
        patches = self._projection(x).flatten(2).transpose(1, 2)

        class_tokens = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_tokens, patches], dim=1)
        x = x + self.positional_embedding

        return self.dropout(x)
