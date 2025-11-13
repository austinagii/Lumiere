from collections.abc import Callable

import torch
from torch import nn

from lumiere.research.src.utils import validation

from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """A decoder transformer block.

    This module is modeled after the decoder block from the "Attention Is All
    You Need" paper (https://arxiv.org/pdf/1706.03762). Like the paper, this
    transformer block is made up of two main sub-layers: a masked multi-head
    attention layer, followed by a position-wise feed-forward layer.

    Example:
        >>> import torch
        >>> from lumiere.components import TransformerBlock
        >>> x = torch.randn(1, 10, 128)
        >>> block = TransformerBlock(128, 12, 64, 64, 256)
        >>> output, attention_weights = block(x)
        >>> print(output.shape)
        torch.Size([1, 10, 128])
        >>> print(attention_weights.shape)
        torch.Size([1, 12, 10, 10])

    """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        d_key: int,
        d_value: int,
        feedforward_factory: Callable,
        normalization_factory: Callable,
        dropout: float = 0.1,
        pre_norm: bool = True,
        post_norm: bool = False,
    ) -> None:
        """Initialize a decoder transformer block.

        Args:
            embedding_size: The dimensionality of the token embeddings.
            num_heads: The number of attention heads.
            d_key: The dimensionality of the key vectors.
            d_value: The dimensionality of the value vectors.
            feedforward_factory: A callable that produces feedforwward modules.
            normalization_factory: A callable that produces normalization layers.
            dropout: The dropout probability.
            pre_norm: Whether to apply normalization before attention and
                feed-forward layers. Defaults to True.
            post_norm: Whether to apply normalization after attention and
                feed-forward layers. Defaults to False.

        Raises:
            TypeError: If any of the specified argument values is of the incorrect
                type.
            ValueError: If any of the specified argument values is not one of the
                allowed values for that parameter.

        """
        super().__init__()

        validation.validate_integer(embedding_size, "embedding_size", min_value=1)
        validation.validate_integer(num_heads, "num_heads", min_value=1)
        validation.validate_integer(d_key, "d_key", min_value=1)
        validation.validate_integer(d_value, "d_value", min_value=1)
        validation.validate_probability(dropout, "dropout")
        validation.validate_boolean(pre_norm, "pre_norm")
        validation.validate_boolean(post_norm, "post_norm")

        self._embedding_size = embedding_size
        self._num_heads = num_heads
        self._d_key = d_key
        self._d_value = d_value
        self._dropout = dropout
        self._pre_norm = pre_norm
        self._post_norm = post_norm

        if self._pre_norm:
            self.normalization_1 = normalization_factory()

        self.attention = MultiHeadAttention(
            num_heads=self._num_heads,
            embedding_size=self._embedding_size,
            d_key=self._d_key,
            d_value=self._d_value,
        )

        if self._pre_norm or self._post_norm:
            self.normalization_2 = normalization_factory()

        self.feedforward = feedforward_factory()
        self.dropout = nn.Dropout(self._dropout)

        if self._post_norm:
            self.normalization_3 = normalization_factory()

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass a batch of token embeddings through the transformer block.

        Args:
            x: A batch of token embeddings of shape
                `(batch_size, context_size, embedding_size)`.
            padding_mask: A boolean mask indicating which of the tokens in the
                batch are padding tokens, with `True` indicating the presence of
                a padding token and `False` for non-padding tokens. Expected to
                have the shape: `(batch_size, context_size)`.

        Returns:
            A tuple of output embeddings and attention weights. The output
            embeddings have shape `(batch_size, context_size, embedding_size)` and
            the attention weights have shape
            `(batch_size, num_heads, context_size, context_size)`.

        Raises:
            ValueError: If the specified token embeddings have the incorrect shape.

        """
        if x.dim() != 3:
            raise ValueError(
                "Input tensor must have shape (batch_size, context_size, "
                "embedding_size)"
            )
        if x.size(-1) != self._embedding_size:
            raise ValueError(
                f"Expected embedding_size={self._embedding_size}, got {x.size(-1)}"
            )

        # Compute the attention values and weights.
        if hasattr(self, "normalization_1"):
            x = self.normalization_1(x)

        attention_values, attention_weights = self.attention(
            x, padding_mask=padding_mask
        )
        x = x + self.dropout(attention_values)

        # Compute the feed-forward output.
        if hasattr(self, "normalization_2"):
            x = self.normalization_2(x)

        feedforward_out = self.feedforward(x)
        x = x + self.dropout(feedforward_out)

        if hasattr(self, "normalization_3"):
            x = self.normalization_3(x)

        return x, attention_weights
