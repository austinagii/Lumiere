import torch
from torch import nn

from lumiere.research.src.utils import validation

from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Applies a transformer block over a batch of token embeddings.

    This module is modeled after the decoder block from the "Attention Is All You Need"
    paper (https://arxiv.org/pdf/1706.03762). Like the paper, this transformer block is
    made up of two main sub-layers: a masked multi-head attention layer, followed by a
    position-wise feed forward layer. It also provides various parameters to slightly
    tweak the architecture of the block: enabling/disabling dropout, specifying pre/post
    normalization, swapping the normalization algorithm and more.

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
        d_ff: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
        post_norm: bool = False,
        norm_type: str = "rms",
    ) -> None:
        """Initialize a new transformer block.

        Args:
            embedding_size (int): The expected dimensionality of the token embeddings.
            num_heads (int): The number of attention heads in the multi-head attention
                layer.
            d_key (int): The dimensionality of the key vectors in the multi-head
                attention layer.
            d_value (int): The dimensionality of the value vectors in the multi-head
                attention layer.
            d_ff (int): The dimensionality of the feed-forward network's hidden
                representations.
            dropout (float): The dropout probability. Setting this value to 0 disables
                dropout. If enabled (any value in the range (0, 1]), dropout will be
                configured just before the residual connection for both sub-layers.
                Defaults to 0.1.
            pre_norm: Whether inputs flowing into both sub-layers should be normalized.
                Defaults to True.
            post_norm: Whether outputs flowing out of both sub-layers should be
                normalized. Defaults to False.
            norm_type: The normalization algorithm to use. Possible values are:
                'rms' and 'layer'. Defaults to 'rms'.

        Raises:
            TypeError: If any of the specified argument values is of the incorrect type.
            ValueEror: If any of the specified argument values is not one of the
                allowed values for that parameter.
        """
        super().__init__()

        validation.validate_integer(embedding_size, "embedding_size", min_value=1)
        validation.validate_integer(num_heads, "num_heads", min_value=1)
        validation.validate_integer(d_key, "d_key", min_value=1)
        validation.validate_integer(d_value, "d_value", min_value=1)
        validation.validate_integer(d_ff, "d_ff", min_value=1)
        validation.validate_probability(dropout, "dropout")
        validation.validate_boolean(pre_norm, "pre_norm")
        validation.validate_boolean(post_norm, "post_norm")
        validation.validate_string(norm_type, "norm_type")

        self._embedding_size = embedding_size
        self._num_heads = num_heads
        self._d_key = d_key
        self._d_value = d_value
        self._d_ff = d_ff
        self._dropout = dropout
        self._pre_norm = pre_norm
        self._post_norm = post_norm
        self._norm_type = norm_type
        self._norm_type = norm_type.strip().lower()

        if self._pre_norm:
            self.normalization_1 = self._create_norm_layer()

        self.attention = MultiHeadAttention(
            num_heads=self._num_heads,
            embedding_size=self._embedding_size,
            d_key=self._d_key,
            d_value=self._d_value,
        )

        if self._pre_norm or self._post_norm:
            self.normalization_2 = self._create_norm_layer()

        self.feedforward = FeedForward(
            self._embedding_size, self._d_ff, dropout=self._dropout
        )
        self.dropout = nn.Dropout(self._dropout)

        if self._post_norm:
            self.normalization_3 = self._create_norm_layer()

    def _create_norm_layer(self) -> nn.Module:
        """Create a normalization layer using the current block's configuration."""
        match self._norm_type:
            case "rms":
                return nn.RMSNorm(self._embedding_size)
            case "layer":
                return nn.LayerNorm(self._embedding_size)
            case _:
                raise ValueError(f"Invalid normalization type: '{self._norm_type}'")

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the transformer block to the specified input tensor.

        Args:
            x: A batch of token embeddings. Expcted to have the shape:
                `(batch_size, context_size, embedding_size)`.
            padding_mask: A boolean mask indicating which of the tokens in the batch
                are padding tokens, with `True` indicating the presence of a padding
                token and `False` for non-padding tokens. Expected to have the shape:
                `(batch_size, context_size)`.

        Returns:
            A tuple of (output_embeddings, attention_weights). With the output
            embeddings, having the same shape as the input embeddings:
            `(batch_size, context_size, embedding_size)` and the attention weights
            having the shape: `(batch_size, num_heads, context_size, context_size)`.

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
