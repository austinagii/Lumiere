import torch
from torch import nn

from lumiere.research.src.utils import validation

from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Applies a transformer block over a batch of token embeddings.

    This transformer block applies the following operations in sequence:
    1. RMS normalization → Multi-head attention → Dropout → Residual connection
    2. RMS normalization → Feed-forward network → Dropout → Residual connection

    Args:
        embedding_size (int): The dimensionality of the token embeddings.
        num_heads (int): The number of attention heads.
        d_key (int): The dimensionality of the key vectors in the multi-head
            attention layer.
        d_value (int): The dimensionality of the value vectors in the multi-head
            attention layer.
        d_ff (int): The dimensionality of the feed-forward network's hidden
            representations.
        dropout (float): The dropout probability. Defaults to 0.1.
        pre_norm

    Shape:
        - Input: `(batch_size, context_size, embedding_size)`
        - Output: `Tuple[torch.Tensor, torch.Tensor]`
            1. `(batch_size, context_size, embedding_size)`
            2. `(batch_size, num_heads, context_size, context_size)`

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
        norm_scheme: str = "rms",
    ) -> None:
        """Initialize a new transformer block.

        Args:
            embedding_size: The dimensionality of the token embeddings.
            num_heads: The number of attention heads.
            d_key: The dimensionality of the key vectors in the multi-head
                attention layer.
            d_value: The dimensionality of the value vectors in the multi-head
                attention layer.
            d_ff: The dimensionality of the feed-forward network's hidden
                representations.
            dropout: The dropout probability. Defaults to 0.1.
            pre_norm: Whether layer inputs should be normalized. Defaults to True.
            post_norm: Whether layer outputs should be normalized. Defaults to False.
            norm_scheme: The normalization algorithm to use. Possible values are:
                ['rms', 'layer']. Defaults to 'rms'
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

        self._embedding_size = embedding_size
        self._num_heads = num_heads
        self._d_key = d_key
        self._d_value = d_value
        self._d_ff = d_ff
        self._dropout = dropout
        self._pre_norm = pre_norm
        self._post_norm = post_norm
        self._norm_scheme = norm_scheme
        self._norm_scheme = norm_scheme

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
        match self._norm_scheme:
            case "rms":
                return nn.RMSNorm(self._embedding_size)
            case "layer":
                return nn.LayerNorm(self._embedding_size)
            case _:
                raise ValueError(f"Invalid normalization scheme: '{self._norm_scheme}'")

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
