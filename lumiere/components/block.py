import torch
from torch import nn

from lumiere.utils import validation

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
    ) -> None:
        super().__init__()

        validation.validate_positive_integer(embedding_size, "embedding_size")
        validation.validate_positive_integer(num_heads, "num_heads")
        validation.validate_positive_integer(d_key, "d_key")
        validation.validate_positive_integer(d_value, "d_value")
        validation.validate_positive_integer(d_ff, "d_ff")
        validation.validate_probability(dropout, "dropout")

        self._embedding_size = embedding_size
        self._num_heads = num_heads
        self._d_key = d_key
        self._d_value = d_value
        self._d_ff = d_ff
        self._dropout = dropout

        self.attention = MultiHeadAttention(
            num_heads=self._num_heads,
            embedding_size=self._embedding_size,
            d_key=self._d_key,
            d_value=self._d_value,
            dropout=self._dropout,
        )
        self.normalization_1 = nn.RMSNorm(self._embedding_size)
        self.feedforward = FeedForward(
            self._embedding_size, self._d_ff, dropout=self._dropout
        )
        self.dropout = nn.Dropout(self._dropout)
        self.normalization_2 = nn.RMSNorm(self._embedding_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        norm_x1 = self.normalization_1(x)
        attention_values, attention_weights = self.attention(norm_x1)
        x = x + self.dropout(attention_values)

        # Compute the feed-forward output.
        norm_x2 = self.normalization_2(x)
        output = self.feedforward(norm_x2)
        x = x + output

        return x, attention_weights
