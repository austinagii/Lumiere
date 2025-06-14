import torch
from torch import nn

from lumiere.utils import validation


class MultiHeadAttention(nn.Module):
    """Performs the multi-head self attention operation on a batch of token embeddings.

    This layer implements the operation as described in the paper:
    https://arxiv.org/abs/1706.03762

    Args:
        num_heads (int): The number of attention heads.
        embedding_size (int): The dimensionality of the token embeddings.
        d_key (int): The dimensionality of the key vectors.
        d_value (int): The dimensionality of the value vectors.
        dropout (float): The dropout probability for the attention weights.
            Defaults to 0.0.

    Shape:
        - Input: `(batch_size, context_size, embedding_size)`
        - Outputs: Tuple[torch.Tensor, torch.Tensor]
            1. `(batch_size, context_size, embedding_size)`
            2. `(batch_size, num_heads, context_size, context_size)`

    Raises:
        ValueError: If any of the following are not positive integers: number of heads,
            embedding size, key dimensionality, or value dimensionality.
        ValueError: If the dropout probability is not a positive float or zero.
        ValueError: If the input tensor is not of shape (batch_size, context_size, 
            embedding_size).

    Example:
        >>> import torch
        >>> from lumiere.components.attention import MultiHeadAttention
        >>> x = torch.randn(1, 3, 128)
        >>> attention = MultiHeadAttention(12, 128, 64, 64)
        >>> output, attention_weights = attention(x)
        >>> print(output.shape)
        torch.Size([1, 3, 128])
        >>> print(attention_weights.shape)
        torch.Size([1, 12, 3, 3])
    """

    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        d_key: int,
        d_value: int,
        dropout: float = 0.0,
    ) -> None:
        validation.validate_positive_integer(num_heads, "num_heads")
        validation.validate_positive_integer(embedding_size, "embedding_size")
        validation.validate_positive_integer(d_key, "d_key")
        validation.validate_positive_integer(d_value, "d_value")
        validation.validate_probability(dropout, "dropout")

        super().__init__()
        self._d_key = d_key
        self._d_value = d_value
        self._num_heads = num_heads
        if dropout > 0.0:
            self._dropout = nn.Dropout(dropout)
        else:
            self._dropout = nn.Identity()

        self._q_proj = nn.Linear(embedding_size, d_key * num_heads)
        self._k_proj = nn.Linear(embedding_size, d_key * num_heads)
        self._v_proj = nn.Linear(embedding_size, d_value * num_heads)
        self._o_proj = nn.Linear(d_value * num_heads, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                "The input tensor must have the shape (batch_size, context_size, "
                "embedding_size)"
            )

        queries = self._q_proj(x)
        keys = self._k_proj(x)
        values = self._v_proj(x)

        queries = split_heads(queries, self._num_heads, self._d_key)
        keys = split_heads(keys, self._num_heads, self._d_key)
        values = split_heads(values, self._num_heads, self._d_value)

        attention_scores = torch.matmul(queries, torch.transpose(keys, -2, -1))
        scaled_attention_scores = attention_scores / torch.sqrt(
            torch.tensor(self._d_key, dtype=queries.dtype, device=queries.device)
        )

        mask = torch.triu(
            torch.ones(
                queries.shape[2],
                queries.shape[2],
                dtype=torch.bool,
                device=queries.device,
            ),
            diagonal=1,
        )
        scaled_attention_scores = scaled_attention_scores.masked_fill(
            mask.unsqueeze(0).unsqueeze(0), -float("inf")
        )

        attention_weights = torch.softmax(scaled_attention_scores, dim=-1)
        attention_weights = self._dropout(attention_weights)
        attention_values = torch.matmul(attention_weights, values)

        attention_values = concat_heads(attention_values)
        output = self._o_proj(attention_values)
        return output, attention_weights


def split_heads(
    tensor: torch.Tensor, num_heads: int, num_features: int
) -> torch.Tensor:
    """Splits the concatenated multi-head features into separate heads.
    
    Args:
        tensor: Tensor to split
        num_heads: Number of attention heads
        num_features: Number of features per attention head

    Shape:
        - Input: `(batch_size, context_size, num_heads * num_dimensions)`
        - Output: `(batch_size, num_heads, context_size, num_dimensions)`

    Returns:
        Tensor of shape (batch_size, num_heads, context_size, num_dimensions)

    Example:
        >>> import torch
        >>> tensor = torch.randn(1, 3, 12)
        >>> output = split_heads(tensor, 2, 6)
        >>> print(output.shape)
        torch.Size([1, 2, 3, 6])
    """
    return tensor.view(
        tensor.shape[0], tensor.shape[1], num_heads, num_features
    ).transpose(1, 2)


def concat_heads(tensor: torch.Tensor) -> torch.Tensor:
    """Concatenates the tensor from multiple attention heads.

    Args:
        tensor: Tensor to concatenate

    Shape:
        - Input: `(batch_size, num_heads, context_size, num_dimensions)`
        - Output: `(batch_size, context_size, num_heads * num_dimensions)`

    Returns:
        Tensor of shape (batch_size, context_size, num_heads * num_dimensions)

    Example:
        >>> import torch
        >>> tensor = torch.randn(1, 2, 3, 4)
        >>> output = concat_heads(tensor)
        >>> print(output.shape)
        torch.Size([1, 3, 12])
    """
    return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], -1)
