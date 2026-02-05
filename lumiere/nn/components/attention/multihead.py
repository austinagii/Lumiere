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
        >>> from lumiere.nn.components.attention import MultiHeadAttention
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
    ) -> None:
        super().__init__()

        validation.validate_integer(num_heads, "num_heads", min_value=1)
        validation.validate_integer(embedding_size, "embedding_size", min_value=1)
        validation.validate_integer(d_key, "d_key", min_value=1)
        validation.validate_integer(d_value, "d_value", min_value=1)

        self.d_key = d_key
        self.d_value = d_value
        self.num_heads = num_heads
        self.embedding_size = embedding_size

        self._q_proj = nn.Linear(embedding_size, d_key * num_heads, bias=False)
        self._k_proj = nn.Linear(embedding_size, d_key * num_heads, bias=False)
        self._v_proj = nn.Linear(embedding_size, d_value * num_heads, bias=False)
        self._o_proj = nn.Linear(d_value * num_heads, embedding_size, bias=False)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                "The input tensor must have the shape (batch_size, context_size, "
                "embedding_size)"
            )
        if x.size(-1) != self.embedding_size:
            raise ValueError(
                f"Expected embedding_size={self.embedding_size}, got {x.size(-1)}"
            )

        queries = self._q_proj(x)
        keys = self._k_proj(x)
        values = self._v_proj(x)

        queries = split_heads(queries, self.num_heads, self.d_key)
        keys = split_heads(keys, self.num_heads, self.d_key)
        values = split_heads(values, self.num_heads, self.d_value)

        attention_scores = torch.matmul(queries, torch.transpose(keys, -2, -1))
        scaled_attention_scores = attention_scores / torch.sqrt(
            torch.tensor(self.d_key, dtype=queries.dtype, device=queries.device)
        )

        mask = create_causal_mask(queries.shape[2], padding_mask)

        masked_attention_scores = scaled_attention_scores.masked_fill(
            mask, -float("inf")
        )

        attention_weights = stable_softmax(masked_attention_scores)
        attention_values = torch.matmul(attention_weights, values)
        attention_values = concat_heads(attention_values)
        output = self._o_proj(attention_values)

        return output, attention_weights


def create_causal_mask(
    context_size: int, padding_mask: torch.Tensor = None
) -> torch.Tensor:
    """Creates a causal mask for the attention operation.

    This mask is used to prevent each token from attending to itself, future tokens,
    and padding tokens (if a padding mask is provided).

    Args:
        context_size: The size of the context.
        padding_mask: A mask indicating which tokens are padding.

    Returns:
        A mask of shape (1, 1, context_size, context_size).
    """
    # Create a mask that prevents each token from attending to itself and future tokens.
    mask = torch.triu(
        torch.ones(
            context_size,
            context_size,
            dtype=torch.bool,
        ),
    )
    mask = mask.unsqueeze(0).unsqueeze(0)

    # Add the padding mask to prevent attention to/from padding tokens
    if padding_mask is not None:
        # TODO: Make this more explicit, unsqueeze is obfuscating the intent.
        padding_mask_cols = padding_mask.unsqueeze(1).unsqueeze(1)
        padding_mask_rows = padding_mask.unsqueeze(1).unsqueeze(3)
        padding_mask_combined = padding_mask_cols | padding_mask_rows

        mask = mask.to(padding_mask.device) | padding_mask_combined

    return mask


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


def stable_softmax(x: torch.Tensor) -> torch.Tensor:
    """Applies softmax with numerical stability.

    This is made to explicity handle the case where the values along the specified
    dimension are all negative infinity. In this case, the softmax will return all
    zeros instead of NaN.

    Args:
        x: Input tensor

    Returns:
        Tensor of the same shape as the input tensor
    """
    return torch.exp(x) / (torch.sum(torch.exp(x), dim=-1, keepdim=True) + 1e-9)
