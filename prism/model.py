import torch
from torch import nn
from prism.embedding import Embedding


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        context_size: int,
        num_heads: int = 12,
        d_key: int = 64,
        d_value: int = 64
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(
            context_size, embedding_size)
        self.masked_multi_head_attention = MultiHeadAttention(
            embedding_size, num_heads, d_key, d_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding(self.embedding(x))


class PositionalEncoding(nn.Module):
    """Adds sinusoidal positional encoding to the input tensor."""

    def __init__(self, context_size: int, embedding_size: int):
        super().__init__()
        self.positional_encoding = sinusoidal_positional_encoding(
            (context_size, embedding_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding + x


def sinusoidal_positional_encoding(shape: tuple[int, int]) -> torch.Tensor:
    """Returns the sinusoidal positional encoding matrix with the given shape.

    The positional encoding matrix has shape (context_size, embedding_size) 
    and is computed using the following formula:

        PE(pos, 2i)   = sin(pos / 10000^(2i/embedding_size))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/embedding_size))

    Where pos is the position of the token in the context and i is the 
    index of the pair of dimensions under consideration.

    Shape is expected to be a pair of positive integers where the second 
    integer is even.

    Args:
        shape: A tuple of (context_size, embedding_size).

    Returns:
        A tensor of shape (context_size, embedding_size) containing the 
        sinusoidal positional encoding matrix.

    Raises:
        ValueError: If the specified shape is invalid.
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("Shape must be a tuple (context_size, embedding_size).")

    context_size, embedding_size = shape
    if not (isinstance(context_size, int) and context_size > 0):
        raise ValueError("Context size must be a positive integer.")
    if not (isinstance(embedding_size, int) and embedding_size > 0 and embedding_size % 2 == 0):
        raise ValueError("Embedding size must be a positive, even integer.")

    positions = torch.arange(context_size, dtype=torch.float32)
    indices = torch.arange(embedding_size / 2, dtype=torch.float32)

    scaling_factor = 10_000 ** ((2 * indices) / embedding_size)
    angles = positions.unsqueeze(1) / scaling_factor
    pos_encoding = torch.zeros((context_size, embedding_size), dtype=torch.float32)
    pos_encoding[:, 0::2] = torch.sin(angles)
    pos_encoding[:, 1::2] = torch.cos(angles)
    # breakpoint()
    return pos_encoding


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        d_key: int,
        d_value: int,
        masked: bool = True
    ) -> None:
        super().__init__()
        self.masked = masked
        self.d_key = d_key
        self.q_proj = nn.Parameter(torch.randn(
            num_heads, embedding_size, d_key))
        self.k_proj = nn.Parameter(torch.randn(
            num_heads, embedding_size, d_key))
        self.v_proj = nn.Parameter(torch.randn(
            num_heads, embedding_size, d_value))

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Creates the query, key and value matrices for a given batch of 
        tokens.

        Args:
            x: A tensor of shape (batch_size, context_size, embedding_size)
            containing the token embeddings.

        Returns:
            A tensor of shape (batch_size, num_heads, context_size, d_value)
            containing the attention weights.
        """
        queries = torch.bmm(x, self.q_proj)
        keys = torch.bmm(x, self.k_proj)
        values = torch.bmm(x, self.v_proj)

        raw_attn_scores = torch.bmm(queries, torch.transpose(keys, -2, -1))
        scaled_attn_scores = raw_attn_scores / torch.sqrt(self.d_key)

        # mask the attention scores now
        if self.masked:
            mask = torch.triu(torch.ones_like(
                raw_attn_scores, dtype=torch.bool))
            scaled_attn_scores = scaled_attn_scores.masked_fill(
                mask, -float('inf'))

        # apply the softmax function
        attn_weights = torch.softmax(scaled_attn_scores, dim=-1)

        # apply the attention weights to the values
        out = torch.bmm(attn_weights, values)

        return out

# Add tests for all modules.
