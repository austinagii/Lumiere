import torch
from torch import nn

from prism.validation import validate_positive_integer, validate_boolean


class MultiHeadAttention(nn.Module):
    """Performs multi-head self attention on a batch of tokens.

    Args:
        num_heads (int): The number of attention heads
        embedding_size (int): The size of the embedding
        d_key (int): The size of the key
        d_value (int): The size of the value
        masked (bool): Whether to mask the attention scores
    """
    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        d_key: int,
        d_value: int,
        masked: bool = True
    ) -> None:
        validate_positive_integer(num_heads, "num_heads")
        validate_positive_integer(embedding_size, "embedding_size")
        validate_positive_integer(d_key, "d_key")
        validate_positive_integer(d_value, "d_value")
        validate_boolean(masked, "masked")

        super().__init__()
        self._masked = masked
        self._q_proj = nn.Parameter(torch.randn(num_heads, embedding_size, d_key))
        self._k_proj = nn.Parameter(torch.randn(num_heads, embedding_size, d_key))
        self._v_proj = nn.Parameter(torch.randn(num_heads, embedding_size, d_value))

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
    

    @property
    def masked(self):
        return self._masked

    @property
    def q_proj(self):
        return self._q_proj

    @property
    def k_proj(self):
        return self._k_proj

    @property
    def v_proj(self):
        return self._v_proj
