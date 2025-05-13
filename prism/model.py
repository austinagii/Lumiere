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
        self.embedding = Embedding(vocab_size, context_size, embedding_size)
        self.masked_multi_head_attention = MultiHeadAttention(
            embedding_size, num_heads, d_key, d_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


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
        self.q_proj = nn.Parameter(torch.randn(num_heads, embedding_size, d_key))
        self.k_proj = nn.Parameter(torch.randn(num_heads, embedding_size, d_key))
        self.v_proj = nn.Parameter(torch.randn(num_heads, embedding_size, d_value))

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