import torch
from torch import nn
from prism.embedding import Embedding
from prism.attention import MultiHeadAttention


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        context_size: int,
        num_heads: int = 12,
        d_key: int = 64,
        d_value: int = 64,
        masked: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, context_size, embedding_size)
        self.masked_multi_head_attention = MultiHeadAttention(
            num_heads,
            embedding_size,
            d_key,
            d_value,
            masked=masked,
            dropout=dropout
        )
        self.normalization = nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.masked_multi_head_attention(x)
        x = self.normalization(x)
        return x
