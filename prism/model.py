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
        d_value: int = 64
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, context_size, embedding_size)
        self.masked_multi_head_attention = MultiHeadAttention(
            embedding_size, num_heads, d_key, d_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)