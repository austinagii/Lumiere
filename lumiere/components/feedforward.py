import torch
from torch import nn
from torch.nn import functional as F

class SwiGLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, input_dim, bias=False)
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden)


class FeedForward(nn.Module):
    """Feed-forward network for the transformer block using SwiGLU.
    
    Args:
        embedding_size (int): The size of the embedding
        d_ff (int): The size of the feed-forward network
        dropout (float): Dropout probability
    """
    def __init__(self, embedding_size: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.swiglu = SwiGLU(embedding_size, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.swiglu(x))