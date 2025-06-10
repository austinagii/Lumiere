import torch
from torch import nn

from lumiere.components.activations import SwiGLU

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
        x = self.swiglu(x)
        x = self.dropout(x)
        return x