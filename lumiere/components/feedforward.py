import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """Applies the SwiGLU feed-forward operation over a batch of token embeddings.

    This layer implements the operation as described in the paper:
    https://arxiv.org/abs/2002.05202

    Args:
        embedding_size (int): The dimensionality of the token embeddings.
        hidden_size (int): The dimensionality of the tokens' hidden representation.
        dropout (float): The dropout probability. Default: 0.1.

    Shape:
        - Input: `(batch_size, context_size, embedding_size)`
        - Output: `(batch_size, context_size, embedding_size)`
    """

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._embedding_size = embedding_size
        self._hidden_size = hidden_size
        self._dropout_p = dropout

        self._gate_proj = nn.Linear(self._embedding_size, self._hidden_size, bias=False)
        self._up_proj = nn.Linear(self._embedding_size, self._hidden_size, bias=False)
        self._down_proj = nn.Linear(self._hidden_size, self._embedding_size, bias=False)
        self._dropout = nn.Dropout(self._dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self._gate_proj(x))
        up = self._up_proj(x)
        hidden = gate * up
        down = self._down_proj(hidden)
        return self._dropout(down)