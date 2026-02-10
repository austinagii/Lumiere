import torch
from torch import nn
from torch.nn import functional as F

from lumiere.discover import discover
from lumiere.utils import validation


@discover(nn.Module, "feedforward.swiglu")
class SwigluFeedForward(nn.Module):
    """A SwiGLU feed-forward network.

    This class implements the SwiGLU variant of the GLU activation function as
    described in the paper `GLU Variants Improve Transformer <https://arxiv.org/abs/2002.05202>`_.

    Attributes:
        embedding_size (int): The dimensionality of the token embeddings.
        hidden_size (int): The hidden dimension of the feed-forward network.
        dropout (float): The dropout probability.

    """

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize a SwiGLU feed-forward network.

        Args:
            embedding_size: The dimensionality of the token embeddings.
            hidden_size: The hidden dimension of the feed-forward network.
            dropout: The dropout probability.

        Raises:
            ValueError: If any of the arguments are invalid.

        """
        super().__init__()

        validation.validate_integer(embedding_size, "embedding_size", min_value=1)
        validation.validate_integer(hidden_size, "hidden_size", min_value=1)
        validation.validate_probability(dropout, "dropout")

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self._gate_proj = nn.Linear(self.embedding_size, self.hidden_size, bias=False)
        self._up_proj = nn.Linear(self.embedding_size, self.hidden_size, bias=False)
        self._down_proj = nn.Linear(self.hidden_size, self.embedding_size, bias=False)
        self._dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass a batch of token embeddings through the SwiGLU feed-forward network.

        Args:
            x: A batch of token embeddings of shape
                `(batch_size, context_size, embedding_size)`.

        Returns:
            A batch of output embeddings of shape
                `(batch_size, context_size, embedding_size)`.

        """
        assert x.dim() >= 2
        assert x.size(-1) == self.embedding_size

        gate = F.silu(self._gate_proj(x))
        up = self._up_proj(x)
        hidden = gate * up
        down = self._down_proj(hidden)
        return self._dropout(down)

    @property
    def gate_proj(self):
        """The linear layer that gates the hidden states."""
        return self._gate_proj

    @property
    def up_proj(self):
        """The linear layer that projects embeddings to the hidden dimension."""
        return self._up_proj

    @property
    def down_proj(self):
        """The linear layer that projects hidden states back to embeddings."""
        return self._down_proj
