from collections import OrderedDict

import torch
from torch import nn

from lumiere.internal.registry import discover
from lumiere.utils import validation


@discover(nn.Module, "feedforward.linear")
class LinearFeedForward(nn.Module):
    """A position-wise feed-forward network.

    This class implements the position-wise feed-forward network as described in the
    paper `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
    """

    def __init__(self, embedding_size: int, d_ff: int, dropout: float = 0.1):
        """Initialize a position-wise feed-forward network.

        Args:
            embedding_size: The dimensionality of the token embeddings.
            d_ff: The hidden dimension of the feed-forward network.
            dropout: The dropout probability. Defaults to 0.1.

        """
        super().__init__()

        validation.validate_integer(embedding_size, "embedding_size", min_value=1)
        validation.validate_integer(d_ff, "d_ff", min_value=1)
        validation.validate_probability(dropout, "dropout")

        self.embedding_size = embedding_size
        self.d_ff = d_ff
        self.dropout = dropout

        self._layers = nn.Sequential(
            OrderedDict(
                [
                    ("up_proj", nn.Linear(embedding_size, d_ff, bias=True)),
                    ("activation", nn.GELU()),
                    ("dropout", nn.Dropout(dropout)),
                    ("down_proj", nn.Linear(d_ff, embedding_size, bias=True)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass a batch of token embeddings through the feed-forward network.

        Args:
            x: A batch of token embeddings of shape
                `(batch_size, context_size, embedding_size)`.

        Returns:
            A batch of transformed token embeddings of shape
            `(batch_size, context_size, embedding_size)`.

        """
        assert x.size(-1) == self.embedding_size
        return self._layers(x)

    @property
    def up_proj(self):
        """The linear layer that projects embeddings to the hidden dimension."""
        return self._layers.up_proj

    @property
    def down_proj(self):
        """The linear layer that projects hidden states back to embeddings."""
        return self._layers.down_proj
