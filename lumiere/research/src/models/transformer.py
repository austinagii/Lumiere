from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from lumiere.research.src.components import Embedding, TransformerBlock


"""
def feedforward_factory(*args, **kwargs):
    # Do some stuff here.
    def _factory():
        return ffmodule 

def transformer_block_factory(*args, **kwargs):
    # Do some stuff here.
    def _factory():
        return block

feedforward_factory = LinearFeedForwardFactory(embedding_size)
transformer_block_factory = TransformerBlockFactory(
    num_heads=10, d_key=12, feedforward=feedforward_factory
)
"""

# @dataclass
# class TransformerArgs:
#     vocab_size: int
#     embedding_size: int
#     context_size: int
#     num_layers: int
#     num_heads: int
#     d_key: int
#     d_value: int
#     d_ff: int
#     dropout: float = 0.1
#     padding_id: int | None = None
#     pre_norm: bool = True
#     post_norm: bool = False
#     norm_type: str = "rms"


class Transformer(nn.Module):
    """A transformer model."""

    # TODO: Create transformer configuration object to avoid passing all these args.
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        context_size: int,
        num_layers: int,
        num_heads: int,
        d_key: int,
        d_value: int,
        feedforward_factory: Callable,
        dropout: float = 0.1,
        padding_id: int | None = None,
        pre_norm: bool = True,
        post_norm: bool = False,
        norm_type: str = "rms",
    ):
        """Initialize a transformer model.

        Args:
            vocab_size: The number of unique tokens in the vocabulary.
            embedding_size: The dimensionality of the token embeddings.
            context_size: The maximum number of tokens in a sequence.
            num_layers: The number of transformer blocks in the network.
            num_heads: The number of attention heads.
            d_key: The dimensionality of the key vectors.
            d_value: The dimensionality of the value vectors.
            feedforward_factory: A callable that produces feedforwward modules.
            dropout: The dropout probability. Defaults to 0.1.
            padding_id: The ID of the padding token.
            pre_norm: Whether to apply normalization before attention and
                feed-forward layers. Defaults to True.
            post_norm: Whether to apply normalization after attention and
                feed-forward layers. Defaults to False.
            norm_type: The type of normalization to use. Defaults to "rms".

        """
        super().__init__()

        self._vocab_size = vocab_size
        self._context_size = context_size
        self._embedding_size = embedding_size
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._d_key = d_key
        self._d_value = d_value
        self._dropout = dropout
        self._pre_norm = pre_norm
        self._post_norm = post_norm
        self._norm_type = norm_type

        self.embedding = Embedding(
            self._vocab_size,
            self._context_size,
            self._embedding_size,
            padding_id=padding_id,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_size=self._embedding_size,
                    num_heads=self._num_heads,
                    d_key=self._d_key,
                    d_value=self._d_value,
                    feedforward_factory=feedforward_factory,
                    dropout=self._dropout,
                    pre_norm=self._pre_norm,
                    post_norm=self._post_norm,
                    norm_type=self._norm_type,
                )
                for _ in range(self._num_layers)
            ]
        )

        # TODO: Fix to use the norm type that is specified.
        self.final_norm = nn.RMSNorm(self._embedding_size)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass a batch of input tokens through the transformer model.

        Args:
            x: A batch of token IDs of shape `(batch_size, context_size)`.
            padding_mask: A boolean mask indicating which of the tokens in the
                batch are padding tokens, with `True` indicating the presence of
                a padding token and `False` for non-padding tokens. Expected to
                have the shape: `(batch_size, context_size)`.

        Returns:
            A tuple of logits and attention weights. The logits have shape
            `(batch_size, context_size, vocab_size)` and the attention weights have
            shape `(batch_size, num_layers, num_heads, context_size, context_size)`.

        Raises:
            ValueError: If the input tensor does not have 2 dimensions or if the
                input tensor has a context size greater than the model's context
                size.

        """
        if x.ndim != 2:
            raise ValueError(
                f"The input tensor must have 2 dimensions, but got {x.ndim}."
            )

        if x.shape[1] > self._context_size:
            raise ValueError(
                f"The input tensor must have a context size of at most"
                f"{self._context_size}, but got {x.shape[1]}."
            )

        x = self.embedding(x, padding_mask)

        attention_weights = []
        for block in self.blocks:
            x, block_attention_weights = block(x, padding_mask)
            attention_weights.append(block_attention_weights)
        attention_weights = torch.stack(attention_weights, dim=1)

        x = self.final_norm(x)
        x = F.linear(x, self.embedding._embedding.weight)
        return x, attention_weights

    @property
    def context_size(self) -> int:
        """The maximum number of tokens in a sequence."""
        return self._context_size

    @property
    def vocab_size(self) -> int:
        """The number of unique tokens in the vocabulary."""
        return self._vocab_size
