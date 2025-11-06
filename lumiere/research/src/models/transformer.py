from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from lumiere.research.src.components import Embedding, TransformerBlock


class Transformer(nn.Module):
    """A transformer model.

    Attributes:
        context_size (int): The maximum allowed number of tokens in a single sequence.
        num_layers (int): The number of transformer blocks in the model.

    """

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

        self.context_size = context_size
        self.num_layers = num_layers

        self.embedding = Embedding(
            vocab_size,
            context_size,
            embedding_size,
            padding_id=padding_id,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_size=embedding_size,
                    num_heads=num_heads,
                    d_key=d_key,
                    d_value=d_value,
                    feedforward_factory=feedforward_factory,
                    dropout=dropout,
                    pre_norm=pre_norm,
                    post_norm=post_norm,
                    norm_type=norm_type,
                )
                for _ in range(self.num_layers)
            ]
        )

        # TODO: Fix to use the norm type that is specified.
        self.final_norm = nn.RMSNorm(embedding_size)

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

        """
        assert x.dim() == 2
        assert x.shape[-1] <= self.context_size

        x = self.embedding(x, padding_mask)

        attention_weights = []
        for block in self.blocks:
            x, block_attention_weights = block(x, padding_mask)
            attention_weights.append(block_attention_weights)
        attention_weights = torch.stack(attention_weights, dim=1)

        x = self.final_norm(x)
        x = F.linear(x, self.embedding._embedding.weight)
        return x, attention_weights
