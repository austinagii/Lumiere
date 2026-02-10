from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


class Transformer(nn.Module):
    """A transformer model.

    Attributes:
        context_size (int): The maximum allowed number of tokens in a single sequence.
        num_layers (int): The number of transformer blocks in the model.

    """

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        num_blocks: int,
        embedding: Callable,
        block: Callable,
        normalization: Callable,
    ):
        """Initialize a transformer model.

        Args:
            vocab_size: The number of unique tokens in the vocabulary.
            context_size: The maximum number of tokens in a sequence.
            num_blocks: The number of transformer blocks in the network.
            embedding: A callable factory that produces embedding modules.
            block: A callable factory that produces transformer block modules.
            normalization: A callable factory that produces normalization layers.

        """
        super().__init__()

        self.context_size = context_size
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.embedding = embedding()
        self.blocks = nn.ModuleList([block() for _ in range(self.num_blocks)])
        self.final_norm = normalization()

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

    def fit(self, tokens, padding_mask) -> None:
        pass
