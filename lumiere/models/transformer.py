import torch
from torch import nn
from torch.nn import functional as F

from lumiere.components import Embedding, TransformerBlock


class Transformer(nn.Module):
    """A transformer model.

    Implements the transformer architecture as described in the paper:
    https://arxiv.org/abs/1706.03762 with the following modifications:
    - Pre-normalization using RMSNorm in the multi-head attention and feed-forward
      layers instead of Post-normalization with LayerNorm.
    - SwiGLU feed-forward network instead of regular ReLU activation.

    Args:
        vocab_size (int): The number of unique tokens in the vocabulary.
        embedding_size (int): The dimensionality of the token embeddings.
        context_size (int): The maximum number of tokens in a sequence.
        num_layers (int): The number of transformer blocks in the network.
        num_heads (int): The number of attention heads in each transformer block.
        d_key (int): The dimensionality of the key vectors in each attention head.
        d_value (int): The dimensionality of the value vectors in each attention head.
        d_ff (int): The dimensionality of each feed-forward layer's hidden
            representations.
        dropout (float): The dropout probability. Defaults to 0.1.

    Shape:
        - Input: `(batch_size, context_size)`
        - Output: `Tuple[torch.Tensor, torch.Tensor]`
            1. `(batch_size, context_size, embedding_size)`
            2. `(batch_size, num_layers, num_heads, context_size, context_size)`

    Raises:
        ValueError: If any of the following conditions are met:
            - The input tensor is not of shape `(batch_size, context_size)`.
            - The input tensor has a context size greater than the model's
              context size.
            - The input tensor has a batch size greater than the model's batch
              size.
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
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._context_size = context_size
        self._embedding_size = embedding_size
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._d_key = d_key
        self._d_value = d_value
        self._d_ff = d_ff
        self._dropout = dropout

        self.embedding = Embedding(
            self._vocab_size, self._context_size, self._embedding_size
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_size=self._embedding_size,
                    num_heads=self._num_heads,
                    d_key=self._d_key,
                    d_value=self._d_value,
                    d_ff=self._d_ff,
                    dropout=self._dropout,
                )
                for _ in range(self._num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(self._embedding_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(
                f"The input tensor must have 2 dimensions, but got {x.ndim}."
            )

        if x.shape[1] > self._context_size:
            raise ValueError(
                f"The input tensor must have a context size of at most"
                f"{self._context_size}, but got {x.shape[1]}."
            )

        x = self.embedding(x)

        attention_weights = []
        for block in self.blocks:
            x, block_attention_weights = block(x)
            attention_weights.append(block_attention_weights)
        attention_weights = torch.stack(attention_weights, dim=1)

        x = self.final_norm(x)
        x = F.linear(x, self.embedding._embedding.weight)
        return x, attention_weights

    @property
    def context_size(self) -> int:
        """The maximum number of tokens in a sequence."""
        return self._context_size
