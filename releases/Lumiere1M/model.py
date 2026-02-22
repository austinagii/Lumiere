"""Lumiére-1M: A 1-million parameter transformer language model."""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class Lumiere1M(nn.Module):
    """A Lumiére-1M transformer model."""

    vocab_size: int = 4096
    embedding_size: int = 128
    context_size: int = 64
    num_blocks: int = 4
    num_heads: int = 4
    d_key: int = 32
    d_value: int = 32
    d_ff: int = 32

    def __init__(self, padding_id: int):
        """Initialize a Lumiére-1M transformer model.

        Args:
            padding_id: The ID of the padding token.

        """
        super().__init__()
        self.padding_id = padding_id

        self.embedding = Embedding(
            self.vocab_size,
            self.context_size,
            self.embedding_size,
            self.padding_id,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_size=self.embedding_size,
                    num_heads=self.num_heads,
                    d_key=self.d_key,
                    d_value=self.d_value,
                    d_ff=self.d_ff,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.norm = nn.RMSNorm(self.embedding_size)

    @torch.inference_mode()
    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform next token prediction for a batch of tokens.

        Args:
            x: A batch of input tokens of shape `(batch_size, context_size)`.
            padding_mask: A boolean mask indicating which of the tokens in the batch
                are padding tokens, with `True` indicating the presence of a padding
                token and `False` for non-padding tokens. Expected to have the shape:
                `(batch_size, context_size)`.

        Returns:
            A tuple of logits and attention weights. The logits have shape
            `(batch_size, context_size, vocab_size)` and the attention weights have
            shape `(batch_size, num_layers, num_heads, context_size, context_size)`.

        """
        x = self.embedding(x, padding_mask)

        attention_weights = []
        for block in self.blocks:
            x, block_attention_weights = block(x, padding_mask)
            attention_weights.append(block_attention_weights)
        attention_weights = torch.stack(attention_weights, dim=1)

        x = self.norm(x)
        x = F.linear(x, self.embedding._embedding.weight)
        return x, attention_weights


class Embedding(nn.Module):
    """Converts token IDs to positional-encoded token embeddings.

    Takes a tensor of token IDs and returns the corresponding token embeddings
    with sinusoidal positional encodings added.

    Attributes:
        embedding_size (int): The dimensionality of the token embeddings.

    """

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embedding_size: int,
        padding_id: int,
    ) -> None:
        """Initialize a token embedding layer.

        Args:
            vocab_size: The number of unique tokens in the vocabulary.
            context_size: The maximum number of tokens in a sequence.
            embedding_size: The dimensionality of the token embeddings.
            padding_id: The ID of the padding token.

        """
        super().__init__()

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_size = embedding_size

        self._padding_id = padding_id

        self._embedding = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=self._padding_id
        )
        positional_encoding = _sinusoidal_positional_encoding(
            self.context_size, self.embedding_size
        )
        self.register_buffer("_positional_encoding", positional_encoding)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Convert a batch of token IDs to sinusoidal position-encoded token embeddings.

        Args:
            x: A batch of token IDs of shape `(batch_size, context_size)`.
            padding_mask: A boolean mask indicating which of the tokens in the batch
                are padding tokens, with `True` indicating the presence of a padding
                token and `False` for non-padding tokens. Expected to have the shape:
                `(batch_size, context_size)`.

        Returns:
            A batch of sinusoidal position-encoded token embeddings of shape
            `(batch_size, context_size, embedding_size)`.

        """
        token_embeddings = self._embedding(x)
        position_encoding = self._positional_encoding[: x.shape[-1], :]

        # Ensure that padding tokens do not receive positional encoding.
        if padding_mask is not None:
            # padding mask shape: [batch_size, context_size]
            # pos encoding shape: [context_size, embedding_size]
            # needed out shape  : [batch_size, context_size, embedding_size]
            position_encoding = position_encoding.expand(
                padding_mask.shape[0], -1, -1
            ).clone()

            position_encoding[padding_mask] = 0

        return token_embeddings + position_encoding


def _sinusoidal_positional_encoding(
    context_size: int, embedding_size: int
) -> torch.Tensor:
    """Computes sinusoidal positional encodings for a given context and embedding size.

    The positional encoding matrix is computed using the following formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/embedding_size))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/embedding_size))

    Args:
        context_size (int): The number of tokens in the context.
        embedding_size (int): The dimensionality of the token embeddings.

    Returns:
        A tensor of shape (context_size, embedding_size) containing the
        sinusoidal positional encoding matrix.

    """
    positions = torch.arange(context_size, dtype=torch.float32)
    indices = torch.arange(embedding_size // 2, dtype=torch.float32)

    scaling_factor = 10_000 ** ((2 * indices) / embedding_size)
    angles = positions.unsqueeze(1) / scaling_factor
    pos_encoding = torch.zeros((context_size, embedding_size), dtype=torch.float32)
    pos_encoding[:, 0::2] = torch.sin(angles)
    pos_encoding[:, 1::2] = torch.cos(angles)
    return pos_encoding


class TransformerBlock(nn.Module):
    """A decoder transformer block."""

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        d_key: int,
        d_value: int,
        d_ff: int,
    ) -> None:
        """Initialize a transformer block.

        Args:
            embedding_size: The dimensionality of the token embeddings.
            num_heads: The number of attention heads.
            d_key: The dimensionality of the key vectors.
            d_value: The dimensionality of the value vectors.
            d_ff: The hidden dimension of the feed-forward network.

        """
        super().__init__()

        self.normalization_1 = nn.RMSNorm(embedding_size)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            embedding_size=embedding_size,
            d_key=d_key,
            d_value=d_value,
        )
        self.normalization_2 = nn.RMSNorm(embedding_size)
        self.feedforward = FeedForward(embedding_size, d_ff)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass a batch of token embeddings through the transformer block.

        Args:
            x: A batch of token embeddings of shape
                `(batch_size, context_size, embedding_size)`.
            padding_mask: A boolean mask indicating which of the tokens in the batch
                are padding tokens, with `True` indicating the presence of a padding
                token and `False` for non-padding tokens. Expected to have the shape:
                `(batch_size, context_size)`.

        Returns:
            A tuple of output embeddings and attention weights. The output embeddings
            have shape `(batch_size, context_size, embedding_size)` and the attention
            weights have shape `(batch_size, num_heads, context_size, context_size)`.
        """
        residual = x
        x = self.normalization_1(x)
        attention_values, attention_weights = self.attention(
            x, padding_mask=padding_mask
        )
        x = residual + attention_values

        residual = x
        x = self.normalization_2(x)
        x = residual + self.feedforward(x)

        return x, attention_weights


class MultiHeadAttention(nn.Module):
    """Performs the multi-head self attention operation on a batch of token embeddings.

    This class implements the multi-head operation as described in the paper `Attention
    Is All You Need <https://arxiv.org/abs/1706.03762>`.

    Attributes:
        d_key (int): The dimensionality of the key vectors.
        d_value (int): The dimensionality of the value vectors.
        num_heads (int): The number of attention heads.
        embedding_size (int): The dimensionality of the token embeddings.

    """

    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        d_key: int,
        d_value: int,
    ):
        """Initialize a multi-head attention module.

        Args:
            num_heads: The number of attention heads.
            embedding_size: The dimensionality of the token embeddings.
            d_key: The dimensionality of the key vectors.
            d_value: The dimensionality of the value vectors.

        """
        super().__init__()

        self.d_key = d_key
        self.d_value = d_value
        self.num_heads = num_heads
        self.embedding_size = embedding_size

        self._q_proj = nn.Linear(embedding_size, d_key * num_heads, bias=False)
        self._k_proj = nn.Linear(embedding_size, d_key * num_heads, bias=False)
        self._v_proj = nn.Linear(embedding_size, d_value * num_heads, bias=False)
        self._o_proj = nn.Linear(d_value * num_heads, embedding_size, bias=False)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Perform the multi-head self attention operation on a batch of tokens.

        Args:
            x: A batch of token embeddings of shape
                `(batch_size, context_size, embedding_size)`
            padding_mask: A boolean mask of shape `(batch_size, context_size)`
                indicating which of the tokens in the batch are padding tokens, with
                `True` indicating the presence of a padding token and `False` for
                non-padding tokens.

        Returns:
            A tuple of output embeddings and attention weights. The output
            embeddings have shape `(batch_size, context_size, embedding_size)` and
            the attention weights have shape
            `(batch_size, num_heads, context_size, context_size)`.

        Raises:
            ValueError: If the specified token embeddings have the incorrect shape.

        """
        queries = self._q_proj(x)
        keys = self._k_proj(x)
        values = self._v_proj(x)

        queries = _split_heads(queries, self._num_heads, self._d_key)
        keys = _split_heads(keys, self._num_heads, self._d_key)
        values = _split_heads(values, self._num_heads, self._d_value)

        attention_scores = torch.matmul(queries, torch.transpose(keys, -2, -1))
        scaled_attention_scores = attention_scores / torch.sqrt(
            torch.tensor(self._d_key, dtype=queries.dtype, device=queries.device)
        )

        mask = _create_causal_mask(queries.shape[2], padding_mask)

        masked_attention_scores = scaled_attention_scores.masked_fill(
            mask, -float("inf")
        )

        attention_weights = _stable_softmax(masked_attention_scores)
        attention_values = torch.matmul(attention_weights, values)
        attention_values = _concat_heads(attention_values)
        output = self._o_proj(attention_values)

        return output, attention_weights


def _create_causal_mask(
    context_size: int, padding_mask: torch.Tensor = None
) -> torch.Tensor:
    """Create a causal mask for the attention operation.

    This mask is used to prevent each token from attending to itself, future tokens,
    and padding tokens (if a padding mask is provided).

    Args:
        context_size: The maximum number of tokens in a sequence.
        padding_mask: A boolean mask indicating which of the tokens are padding
            tokens, with `True` indicating a padding token and `False` for
            non-padding tokens.

    Returns:
        A mask of shape `(1, 1, context_size, context_size)`.
    """
    # Create a mask that prevents each token from attending to itself and future tokens.
    mask = torch.triu(
        torch.ones(
            context_size,
            context_size,
            dtype=torch.bool,
        ),
    )
    mask = mask.unsqueeze(0).unsqueeze(0)

    # Add the padding mask to prevent attention to/from padding tokens
    if padding_mask is not None:
        # TODO: Make this more explicit, unsqueeze is obfuscating the intent.
        padding_mask_cols = padding_mask.unsqueeze(1).unsqueeze(1)
        padding_mask_rows = padding_mask.unsqueeze(1).unsqueeze(3)
        padding_mask_combined = padding_mask_cols | padding_mask_rows

        mask = mask.to(padding_mask.device) | padding_mask_combined

    return mask


def _split_heads(
    tensor: torch.Tensor, num_heads: int, num_features: int
) -> torch.Tensor:
    """Split the concatenated multi-head features into separate heads.

    Args:
        tensor: The tensor to split of shape
            `(batch_size, context_size, num_heads * num_dimensions)`.
        num_heads: The number of attention heads.
        num_features: The number of features per attention head.

    Returns:
        A tensor of shape `(batch_size, num_heads, context_size, num_dimensions)`.
    """
    return tensor.view(
        tensor.shape[0], tensor.shape[1], num_heads, num_features
    ).transpose(1, 2)


def _concat_heads(tensor: torch.Tensor) -> torch.Tensor:
    """Concatenate the tensor from multiple attention heads.

    Args:
        tensor: The tensor to concatenate of shape
            `(batch_size, num_heads, context_size, num_dimensions)`.

    Returns:
        A tensor of shape `(batch_size, context_size, num_heads * num_dimensions)`.
    """
    return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], -1)


def _stable_softmax(x: torch.Tensor) -> torch.Tensor:
    """Apply softmax with numerical stability.

    This is made to explicitly handle the case where the values along the specified
    dimension are all negative infinity. In this case, the softmax will return all
    zeros instead of NaN.

    Args:
        x: The input tensor.

    Returns:
        A tensor of the same shape as the input tensor.
    """
    x_max = x.amax(dim=-1, keepdim=True)
    x_max = torch.clamp(x_max, min=0)
    exp_x = torch.exp(x - x_max)
    return exp_x / (exp_x.sum(dim=-1, keepdim=True) + 1e-9)


class FeedForward(nn.Module):
    """A position-wise feed-forward network.

    This class implements the position-wise feed-forward network as described in the
    paper `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`.
    """

    def __init__(self, embedding_size: int, d_ff: int):
        """Initialize a position-wise feed-forward network.

        Args:
            embedding_size: The dimensionality of the token embeddings.
            d_ff: The hidden dimension of the feed-forward network.

        """
        super().__init__()

        self.embedding_size = embedding_size
        self.d_ff = d_ff

        self._layers = nn.Sequential(
            OrderedDict(
                [
                    ("up_proj", nn.Linear(embedding_size, d_ff, bias=True)),
                    ("activation", nn.GELU()),
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
