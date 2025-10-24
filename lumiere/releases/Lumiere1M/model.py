import torch
from torch import nn
from torch.nn import functional as F

from lumiere.research.src.utils import validation


class Embedding(nn.Module):
    """Converts token IDs to positional-encoded token embeddings.

    Takes a tensor of token IDs and returns the corresponding token embeddings
    with sinusoidal positional encodings added.

    Args:
        vocab_size (int): The number of unique tokens in the vocabulary.
        context_size (int): The number of tokens in the context.
        embedding_size (int): The dimensionality of the token embeddings.

    Shape:
        - Input: `(..., context_size)`
        - Output: `(..., context_size, embedding_size)`

    Raises:
        ValueError: If the vocabulary size or embedding size is not a positive
            integer.
        IndexError: If any of the token ids in the input tensor are outside of
            the range [0, vocab_size).

    Example:
        >>> import torch
        >>> from lumiere.components.embedding import Embedding
        >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> embedding = Embedding(10, 10)
        >>> output = embedding(x)
        >>> print(output.shape)
        torch.Size([2, 3, 10])
    """

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embedding_size: int,
        padding_id: int | None = None,
    ) -> None:
        super().__init__()
        validation.validate_integer(vocab_size, "vocab_size", min_value=1)
        validation.validate_integer(context_size, "context_size", min_value=1)
        validation.validate_integer(embedding_size, "embedding_size", min_value=1)

        self._vocab_size = vocab_size
        self._context_size = context_size
        self._embedding_size = embedding_size
        self._padding_id = padding_id

        self._embedding = nn.Embedding(
            self._vocab_size, self._embedding_size, padding_idx=self._padding_id
        )
        positional_encoding = sinusoidal_positional_encoding(
            self._context_size, self._embedding_size
        )
        self.register_buffer("_positional_encoding", positional_encoding)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if torch.any(x < 0) or torch.any(x >= self._vocab_size):
            raise IndexError("Token ids are outside of the range [0, vocab_size).")

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

    @property
    def vocab_size(self) -> int:
        """The number of unique tokens in the vocabulary."""
        return self._vocab_size

    @property
    def embedding_size(self) -> int:
        """The dimensionality of the token embeddings."""
        return self._embedding_size


def sinusoidal_positional_encoding(
    context_size: int, embedding_size: int, padding_mask: torch.Tensor = None
) -> torch.Tensor:
    """Computes sinusoidal positional encodings for a given context size and embedding
    size.

    The positional encoding matrix is computed using the following formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/embedding_size))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/embedding_size))

    Args:
        context_size (int): The number of tokens in the context.
        embedding_size (int): The dimensionality of the token embeddings.

    Returns:
        A tensor of shape (context_size, embedding_size) containing the
        sinusoidal positional encoding matrix.

    Raises:
        ValueError: If the context size or embedding size is not a positive
            integer.
    """
    validation.validate_integer(context_size, "context_size", min_value=1)
    validation.validate_positive_even_integer(embedding_size, "embedding_size")

    positions = torch.arange(context_size, dtype=torch.float32)
    indices = torch.arange(embedding_size // 2, dtype=torch.float32)

    scaling_factor = 10_000 ** ((2 * indices) / embedding_size)
    angles = positions.unsqueeze(1) / scaling_factor
    pos_encoding = torch.zeros((context_size, embedding_size), dtype=torch.float32)
    pos_encoding[:, 0::2] = torch.sin(angles)
    pos_encoding[:, 1::2] = torch.cos(angles)
    return pos_encoding

    return pos_encoding


class MultiHeadAttention(nn.Module):
    """Performs the multi-head self attention operation on a batch of token embeddings.

    This layer implements the operation as described in the paper:
    https://arxiv.org/abs/1706.03762

    Args:
        num_heads (int): The number of attention heads.
        embedding_size (int): The dimensionality of the token embeddings.
        d_key (int): The dimensionality of the key vectors.
        d_value (int): The dimensionality of the value vectors.
        dropout (float): The dropout probability for the attention weights.
            Defaults to 0.0.

    Shape:
        - Input: `(batch_size, context_size, embedding_size)`
        - Outputs: Tuple[torch.Tensor, torch.Tensor]
            1. `(batch_size, context_size, embedding_size)`
            2. `(batch_size, num_heads, context_size, context_size)`

    Raises:
        ValueError: If any of the following are not positive integers: number of heads,
            embedding size, key dimensionality, or value dimensionality.
        ValueError: If the dropout probability is not a positive float or zero.
        ValueError: If the input tensor is not of shape (batch_size, context_size,
            embedding_size).

    Example:
        >>> import torch
        >>> from lumiere.components.attention import MultiHeadAttention
        >>> x = torch.randn(1, 3, 128)
        >>> attention = MultiHeadAttention(12, 128, 64, 64)
        >>> output, attention_weights = attention(x)
        >>> print(output.shape)
        torch.Size([1, 3, 128])
        >>> print(attention_weights.shape)
        torch.Size([1, 12, 3, 3])
    """

    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        d_key: int,
        d_value: int,
    ) -> None:
        super().__init__()

        validation.validate_integer(num_heads, "num_heads", min_value=1)
        validation.validate_integer(embedding_size, "embedding_size", min_value=1)
        validation.validate_integer(d_key, "d_key", min_value=1)
        validation.validate_integer(d_value, "d_value", min_value=1)

        self._d_key = d_key
        self._d_value = d_value
        self._num_heads = num_heads

        self._q_proj = nn.Linear(embedding_size, d_key * num_heads, bias=False)
        self._k_proj = nn.Linear(embedding_size, d_key * num_heads, bias=False)
        self._v_proj = nn.Linear(embedding_size, d_value * num_heads, bias=False)
        self._o_proj = nn.Linear(d_value * num_heads, embedding_size, bias=False)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                "The input tensor must have the shape (batch_size, context_size, "
                "embedding_size)"
            )

        queries = self._q_proj(x)
        keys = self._k_proj(x)
        values = self._v_proj(x)

        queries = split_heads(queries, self._num_heads, self._d_key)
        keys = split_heads(keys, self._num_heads, self._d_key)
        values = split_heads(values, self._num_heads, self._d_value)

        attention_scores = torch.matmul(queries, torch.transpose(keys, -2, -1))
        scaled_attention_scores = attention_scores / torch.sqrt(
            torch.tensor(self._d_key, dtype=queries.dtype, device=queries.device)
        )

        mask = create_causal_mask(queries.shape[2], padding_mask)

        masked_attention_scores = scaled_attention_scores.masked_fill(
            mask, -float("inf")
        )

        attention_weights = stable_softmax(masked_attention_scores)
        attention_values = torch.matmul(attention_weights, values)
        attention_values = concat_heads(attention_values)
        output = self._o_proj(attention_values)

        return output, attention_weights


def create_causal_mask(
    context_size: int, padding_mask: torch.Tensor = None
) -> torch.Tensor:
    """Creates a causal mask for the attention operation.

    This mask is used to prevent each token from attending to itself, future tokens,
    and padding tokens (if a padding mask is provided).

    Args:
        context_size: The size of the context.
        padding_mask: A mask indicating which tokens are padding.

    Returns:
        A mask of shape (1, 1, context_size, context_size).
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


def split_heads(
    tensor: torch.Tensor, num_heads: int, num_features: int
) -> torch.Tensor:
    """Splits the concatenated multi-head features into separate heads.

    Args:
        tensor: Tensor to split
        num_heads: Number of attention heads
        num_features: Number of features per attention head

    Shape:
        - Input: `(batch_size, context_size, num_heads * num_dimensions)`
        - Output: `(batch_size, num_heads, context_size, num_dimensions)`

    Returns:
        Tensor of shape (batch_size, num_heads, context_size, num_dimensions)

    Example:
        >>> import torch
        >>> tensor = torch.randn(1, 3, 12)
        >>> output = split_heads(tensor, 2, 6)
        >>> print(output.shape)
        torch.Size([1, 2, 3, 6])
    """
    return tensor.view(
        tensor.shape[0], tensor.shape[1], num_heads, num_features
    ).transpose(1, 2)


def concat_heads(tensor: torch.Tensor) -> torch.Tensor:
    """Concatenates the tensor from multiple attention heads.

    Args:
        tensor: Tensor to concatenate

    Shape:
        - Input: `(batch_size, num_heads, context_size, num_dimensions)`
        - Output: `(batch_size, context_size, num_heads * num_dimensions)`

    Returns:
        Tensor of shape (batch_size, context_size, num_heads * num_dimensions)

    Example:
        >>> import torch
        >>> tensor = torch.randn(1, 2, 3, 4)
        >>> output = concat_heads(tensor)
        >>> print(output.shape)
        torch.Size([1, 3, 12])
    """
    return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], -1)


def stable_softmax(x: torch.Tensor) -> torch.Tensor:
    """Applies softmax with numerical stability.

    This is made to explicity handle the case where the values along the specified
    dimension are all negative infinity. In this case, the softmax will return all
    zeros instead of NaN.

    Args:
        x: Input tensor

    Returns:
        Tensor of the same shape as the input tensor
    """
    return torch.exp(x) / (torch.sum(torch.exp(x), dim=-1, keepdim=True) + 1e-9)


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

    Raises:
        ValueError: If any of the following conditions are met:
            - The embedding size is not a positive integer.
            - The hidden size is not a positive integer.
            - The dropout probability is not a valid probability.
            - The input tensor has less than 2 dimensions.
            - The input tensor's embedding size does not match the configured
              embedding size.
    """

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        validation.validate_integer(embedding_size, "embedding_size", min_value=1)
        validation.validate_integer(hidden_size, "hidden_size", min_value=1)
        validation.validate_probability(dropout, "dropout")

        self._embedding_size = embedding_size
        self._hidden_size = hidden_size
        self._dropout_p = dropout

        self._gate_proj = nn.Linear(self._embedding_size, self._hidden_size, bias=False)
        self._up_proj = nn.Linear(self._embedding_size, self._hidden_size, bias=False)
        self._down_proj = nn.Linear(self._hidden_size, self._embedding_size, bias=False)
        self._dropout = nn.Dropout(self._dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError(
                "Expected input tensor to have at least 2 dimensions,"
                f"but got {x.dim()}."
            )
        if x.size(-1) != self._embedding_size:
            raise ValueError(
                f"Expected input tensor to have embedding size {self._embedding_size}, "
                f"but got {x.size(-1)}."
            )

        gate = F.silu(self._gate_proj(x))
        up = self._up_proj(x)
        hidden = gate * up
        down = self._down_proj(hidden)
        return self._dropout(down)


class TransformerBlock(nn.Module):
    """Applies a transformer block over a batch of token embeddings.

    This transformer block applies the following operations in sequence:
    1. RMS normalization → Multi-head attention → Dropout → Residual connection
    2. RMS normalization → Feed-forward network → Dropout → Residual connection

    Args:
        embedding_size (int): The dimensionality of the token embeddings.
        num_heads (int): The number of attention heads.
        d_key (int): The dimensionality of the key vectors in the multi-head
            attention layer.
        d_value (int): The dimensionality of the value vectors in the multi-head
            attention layer.
        d_ff (int): The dimensionality of the feed-forward network's hidden
            representations.
        dropout (float): The dropout probability. Defaults to 0.1.

    Shape:
        - Input: `(batch_size, context_size, embedding_size)`
        - Output: `Tuple[torch.Tensor, torch.Tensor]`
            1. `(batch_size, context_size, embedding_size)`
            2. `(batch_size, num_heads, context_size, context_size)`

    Example:
        >>> import torch
        >>> from lumiere.components import TransformerBlock
        >>> x = torch.randn(1, 10, 128)
        >>> block = TransformerBlock(128, 12, 64, 64, 256)
        >>> output, attention_weights = block(x)
        >>> print(output.shape)
        torch.Size([1, 10, 128])
        >>> print(attention_weights.shape)
        torch.Size([1, 12, 10, 10])
    """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        d_key: int,
        d_value: int,
        d_ff: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
        post_norm: bool = False,
    ) -> None:
        super().__init__()

        validation.validate_integer(embedding_size, "embedding_size", min_value=1)
        validation.validate_integer(num_heads, "num_heads", min_value=1)
        validation.validate_integer(d_key, "d_key", min_value=1)
        validation.validate_integer(d_value, "d_value", min_value=1)
        validation.validate_integer(d_ff, "d_ff", min_value=1)
        validation.validate_probability(dropout, "dropout")
        validation.validate_boolean(pre_norm, "pre_norm")
        validation.validate_boolean(post_norm, "post_norm")

        self._embedding_size = embedding_size
        self._num_heads = num_heads
        self._d_key = d_key
        self._d_value = d_value
        self._d_ff = d_ff
        self._dropout = dropout
        self._pre_norm = pre_norm
        self._post_norm = post_norm

        if self._pre_norm:
            self.normalization_1 = nn.RMSNorm(self._embedding_size)

        self.attention = MultiHeadAttention(
            num_heads=self._num_heads,
            embedding_size=self._embedding_size,
            d_key=self._d_key,
            d_value=self._d_value,
        )

        if self._pre_norm or self._post_norm:
            self.normalization_2 = nn.RMSNorm(self._embedding_size)

        self.feedforward = FeedForward(
            self._embedding_size, self._d_ff, dropout=self._dropout
        )
        self.dropout = nn.Dropout(self._dropout)

        if self._post_norm:
            self.normalization_3 = nn.RMSNorm(self._embedding_size)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(
                "Input tensor must have shape (batch_size, context_size, "
                "embedding_size)"
            )
        if x.size(-1) != self._embedding_size:
            raise ValueError(
                f"Expected embedding_size={self._embedding_size}, got {x.size(-1)}"
            )

        # Compute the attention values and weights.
        if hasattr(self, "normalization_1"):
            x = self.normalization_1(x)

        attention_values, attention_weights = self.attention(
            x, padding_mask=padding_mask
        )
        x = x + self.dropout(attention_values)

        # Compute the feed-forward output.
        if hasattr(self, "normalization_2"):
            x = self.normalization_2(x)

        feedforward_out = self.feedforward(x)
        x = x + self.dropout(feedforward_out)

        if hasattr(self, "normalization_3"):
            x = self.normalization_3(x)

        return x, attention_weights


class Lumiere1M(nn.Module):
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
        d_ff: int,
        dropout: float = 0.1,
        padding_id: int | None = None,
        pre_norm: bool = True,
        post_norm: bool = False,
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
        self._pre_norm = pre_norm
        self._post_norm = post_norm

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
                    d_ff=self._d_ff,
                    dropout=self._dropout,
                    pre_norm=self._pre_norm,
                    post_norm=self._post_norm,
                )
                for _ in range(self._num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(self._embedding_size)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
