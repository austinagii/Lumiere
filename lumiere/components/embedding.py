import torch
from torch import nn

from lumiere.preprocessing.tokenizer import SPECIAL_TOKENS
from lumiere.utils import validation


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
    ) -> None:
        super().__init__()
        validation.validate_positive_integer(vocab_size, "vocab_size")
        validation.validate_positive_integer(context_size, "context_size")
        validation.validate_positive_integer(embedding_size, "embedding_size")

        self._vocab_size = vocab_size
        self._context_size = context_size
        self._embedding_size = embedding_size

        self._embedding = nn.Embedding(self._vocab_size, self._embedding_size)
        positional_encoding = sinusoidal_positional_encoding(
            self._context_size, self._embedding_size
        )
        self.register_buffer("_positional_encoding", positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.any(x < 0) or torch.any(x >= self._vocab_size):
            raise IndexError("Token ids are outside of the range [0, vocab_size).")

        token_embeddings = self._embedding(x)

        # Add the positional encoding for the context size.
        position_encoding = self._positional_encoding[: x.shape[-1], :]
        token_embeddings += position_encoding

        # Zero the embedding of padding tokens.
        token_embeddings[x == SPECIAL_TOKENS["padding"].id] = 0

        return token_embeddings

    @property
    def vocab_size(self) -> int:
        """The number of unique tokens in the vocabulary."""
        return self._vocab_size

    @property
    def embedding_size(self) -> int:
        """The dimensionality of the token embeddings."""
        return self._embedding_size


def sinusoidal_positional_encoding(
    context_size: int, embedding_size: int
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
    validation.validate_positive_integer(context_size, "context_size")
    validation.validate_positive_even_integer(embedding_size, "embedding_size")

    positions = torch.arange(context_size, dtype=torch.float32)
    indices = torch.arange(embedding_size // 2, dtype=torch.float32)

    scaling_factor = 10_000 ** ((2 * indices) / embedding_size)
    angles = positions.unsqueeze(1) / scaling_factor
    pos_encoding = torch.zeros((context_size, embedding_size), dtype=torch.float32)
    pos_encoding[:, 0::2] = torch.sin(angles)
    pos_encoding[:, 1::2] = torch.cos(angles)
    return pos_encoding
