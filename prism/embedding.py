import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, context_size: int, embedding_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoding = sinusoidal_positional_encoding(
            (context_size, embedding_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the embedding for the given token ids.

        `x` is expected to be a tensor of shape (..., [batch_size], context_size)
        where all elements in the last dimension are integers in the range [0, vocab_size) 
        representing token ids.

        Args:
            x: A tensor of shape (..., [batch_size], context_size) containing the 
               token ids.

        Returns:
            A tensor of shape (..., [batch_size], context_size, embedding_size) 
            containing the token embeddings.

        Raises:
            IndexError: If the token ids are outside of the range [0, vocab_size).
        """
        if torch.any(x < 0) or torch.any(x >= self.embedding.num_embeddings):
            raise IndexError("Token ids are outside of the range [0, vocab_size).")
        return self.embedding(x) + self.positional_encoding
    

def sinusoidal_positional_encoding(shape: tuple[int, int]) -> torch.Tensor:
    """Returns the sinusoidal positional encoding matrix with the given shape.

    The positional encoding matrix has shape (context_size, embedding_size) 
    and is computed using the following formula:

        PE(pos, 2i)   = sin(pos / 10000^(2i/embedding_size))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/embedding_size))

    Where pos is the position of the token in the context and i is the 
    index of the pair of dimensions under consideration.

    Shape is expected to be a pair of positive integers where the second 
    integer is even.

    Args:
        shape: A tuple of (context_size, embedding_size).

    Returns:
        A tensor of shape (context_size, embedding_size) containing the 
        sinusoidal positional encoding matrix.

    Raises:
        ValueError: If the specified shape is invalid.
    """
    if not (isinstance(shape, tuple) and len(shape) == 2):
        raise ValueError("Shape must be a tuple (context_size, embedding_size).")

    context_size, embedding_size = shape
    if not (isinstance(context_size, int) and context_size > 0):
        raise ValueError("Context size must be a positive integer.")
    if not (isinstance(embedding_size, int) and embedding_size > 0 and embedding_size % 2 == 0):
        raise ValueError("Embedding size must be a positive, even integer.")

    positions = torch.arange(context_size, dtype=torch.float32)
    indices = torch.arange(embedding_size / 2, dtype=torch.float32)

    scaling_factor = 10_000 ** ((2 * indices) / embedding_size)
    angles = positions.unsqueeze(1) / scaling_factor
    pos_encoding = torch.zeros((context_size, embedding_size), dtype=torch.float32)
    pos_encoding[:, 0::2] = torch.sin(angles)
    pos_encoding[:, 1::2] = torch.cos(angles)
    return pos_encoding