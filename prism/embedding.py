import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the embedding for the given token ids.

        `x` is expected to be a tensor of shape (..., batch_size) where each 
        element is an integer in the range [0, vocab_size) representing a token id. 
        If `x` has only one dimension, the batch size is assumed to be 1.

        Args:
            x: A tensor of shape (..., batch_size) containing the token ids.

        Returns:
            A tensor of shape (..., batch_size, embedding_size) containing the token embeddings.

        Raises:
            IndexError: If the token ids are outside of the range [0, vocab_size).
        """
        if torch.any(x < 0) or torch.any(x >= self.embedding.num_embeddings):
            raise IndexError("Token ids are outside of the range [0, vocab_size).")
        return self.embedding(x) 