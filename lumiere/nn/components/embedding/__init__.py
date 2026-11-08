from .patch import PatchEmbedding
from .sinusoidal import (
    SinusoidalPositionalEmbedding,
    sinusoidal_positional_encoding,
)


# Alias for tests
Embedding = SinusoidalPositionalEmbedding

__all__ = [
    "Embedding",
    "PatchEmbedding",
    "SinusoidalPositionalEmbedding",
    "sinusoidal_positional_encoding",
]
