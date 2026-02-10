from .sinusoidal import (
    SinusoidalPositionalEmbedding,
    sinusoidal_positional_encoding,
)

# Alias for tests
Embedding = SinusoidalPositionalEmbedding

__all__ = [
    "Embedding",
    "SinusoidalPositionalEmbedding",
    "sinusoidal_positional_encoding",
]
