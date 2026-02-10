import pytest
import torch
from torch.nn import RMSNorm

from lumiere.nn.components.attention import MultiHeadAttention
from lumiere.nn.components.embedding import SinusoidalPositionalEmbedding as Embedding
from lumiere.nn.components.feedforward import LinearFeedForward
from lumiere.nn.architectures.transformer import Transformer
from lumiere.nn.components.blocks import StandardTransformerBlock


class TestModel:
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("embedding_size", [64])
    @pytest.mark.parametrize("context_size", [16])
    @pytest.mark.parametrize("batch_size", [2])
    def test_model_forward(
        self, vocab_size: int, embedding_size: int, context_size: int, batch_size: int
    ) -> None:
        model = Transformer(
            vocab_size=vocab_size,
            context_size=context_size,
            num_blocks=2,
            embedding=lambda: Embedding(
                vocab_size=vocab_size,
                context_size=context_size,
                embedding_size=embedding_size,
                padding_id=0,
            ),
            block=lambda: StandardTransformerBlock(
                attention=lambda: MultiHeadAttention(
                    num_heads=4, embedding_size=embedding_size, d_key=16, d_value=16
                ),
                feedforward=lambda: LinearFeedForward(embedding_size, 16),
                normalization=lambda: RMSNorm(embedding_size),
                dropout=0.0,
                pre_norm=True,
                post_norm=False,
            ),
            normalization=lambda: RMSNorm(embedding_size),
        )

        # Create random token IDs
        token_ids = torch.randint(
            0, vocab_size, (batch_size, context_size), dtype=torch.int
        )

        # Forward pass
        output, _ = model(token_ids)

        # Check shape
        assert output.shape == (batch_size, context_size, vocab_size)
