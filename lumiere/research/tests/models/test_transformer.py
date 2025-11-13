import pytest
import torch
from torch.nn import RMSNorm

from lumiere.research.src.components.feedforward import LinearFeedForward
from lumiere.research.src.models.transformer import Transformer


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
            embedding_size=embedding_size,
            context_size=context_size,
            num_layers=2,
            num_heads=4,
            d_key=16,
            d_value=16,
            feedforward_factory=lambda: LinearFeedForward(embedding_size, 16),
            normalization_factory=lambda: RMSNorm(embedding_size),
            dropout=0.0,
            padding_id=0,
        )

        # Create random token IDs
        token_ids = torch.randint(
            0, vocab_size, (batch_size, context_size), dtype=torch.int
        )

        # Forward pass
        output, _ = model(token_ids)

        # Check shape
        assert output.shape == (batch_size, context_size, vocab_size)
