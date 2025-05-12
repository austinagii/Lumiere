import torch
import pytest

from prism.model import Model


class TestModel:
    @pytest.mark.parametrize("vocab_size", [100])
    @pytest.mark.parametrize("embedding_size", [64])
    @pytest.mark.parametrize("context_size", [16])
    @pytest.mark.parametrize("batch_size", [2])
    def test_model_forward(
        self,
        vocab_size: int,
        embedding_size: int,
        context_size: int,
        batch_size: int
    ) -> None:
        model = Model(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            context_size=context_size
        )
        
        # Create random token IDs
        token_ids = torch.randint(
            0, vocab_size, (batch_size, context_size), dtype=torch.int)
        
        # Forward pass
        output = model(token_ids)
        
        # Check shape
        assert output.shape == (batch_size, context_size, embedding_size)
