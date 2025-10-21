import pytest
import torch

from lumiere.research.src.components.feedforward import FeedForward


class TestFeedForward:
    @pytest.fixture
    def feedforward(self):
        return FeedForward(embedding_size=10, hidden_size=10, dropout=0.1)

    @pytest.mark.parametrize(
        "embedding_size, hidden_size, dropout",
        [
            (-1, 10, 0.1),
            (0, 10, 0.1),
            (10, -1, 0.1),
            (10, 0, 0.1),
            (10, 10, -0.1),
            (10, 10, 1.1),
        ],
    )
    def test_raises_value_error_if_arguments_are_invalid(
        self, embedding_size, hidden_size, dropout
    ):
        with pytest.raises(ValueError):
            FeedForward(embedding_size, hidden_size, dropout)

    @pytest.mark.parametrize("input_shape", [(), (10)])
    def test_raises_value_error_if_input_has_less_than_2_dimensions(
        self, feedforward, input_shape
    ):
        with pytest.raises(ValueError):
            feedforward(torch.randn(input_shape))

    # TODO: Add tests for the forward pass.
