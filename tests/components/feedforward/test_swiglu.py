import pytest
import torch

from lumiere.nn.components.feedforward.swiglu import SwigluFeedForward


def _capture_inputs(storage, key):
    def _capture(module, args, outputs):
        storage[key] = args[0]

    return _capture


class TestSwigluFeedForward:
    @pytest.mark.parametrize(
        "embedding_size,hidden_size,dropout",
        [
            (128, 64, 0.3),
            (256, 128, 0.5),
            (512, 256, 0.5),
        ],
    )
    def test_init_initializes_network_successfully(
        self, embedding_size, hidden_size, dropout
    ):
        torch.manual_seed(42)
        batch_size = 16
        context_size = 32

        intermediate_tensors = {}
        ff = SwigluFeedForward(embedding_size, hidden_size, dropout=dropout)
        ff.down_proj.register_forward_hook(
            _capture_inputs(intermediate_tensors, "down_proj_input")
        )

        out = ff(torch.rand(batch_size, context_size, embedding_size))
        down_proj_input = intermediate_tensors.get("down_proj_input")

        assert ff.gate_proj is not None
        assert ff.up_proj is not None
        assert ff.down_proj is not None
        assert ff.embedding_size == embedding_size
        assert ff.hidden_size == hidden_size
        assert ff.dropout == dropout

        assert down_proj_input is not None
        assert down_proj_input.shape == (batch_size, context_size, hidden_size)

        assert out is not None
        assert out.shape == (batch_size, context_size, embedding_size)
        actual_dropout = (out == 0).sum() / out.numel()
        assert torch.allclose(actual_dropout, torch.tensor(dropout), atol=0.01)

    @pytest.mark.parametrize(
        "embedding_size,hidden_size,dropout,expected_error",
        [(0, 1, 0.1, ValueError), (2, 0, 0.1, ValueError), (2, 2, None, TypeError)],
    )
    def test_init_returns_error_for_invalid_params(
        self, embedding_size, hidden_size, dropout, expected_error
    ):
        with pytest.raises(expected_error):
            SwigluFeedForward(embedding_size, hidden_size, dropout=dropout)

    def test_forward_returns_error_if_input_tensor_is_invalid(self):
        batch_size = 1
        context_size = 2
        embedding_size = 4
        hidden_size = 4

        ff = SwigluFeedForward(embedding_size, hidden_size)

        with pytest.raises(AssertionError):
            ff(torch.rand(batch_size, context_size, 6))
