import pytest
import torch

from lumiere.nn.components.feedforward.linear import LinearFeedForward


def _capture_inputs(storage, key):
    def _capture(module, args, outputs):
        storage[key] = args[0]

    return _capture


class TestLinearFeedForward:
    @pytest.mark.parametrize(
        "embedding_size,d_ff,dropout",
        [
            (128, 64, 0.3),
            (256, 128, 0.5),
            (512, 256, 0.5),
        ],
    )
    def test_init_initializes_network_successfully(self, embedding_size, d_ff, dropout):
        torch.manual_seed(42)
        batch_size = 16
        context_size = 32

        intermediate_tensors = {}
        ff = LinearFeedForward(embedding_size, d_ff, dropout=dropout)
        ff.down_proj.register_forward_hook(
            _capture_inputs(intermediate_tensors, "down_proj_input")
        )

        out = ff(torch.rand(batch_size, context_size, embedding_size))
        down_proj_input = intermediate_tensors.get("down_proj_input")

        assert ff.up_proj is not None
        assert ff.down_proj is not None
        assert ff.embedding_size == embedding_size
        assert ff.d_ff == d_ff
        assert ff.dropout == dropout

        assert down_proj_input is not None
        assert down_proj_input.shape == (batch_size, context_size, d_ff)
        actual_dropout = (down_proj_input == 0).sum() / down_proj_input.numel()
        assert torch.allclose(actual_dropout, torch.tensor(dropout), atol=0.01)

        assert out is not None
        assert out.shape == (batch_size, context_size, embedding_size)

    @pytest.mark.parametrize(
        "embedding_size,d_ff,dropout,expected_error",
        [(0, 1, 0.1, ValueError), (2, 0, 0.1, ValueError), (2, 2, None, TypeError)],
    )
    def test_init_returns_error_for_invalid_params(
        self, embedding_size, d_ff, dropout, expected_error
    ):
        with pytest.raises(expected_error):
            LinearFeedForward(embedding_size, d_ff, dropout=dropout)

    def test_forward_returns_error_if_input_tensor_is_invalid(self):
        batch_size = 1
        context_size = 2
        embedding_size = 4
        d_ff = 4

        ff = LinearFeedForward(embedding_size, d_ff)

        with pytest.raises(AssertionError):
            ff(torch.rand(batch_size, context_size, 6))
