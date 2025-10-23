import pytest
import torch

from lumiere.research.src.components import TransformerBlock


def _is_rms_normalized(tensor):
    rms = torch.sqrt(torch.mean(tensor**2, dim=-1, keepdim=True))
    return torch.allclose(rms, torch.ones_like(rms))


def _capture_input(storage, key):
    """Returns a hook that captures the input tensor into a dict."""

    def hook(module, input, output):
        storage[key] = input[0]

    return hook


def _capture_output(storage, key):
    """Returns a hook that captures the output tensor into a dict."""

    def hook(module, input, output):
        storage[key] = output[0]

    return hook


@pytest.fixture
def transformer_block_factory():
    """Create a transformer block with preconfigured sizes."""

    def factory(pre_norm: bool, post_norm: bool) -> TransformerBlock:
        return TransformerBlock(
            embedding_size=16,
            num_heads=1,
            d_key=16,
            d_value=16,
            d_ff=16,
            pre_norm=pre_norm,
            post_norm=post_norm,
        )

    return factory


class TestTransformerBlock:
    @pytest.mark.parametrize(
        "pre_norm, post_norm, expected_normalized",
        [
            # (pre_norm, post_norm, {location: should_be_normalized})
            (
                True,
                False,
                {
                    "block_input": False,
                    "attention_input": True,
                    "feedforward_input": True,
                    "block_output": False,
                },
            ),
            (
                False,
                True,
                {
                    "block_input": False,
                    "attention_input": False,
                    "feedforward_input": True,
                    "block_output": True,
                },
            ),
            (
                False,
                False,
                {
                    "block_input": False,
                    "attention_input": False,
                    "feedforward_input": False,
                    "block_output": False,
                },
            ),
        ],
    )
    def test_normalization_configuration(
        self, transformer_block_factory, pre_norm, post_norm, expected_normalized
    ):
        block = transformer_block_factory(pre_norm=pre_norm, post_norm=post_norm)
        intermediate_tensors = {}

        # Register hooks to capture tensors at key points
        block.register_forward_hook(
            _capture_input(intermediate_tensors, "block_input")
        )
        block.attention.register_forward_hook(
            _capture_input(intermediate_tensors, "attention_input")
        )
        block.feedforward.register_forward_hook(
            _capture_input(intermediate_tensors, "feedforward_input")
        )
        block.register_forward_hook(
            _capture_output(intermediate_tensors, "block_output")
        )

        batch_size, seq_length, embedding_size = 3, 8, 16
        block(torch.rand(batch_size, seq_length, embedding_size))

        # Assert normalization state at each point.
        for location, should_be_normalized in expected_normalized.items():
            is_normalized = _is_rms_normalized(intermediate_tensors[location])
            if should_be_normalized:
                assert is_normalized, f"{location} should be RMS normalized."
            else:
                assert not is_normalized, f"{location} should NOT be RMS normalized."
