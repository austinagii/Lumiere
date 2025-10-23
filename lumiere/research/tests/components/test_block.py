import pytest
import torch

from lumiere.research.src.components import TransformerBlock


def _is_rms_normalized(t):
    rms = torch.sqrt(torch.mean(t**2, dim=-1, keepdim=True))
    return torch.allclose(rms, torch.ones_like(rms))


def _capture_input(captured_dict, key):
    """Returns a hook that capture the input tensor into a dict."""

    def hook(module, args):
        captured_dict[key] = args[0]

    return hook


def _capture_output(captured_dict, key):
    """Returns a hook that capture the output tensor into a dict."""

    def hook(module, args, output):
        captured_dict[key] = output[0]

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
        captured = {}

        # Register hooks to capture tensors at key points
        block.register_forward_pre_hook(_capture_input(captured, "block_input"))
        block.attention.register_forward_pre_hook(
            _capture_input(captured, "attention_input")
        )
        block.feedforward.register_forward_pre_hook(
            _capture_input(captured, "feedforward_input")
        )
        block.register_forward_hook(_capture_output(captured, "block_output"))

        block(torch.rand(3, 8, 16))

        # Assert normalization state at each point.
        for location, should_be_normalized in expected_normalized.items():
            is_normalized = _is_rms_normalized(captured[location])
            if should_be_normalized:
                assert is_normalized, f"{location} should be RMS normalized."
            else:
                assert not is_normalized, f"{location} should NOT be RMS normalized."
