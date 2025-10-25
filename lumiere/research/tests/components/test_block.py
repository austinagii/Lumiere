import pytest
import torch

from lumiere.research.src.components import TransformerBlock


def _is_layer_normalized(tensor: torch.Tensor) -> bool:
    """Return whether the specified tensor has been layer normalized"""
    sum = tensor.sum(axis=-1, keepdim=True)
    is_zero_centered = torch.allclose(sum, torch.zeros_like(sum), atol=1e-05)

    mean_diff = tensor - tensor.mean(axis=-1, keepdim=True)
    sum_squares = torch.sum(mean_diff**2, axis=-1, keepdim=True)
    variance = torch.sqrt(sum_squares / tensor.size(-1))
    is_unit_variance = torch.allclose(variance, torch.ones_like(variance), atol=1e-03)

    return is_zero_centered and is_unit_variance


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

    def factory(
        pre_norm: bool, post_norm: bool, norm_type: str = "rms"
    ) -> TransformerBlock:
        return TransformerBlock(
            embedding_size=16,
            num_heads=1,
            d_key=16,
            d_value=16,
            d_ff=16,
            pre_norm=pre_norm,
            post_norm=post_norm,
            norm_type=norm_type,
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
        block.register_forward_hook(_capture_input(intermediate_tensors, "block_input"))
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

    # ==============================================
    # =======##=== TEST NORMALIZATION ==============
    # ==============================================
    @pytest.mark.parametrize(
        "norm_type, validation_fn",
        [("rms", _is_rms_normalized), ("layer", _is_layer_normalized)],
    )
    def test_specified_normalization_type_is_applied(
        self, transformer_block_factory, norm_type, validation_fn
    ):
        block = transformer_block_factory(
            pre_norm=True, post_norm=False, norm_type=norm_type
        )
        intermediate_tensors = {}

        block.attention.register_forward_hook(
            _capture_input(intermediate_tensors, "attention")
        )

        batch_size, seq_length, embedding_size = 3, 8, 16
        block(torch.rand(batch_size, seq_length, embedding_size))

        assert validation_fn(intermediate_tensors["attention"])

    def test_rms_normalization_is_used_by_default(self):
        block = TransformerBlock(
            embedding_size=16,
            num_heads=2,
            d_key=8,
            d_ff=8,
            d_value=8,
            pre_norm=True,
            post_norm=True,
        )
        intermediate_tensors = {}

        block.attention.register_forward_hook(
            _capture_input(intermediate_tensors, "attention_input")
        )
        block.feedforward.register_forward_hook(
            _capture_input(intermediate_tensors, "attention_input")
        )
        block.register_forward_hook(
            _capture_output(intermediate_tensors, "attention_input")
        )

        assert all([_is_rms_normalized(t) for t in intermediate_tensors.values()])

    @pytest.mark.parametrize("norm_type", ["test", "", "   "])
    def test_error_is_raised_if_norm_type_is_invalid(
        self, transformer_block_factory, norm_type
    ):
        with pytest.raises(ValueError):
            TransformerBlock(
                embedding_size=2,
                num_heads=1,
                d_key=4,
                d_value=4,
                d_ff=8,
                norm_type=norm_type,
            )

    @pytest.mark.parametrize(
        "norm_type", [1, 1.0, set("rms"), {"scheme": "rms"}, True, False, ["rms"]]
    )
    def test_error_is_raised_if_norm_type_is_not_string(self, norm_type):
        with pytest.raises(TypeError):
            TransformerBlock(
                embedding_size=2,
                num_heads=1,
                d_key=4,
                d_value=4,
                d_ff=8,
                norm_type=norm_type,
            )
