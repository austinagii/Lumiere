import torch
import pytest

from prism.model import sinusoidal_positional_encoding


class TestPositionalEncoding:

    @pytest.mark.parametrize("shape", [
        # Invalid types
        ("1", "1"), ("1", 1), (1, "1"), (1.0, 0.999),
        # Negative values
        (-1, -1), (-1, 1), (1, -1),
        # Zero values
        (0, 0), (0, 1), (1, 0),
        # None values
        None, (None, 1), (1, None),
        # Wrong types/length
        (), (1), ((1, 1)), (1, 1, 1)
    ])
    def test_value_error_is_raised_if_input_shape_is_invalid(self, shape: tuple[int, int]) -> None:
        with pytest.raises(ValueError):
            sinusoidal_positional_encoding(shape)

    @pytest.mark.parametrize("context_size", [1, 16, 32])
    @pytest.mark.parametrize("embedding_size", [2, 64, 128])
    def test_positional_encoding_has_correct_shape_and_dtype(
        self,
        context_size: int,
        embedding_size: int
    ) -> None:
        positional_encoding = sinusoidal_positional_encoding(
            (context_size, embedding_size))
        assert positional_encoding.shape == (context_size, embedding_size)
        assert positional_encoding.dtype == torch.float32

    @pytest.mark.parametrize(("shape", "expected_positional_encoding"), [
        ((1, 2), torch.tensor([[0.0000, 1.0000]], dtype=torch.float32)),
        ((2, 2), torch.tensor([[0.0000, 1.0000], [
         0.8414, 0.5403]], dtype=torch.float32)),
        ((1, 4), torch.tensor(
            [[0.0000, 1.0000, 0.0000, 1.0000]], dtype=torch.float32)),
        ((2, 4), torch.tensor([[0.0000, 1.0000, 0.0000, 1.0000],
                               [0.8414, 0.5403, 0.0099, 0.9999]], dtype=torch.float32)),
    ])
    def test_positional_encoding_matrix_has_correct_values(
        self,
        shape: tuple[int, int],
        expected_positional_encoding: torch.Tensor
    ) -> None:
        actual_positional_encoding = sinusoidal_positional_encoding(shape)
        assert torch.allclose(actual_positional_encoding,
                              expected_positional_encoding, atol=1e-4)