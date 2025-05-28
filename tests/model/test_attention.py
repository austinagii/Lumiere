import pytest
import torch

from lumiere.core.attention import MultiHeadAttention


class TestMultiHeadAttention:
    INVALID_INTEGER_VALUES = [-1, 0, "1", 1.0, None, (1,)]

    @pytest.mark.parametrize("num_heads", INVALID_INTEGER_VALUES)
    def test_error_is_raised_if_num_heads_is_not_positive_integer(self, num_heads):
        with pytest.raises(ValueError, match="num_heads must be a positive integer"):
            MultiHeadAttention(num_heads=num_heads,
                               embedding_size=4, d_key=2, d_value=2)

    @pytest.mark.parametrize("embedding_size", INVALID_INTEGER_VALUES)
    def test_error_is_raised_if_embedding_size_is_not_positive_integer(self, embedding_size):
        with pytest.raises(ValueError, match="embedding_size must be a positive integer"):
            MultiHeadAttention(
                num_heads=2, embedding_size=embedding_size, d_key=2, d_value=2)

    @pytest.mark.parametrize("d_key", INVALID_INTEGER_VALUES)
    def test_error_is_raised_if_d_key_is_not_positive_integer(self, d_key):
        with pytest.raises(ValueError, match="d_key must be a positive integer"):
            MultiHeadAttention(num_heads=2, embedding_size=4,
                               d_key=d_key, d_value=2)

    @pytest.mark.parametrize("d_value", INVALID_INTEGER_VALUES)
    def test_error_is_raised_if_d_value_is_not_positive_integer(self, d_value):
        with pytest.raises(ValueError, match="d_value must be a positive integer"):
            MultiHeadAttention(num_heads=2, embedding_size=4,
                               d_key=2, d_value=d_value)

    @pytest.mark.parametrize("masked", ["True", 1, None, "masked"])
    def test_error_is_raised_if_masked_is_not_boolean(self, masked):
        with pytest.raises(ValueError, match="masked must be a boolean"):
            MultiHeadAttention(num_heads=2, embedding_size=4,
                               d_key=2, d_value=2, masked=masked)

    def test_masked_is_initialized_to_true_by_default(self):
        attention = MultiHeadAttention(
            num_heads=2, embedding_size=4, d_key=2, d_value=2)
        assert attention.masked == True

    def test_attention_scores_are_calculated_correctly(self):
        x = torch.arange(0, 1.6, step=0.1, dtype=torch.float32).view(1, 4, 4)
        q_proj = torch.arange(0, 2.4, step=0.05, dtype=torch.float32).view(4, 12)
        k_proj = torch.arange(0.5, 2.9, step=0.05, dtype=torch.float32).view(4, 12)
        v_proj = torch.arange(1.0, 4.2, step=0.05, dtype=torch.float32).view(4, 16)
        o_proj = torch.arange(0, 3.2, step=0.05, dtype=torch.float32).reshape(16, 4)
        
        attention = MultiHeadAttention(num_heads=2, embedding_size=4, d_key=6, d_value=8)
        attention._q_proj.data.copy_(q_proj)
        attention._k_proj.data.copy_(k_proj)
        attention._v_proj.data.copy_(v_proj)
        attention._o_proj.data.copy_(o_proj)

        expected = torch.tensor([[[ 48.7200,  50.3760,  52.0320,  53.6880],
                                  [153.4400, 158.3920, 163.3440, 168.2960],
                                  [258.1600, 266.4080, 274.6560, 282.9040],
                                  [362.8800, 374.4240, 385.9680, 397.5120]]])
        actual = attention(x)
        assert torch.allclose(actual, expected, atol=1e-4)