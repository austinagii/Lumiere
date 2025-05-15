import pytest
import torch

from prism.attention import MultiHeadAttention


class TestMultiHeadAttention:
    INVALID_INTEGER_VALUES = [-1, 0, "1", 1.0, None, (1,)]
    
    @pytest.mark.parametrize("num_heads", INVALID_INTEGER_VALUES)
    def test_error_is_raised_if_num_heads_is_not_positive_integer(self, num_heads):
        with pytest.raises(ValueError, match="num_heads must be a positive integer"):
            MultiHeadAttention(num_heads=num_heads, embedding_size=4, d_key=2, d_value=2)

    @pytest.mark.parametrize("embedding_size", INVALID_INTEGER_VALUES)
    def test_error_is_raised_if_embedding_size_is_not_positive_integer(self, embedding_size):
        with pytest.raises(ValueError, match="embedding_size must be a positive integer"):
            MultiHeadAttention(num_heads=2, embedding_size=embedding_size, d_key=2, d_value=2)

    @pytest.mark.parametrize("d_key", INVALID_INTEGER_VALUES)
    def test_error_is_raised_if_d_key_is_not_positive_integer(self, d_key):
        with pytest.raises(ValueError, match="d_key must be a positive integer"):
            MultiHeadAttention(num_heads=2, embedding_size=4, d_key=d_key, d_value=2)
            
    @pytest.mark.parametrize("d_value", INVALID_INTEGER_VALUES)
    def test_error_is_raised_if_d_value_is_not_positive_integer(self, d_value):
        with pytest.raises(ValueError, match="d_value must be a positive integer"):
            MultiHeadAttention(num_heads=2, embedding_size=4, d_key=2, d_value=d_value)
    
    @pytest.mark.parametrize("masked", ["True", 1, None, "masked"])
    def test_error_is_raised_if_masked_is_not_boolean(self, masked):
        with pytest.raises(ValueError, match="masked must be a boolean"):
            MultiHeadAttention(num_heads=2, embedding_size=4, d_key=2, d_value=2, masked=masked)
    
    def test_masked_is_initialized_to_true_by_default(self):
        attention = MultiHeadAttention(num_heads=2, embedding_size=4, d_key=2, d_value=2)
        assert attention.masked == True