import pytest
import torch
from torch import nn

from lumiere.nn.components.embedding import (
    Embedding,
    sinusoidal_positional_encoding,
)


class TestEmbedding:
    @pytest.mark.parametrize("vocab_size", [1, 16, 32])
    @pytest.mark.parametrize("context_size", [1, 16, 32])
    @pytest.mark.parametrize("embedding_size", [2, 64, 128])
    @pytest.mark.parametrize("batch_size", [1, 64, 256])
    def test_token_embeddings_have_correct_shape_and_dtype(
        self, vocab_size: int, context_size: int, embedding_size: int, batch_size: int
    ) -> None:
        embedding = Embedding(vocab_size, context_size, embedding_size)

        token_ids = torch.randint(0, vocab_size, (batch_size, context_size))
        token_embeddings = embedding(token_ids)

        assert token_embeddings.shape == (batch_size, context_size, embedding_size)
        assert token_embeddings.dtype == torch.float32

    @pytest.mark.parametrize("vocab_size", [1, 32, 64])
    @pytest.mark.parametrize("context_size", [1])
    @pytest.mark.parametrize("embedding_size", [8, 16])
    def test_each_token_has_a_corresponding_embedding(
        self, vocab_size: int, context_size: int, embedding_size: int
    ) -> None:
        embedding = Embedding(vocab_size, context_size, embedding_size)
        token_ids = torch.arange(vocab_size).unsqueeze(1)

        token_embedding = embedding(token_ids)

        base_embeddings = token_embedding - sinusoidal_positional_encoding(
            context_size, embedding_size
        )
        assert not torch.any(torch.all(base_embeddings == 0, dim=2))

    def test_embedding_produces_correct_output(self):
        embedding = Embedding(vocab_size=4, context_size=2, embedding_size=4)

        embedding_weights = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5, 1.6],
            ],
            dtype=torch.float32,
        )
        # Replace the embedding weights with the known pre-defined weights.
        embedding._embedding.weight = nn.Parameter(embedding_weights)

        expected_output = torch.tensor(
            [
                [
                    [
                        [0.1000, 1.2000, 0.3000, 1.4000],
                        [1.3414, 1.1403, 0.7099, 1.7999],
                    ],
                    [
                        [0.9000, 2.0000, 1.1000, 2.2000],
                        [2.1414, 1.9403, 1.5099, 2.5999],
                    ],
                ]
            ],
            dtype=torch.float32,
        )
        actual_output = embedding(torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int))

        assert torch.allclose(expected_output, actual_output, atol=1e-4)

    @pytest.mark.parametrize("vocab_size", [16, 32])
    @pytest.mark.parametrize("embedding_size", [64, 128])
    @pytest.mark.parametrize("batch_size", [1, 8, 16])
    @pytest.mark.parametrize("num_lookups", [1, 2, 4])
    def test_embedding_is_same_across_multiple_lookups(
        self, vocab_size: int, embedding_size: int, batch_size: int, num_lookups: int
    ) -> None:
        context_size = 5
        embedding = Embedding(vocab_size, context_size, embedding_size)
        token_ids = torch.randint(0, vocab_size, (batch_size, context_size))
        initial_token_embedding = embedding(token_ids)
        for _ in range(num_lookups):
            token_embedding = embedding(token_ids)
            assert torch.allclose(initial_token_embedding, token_embedding)

    @pytest.mark.parametrize("vocab_size", [32])
    @pytest.mark.parametrize("embedding_size", [16])
    @pytest.mark.parametrize("token_id", [-1, 64])
    def test_error_is_raised_for_out_of_range_token_ids(
        self, vocab_size: int, embedding_size: int, token_id: int
    ) -> None:
        context_size = 3  # Add context_size parameter
        embedding = Embedding(vocab_size, context_size, embedding_size)
        token_ids = torch.full((1, context_size), token_id, dtype=torch.int)
        with pytest.raises(IndexError):
            embedding(token_ids)

    def test_padding_tokens_are_encoded_to_zeros(self):
        embedding = Embedding(
            vocab_size=8, context_size=4, embedding_size=8, padding_id=0
        )

        token_ids = torch.tensor([[1, 2, 0, 0]], dtype=torch.int)
        padding_mask = torch.tensor([[False, False, True, True]], dtype=torch.bool)

        token_embeddings = embedding(token_ids, padding_mask=padding_mask)

        assert torch.all(token_embeddings[0, -2] == 0)
        assert torch.all(token_embeddings[0, -1] == 0)


class TestPositionalEncoding:
    @pytest.mark.parametrize(
        ("context_size", "embedding_size"),
        [
            # Invalid types
            ("1", "1"),
            ("1", 1),
            (1, "1"),
            (1.0, 0.999),
            # Negative values
            (-1, -1),
            (-1, 1),
            (1, -1),
            # Zero values
            (0, 0),
            (0, 1),
            (1, 0),
            # None values
            (None, None),
            (None, 1),
            (1, None),
            # Wrong types/length
            ((1, 1), 1),
        ],
    )
    def test_value_error_is_raised_if_input_shape_is_invalid(
        self, context_size: int, embedding_size: int
    ) -> None:
        with pytest.raises(ValueError):
            sinusoidal_positional_encoding(context_size, embedding_size)

    @pytest.mark.parametrize("context_size", [1, 16, 32])
    @pytest.mark.parametrize("embedding_size", [2, 64, 128])
    def test_positional_encoding_has_correct_shape_and_dtype(
        self, context_size: int, embedding_size: int
    ) -> None:
        positional_encoding = sinusoidal_positional_encoding(
            context_size, embedding_size
        )
        assert positional_encoding.shape == (context_size, embedding_size)
        assert positional_encoding.dtype == torch.float32

    @pytest.mark.parametrize(
        ("context_size", "embedding_size", "expected_positional_encoding"),
        [
            (1, 2, torch.tensor([[0.0000, 1.0000]], dtype=torch.float32)),
            (
                2,
                2,
                torch.tensor([[0.0000, 1.0000], [0.8414, 0.5403]], dtype=torch.float32),
            ),
            (
                1,
                4,
                torch.tensor([[0.0000, 1.0000, 0.0000, 1.0000]], dtype=torch.float32),
            ),
            (
                2,
                4,
                torch.tensor(
                    [
                        [0.0000, 1.0000, 0.0000, 1.0000],
                        [0.8414, 0.5403, 0.0099, 0.9999],
                    ],
                    dtype=torch.float32,
                ),
            ),
        ],
    )
    def test_positional_encoding_matrix_has_correct_values(
        self,
        context_size: int,
        embedding_size: int,
        expected_positional_encoding: torch.Tensor,
    ) -> None:
        actual_positional_encoding = sinusoidal_positional_encoding(
            context_size, embedding_size
        )
        assert torch.allclose(
            actual_positional_encoding,
            expected_positional_encoding,
            atol=1e-4,
        )
