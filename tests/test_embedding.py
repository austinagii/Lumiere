import torch
import pytest

from prism.embedding import Embedding


class TestEmbedding:
    @pytest.mark.parametrize("vocab_size", [1, 16, 32])
    @pytest.mark.parametrize("embedding_size", [1, 64, 128])
    @pytest.mark.parametrize("batch_size", [1, 64, 256])
    def test_embedding_has_correct_shape_and_dtype(
        self,
        vocab_size: int,
        embedding_size: int,
        batch_size: int
    ) -> None:
        embedding = Embedding(vocab_size, embedding_size)
        token_ids = torch.randint(
            0, vocab_size, (batch_size,), dtype=torch.int)
        token_embedding = embedding(token_ids)
        assert token_embedding.shape == (batch_size, embedding_size)
        assert token_embedding.dtype == torch.float32

    @pytest.mark.parametrize("vocab_size", [1, 32, 64])
    @pytest.mark.parametrize("embedding_size", [8, 16])
    def test_each_token_id_has_an_embedding(
        self,
        vocab_size: int,
        embedding_size: int
    ) -> None:
        embedding = Embedding(vocab_size, embedding_size)
        token_ids = torch.arange(vocab_size)
        token_embedding = embedding(token_ids)
        assert not torch.any(torch.all(token_embedding == 0, dim=1))

    @pytest.mark.parametrize("vocab_size", [16, 32])
    @pytest.mark.parametrize("embedding_size", [64, 128])
    @pytest.mark.parametrize("batch_size", [1, 8, 16])
    @pytest.mark.parametrize("num_lookups", [1, 2, 4])
    def test_embedding_is_same_across_multiple_lookups(
        self,
        vocab_size: int,
        embedding_size: int,
        batch_size: int,
        num_lookups: int
    ) -> None:
        embedding = Embedding(vocab_size, embedding_size)
        token_ids = torch.arange(batch_size)
        initial_token_embedding = embedding(token_ids)
        for _ in range(num_lookups):
            token_embedding = embedding(token_ids)
            assert torch.allclose(initial_token_embedding, token_embedding)

    @pytest.mark.parametrize("vocab_size", [32])
    @pytest.mark.parametrize("embedding_size", [16])
    @pytest.mark.parametrize("token_id", [-1, 64])
    def test_error_is_raised_for_out_of_range_token_ids(
        self,
        vocab_size: int,
        embedding_size: int,
        token_id: int
    ) -> None:
        embedding = Embedding(vocab_size, embedding_size)
        token_id = torch.tensor([token_id], dtype=torch.int)
        with pytest.raises(IndexError):
            embedding(token_id) 