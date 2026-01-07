import functools as ft

import pytest
import torch

from lumiere.data import DataLoader, Pipeline
from lumiere.data.tokenizer import SPECIAL_TOKENS, Tokenizer
from lumiere.utils.testing.datasets import LoremIpsumDataset


def assert_mask(tokens, mask, pad_id):
    # True means padding; False means token
    assert tokens.shape == mask.shape
    pad_locs = tokens == pad_id
    assert torch.equal(mask, pad_locs)


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer({"A": [1, 2, 3, 4], "B": [9, 9]})


@pytest.fixture(scope="module")
def dataloader():
    return DataLoader.from_datasets([LoremIpsumDataset(source="ipsala", count=10)])


@pytest.fixture
def tokenizer(dataloader):
    tokenizer = Tokenizer()
    tokenizer.train(dataloader["train"])
    return tokenizer


@pytest.fixture
def pipeline_factory(dataloader, tokenizer):
    return ft.partial(
        Pipeline,
        split="train",
        dataloader=dataloader,
        tokenizer=tokenizer,
        pad_id=SPECIAL_TOKENS["padding"].id,
    )


class TestPipeline:
    def test_pipeline_can_be_initialized_with_datasets(self, pipeline_factory):
        pipeline = pipeline_factory(
            context_size=8,
            batch_size=2,
            sliding_window_size=4,
        )

        batches = pipeline.batches(num_batches=3)
        for batch, batch_mask in batches:
            assert batch.shape == (2, 8)
            assert batch_mask.shape == (2, 8)


class TestToTrainingBatches:
    def test_basic_no_overlap_single_seq_exact_fit(self):
        corpus = ["A"]
        tok = FakeTokenizer({"A": [1, 2, 3, 4]})

        actual_output = collect_batches(
            corpus=corpus,
            tokenizer=tok,
            context_size=2,
            batch_size=2,
            pad_id=0,
            sliding_window_size=0,
        )

        assert len(actual_output) == 1
        tokens, padding_mask = actual_output[0]
        # Expect: two rows, each of length 2, no padding
        assert tokens.shape == (2, 2)
        assert isinstance(tokens, torch.Tensor)
        assert isinstance(padding_mask, torch.Tensor)
        assert padding_mask.dtype == torch.bool
        # Order preserved, chunked by context_size
        assert torch.equal(tokens[0], torch.tensor([1, 2]))
        assert torch.equal(tokens[1], torch.tensor([3, 4]))
        assert torch.equal(
            padding_mask, torch.zeros_like(padding_mask, dtype=torch.bool)
        )

    def test_partial_last_row_no_overlap(self):
        corpus = ["A"]
        tok = FakeTokenizer({"A": [1, 2, 3]})

        actual_output = collect_batches(
            corpus=corpus,
            tokenizer=tok,
            context_size=2,
            batch_size=4,
            pad_id=0,
            sliding_window_size=0,
        )

        # contexts: [1,2], [3,PAD] -> only 2 rows; trimmed partial batch
        assert len(actual_output) == 1
        tokens, padding_mask = actual_output[0]
        assert tokens.shape == (2, 2)  # trimmed
        assert torch.equal(tokens[0], torch.tensor([1, 2]))
        assert torch.equal(tokens[1], torch.tensor([3, 0]))
        assert_mask(tokens, padding_mask, 0)

    def test_overlap_applied_only_when_previous_context_full(self):
        # seq tokens of length 5 with seq_len=4, overlap=2:
        # Row0: [1,2,3,4] (full)
        # Row1 starts with last 2 of Row0 => [3,4, 5, PAD]
        corpus = ["A"]
        tok = FakeTokenizer({"A": [1, 2, 3, 4, 5]})

        actual_output = collect_batches(
            corpus=corpus,
            tokenizer=tok,
            context_size=4,
            batch_size=4,
            pad_id=0,
            sliding_window_size=2,
        )

        assert len(actual_output) == 1
        tokens, padding_mask = actual_output[0]
        assert tokens.shape == (2, 4)
        assert torch.equal(tokens[0], torch.tensor([1, 2, 3, 4]))
        assert torch.equal(
            tokens[1], torch.tensor([3, 4, 5, 0])
        )  # overlap from full previous row
        assert_mask(tokens, padding_mask, 0)

    def test_overlap_not_applied_if_previous_row_padded(self):
        # seq_len=4, overlap=2, tokens len=3:
        # Row0: [1,2,3,PAD] (NOT full)
        # Row1 should NOT start with [2,3]; it should start fresh (since prev row had
        # padding).
        corpus = ["A"]
        tok = FakeTokenizer({"A": [1, 2, 3]})

        actual_output = collect_batches(
            corpus=corpus,
            tokenizer=tok,
            context_size=4,
            batch_size=4,
            pad_id=0,
            sliding_window_size=2,
        )

        # Only one row should be produced (no reason to create Row1)
        assert len(actual_output) == 1
        tokens, padding_mask = actual_output[0]
        assert tokens.shape == (1, 4)
        assert torch.equal(tokens[0], torch.tensor([1, 2, 3, 0]))
        assert_mask(tokens, padding_mask, 0)

    def test_no_cross_sequence_overlap(self):
        # Two sequences; overlap must not carry across boundary.
        # First seq fills a full row; second should NOT inherit its tail.
        corpus = ["A", "B"]
        tok = FakeTokenizer(
            {
                "A": [1, 2, 3, 4],  # full row
                "B": [9, 9],  # starts clean
            }
        )

        actual_output = collect_batches(
            corpus=corpus,
            tokenizer=tok,
            context_size=4,
            batch_size=4,
            pad_id=0,
            sliding_window_size=2,
        )

        assert len(actual_output) == 1
        tokens, padding_mask = actual_output[0]
        assert tokens.shape == (2, 4)
        assert torch.equal(tokens[0], torch.tensor([1, 2, 3, 4]))
        # Row1 must NOT begin with [3,4]; it should be just the B tokens then padding.
        assert torch.equal(tokens[1], torch.tensor([9, 9, 0, 0]))
        assert_mask(tokens, padding_mask, 0)

    def test_multiple_full_batches_and_cap(self):
        corpus = ["A"]
        tok = FakeTokenizer({"A": list(range(1, 1 + 12))})  # 12 tokens

        # seq_len=3 -> 4 rows; batch_size=2 -> 2 full batches
        actual_output = collect_batches(
            corpus=corpus,
            tokenizer=tok,
            context_size=3,
            batch_size=2,
            pad_id=0,
            sliding_window_size=0,
        )

        assert len(actual_output) == 2

        # Cap to 1 full batch
        capped_output = collect_batches(
            corpus=corpus,
            tokenizer=tok,
            context_size=3,
            batch_size=2,
            pad_id=0,
            sliding_window_size=0,
            num_batches=1,
        )

        assert len(capped_output) == 1
        tokens, padding_mask = capped_output[0]
        assert tokens.shape == (2, 3)
        assert_mask(tokens, padding_mask, 0)

    @pytest.mark.parametrize(
        "context_size, batch_size, overlap",
        [
            (0, 2, 0),  # invalid context_size
            (2, 0, 0),  # invalid batch_size
            (4, 2, 4),  # overlap == context_size
            (4, 2, 5),  # overlap > context_size
            (4, 2, -1),  # negative overlap
        ],
    )
    def test_validation_errors(self, context_size, batch_size, overlap):
        corpus = ["A"]
        tok = FakeTokenizer({"A": [1, 2, 3, 4]})

        with pytest.raises(Exception):
            collect_batches(
                corpus=corpus,
                tokenizer=tok,
                context_size=context_size,
                batch_size=batch_size,
                pad_id=0,
                sliding_window_size=overlap,
            )
