import functools as ft

import pytest
import torch

from lumiere.data import DataLoader, Pipeline
from lumiere.tokenizers import SPECIAL_TOKENS, AsciiTokenizer
from lumiere.testing.datasets import FamousQuotesDataset, StringDataset


def assert_mask(tokens, mask, pad_id):
    # True means padding; False means token
    assert tokens.shape == mask.shape
    pad_locs = tokens == pad_id
    assert torch.equal(mask, pad_locs)


@pytest.fixture(scope="module")
def dataloader():
    quotes_dataset = FamousQuotesDataset(tone="glass", topics=["success"])
    return DataLoader.from_datasets([quotes_dataset])


@pytest.fixture
def tokenizer():
    return AsciiTokenizer()


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
    """Tests for the Pipeline class."""

    def test_produces_batches_of_data_from_datasources(self, pipeline_factory):
        num_batches = 3
        batch_size = 2
        context_size = 8
        sliding_window_size = 4

        pipeline = pipeline_factory(
            context_size=context_size,
            batch_size=batch_size,
            sliding_window_size=sliding_window_size,
        )

        expected_token_batches = torch.tensor(
            [
                [
                    [66, 101, 32, 121, 111, 117, 114, 115],  # "Be yours"
                    [111, 117, 114, 115, 101, 108, 102, 59],  # "ourself;"
                ],
                [
                    [101, 108, 102, 59, 32, 101, 118, 101],  # "elf; eve"
                    [32, 101, 118, 101, 114, 121, 111, 110],  # " everyon"
                ],
                [
                    [114, 121, 111, 110, 101, 32, 101, 108],  # "ryone el"
                    [101, 32, 101, 108, 115, 101, 32, 105],  # "e else i"
                ],
            ],
            dtype=torch.long,
        )
        expected_mask_batches = torch.zeros(
            (num_batches, batch_size, context_size), dtype=torch.bool
        )

        batches = pipeline.batches(num_batches=num_batches)
        token_batches, mask_batches = [
            torch.stack(b) for b in zip(*batches, strict=False)
        ]

        assert expected_token_batches.shape == (num_batches, batch_size, context_size)
        assert expected_token_batches.dtype == torch.long
        assert torch.equal(token_batches, expected_token_batches)

        assert expected_mask_batches.shape == (num_batches, batch_size, context_size)
        assert expected_mask_batches.dtype == torch.bool
        assert torch.equal(mask_batches, expected_mask_batches)

    def test_pads_contexts_where_samples_have_insufficient_tokens(
        self, pipeline_factory
    ):
        num_batches = 3
        batch_size = 2
        context_size = 8
        tokenizer = AsciiTokenizer()
        datasets = [StringDataset({"train": ["A short sample", "And another"]})]
        dataloader = DataLoader.from_datasets(datasets)

        pipeline = Pipeline(
            dataloader=dataloader,
            split="train",
            tokenizer=tokenizer,
            pad_id=SPECIAL_TOKENS["padding"].id,
            batch_size=batch_size,
            context_size=context_size,
            sliding_window_size=0,
        )

        batches = pipeline.batches(num_batches=num_batches)
        token_batches, mask_batches = [
            torch.stack(b) for b in zip(*batches, strict=False)
        ]

        pad_id = SPECIAL_TOKENS["padding"].id
        expected_token_batches = torch.tensor(
            [
                [
                    [65, 32, 115, 104, 111, 114, 116, 32],
                    [115, 97, 109, 112, 108, 101, pad_id, pad_id],
                ],
                [
                    [65, 110, 100, 32, 97, 110, 111, 116],
                    [104, 101, 114, pad_id, pad_id, pad_id, pad_id, pad_id],
                ],
            ],
            dtype=torch.long,
        )

        expected_mask_batches = torch.tensor(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1],
                ],
            ],
            dtype=torch.bool,
        )

        assert torch.equal(token_batches, expected_token_batches)
        assert torch.equal(mask_batches, expected_mask_batches)

    def test_produces_partial_batch_if_insufficient_tokens_in_samples(self):
        dataloader = DataLoader.from_datasets(
            [StringDataset({"train": ["A short sample"]})]
        )
        pipeline = Pipeline(
            dataloader=dataloader,
            split="train",
            tokenizer=AsciiTokenizer(),
            batch_size=8,
            context_size=6,
            pad_id=SPECIAL_TOKENS["padding"].id,
            sliding_window_size=0,
        )

        batches = pipeline.batches()

        expected_shape = (3, 6)
        assert all(
            tokens.shape == expected_shape and mask.shape == expected_shape
            for tokens, mask in batches
        )
