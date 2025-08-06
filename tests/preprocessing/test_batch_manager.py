import pytest

from lumiere.preprocessing.batch_manager import BatchManager


@pytest.fixture
def batch_manager():
    return BatchManager(
        context_size=4,
        batch_size=2,
        sliding_window_size=0,
        padding_token="<|pad|>",
    )


class TestContextBatchManager:
    @pytest.mark.parametrize("input", ["abc", 123, None, [1, 2, 3]])
    def test_raises_error_if_input_is_not_iterable_of_sequences(
        self, batch_manager, input
    ):
        with pytest.raises(TypeError):
            list(batch_manager.to_batches(input))

    def test_correctly_splits_tokens_into_batches(self, batch_manager):
        text = [
            ["a", "b", "c", "d", "e", "f", "g"],  # 7 tokens
            ["h", "i", "j", "k", "l", "m", "n", "o", "p"],  # 9 tokens
            ["q", "r", "s", "t"],  # 4 tokens
        ]

        actual_batches, actual_padding_masks = list(
            zip(*batch_manager.to_batches(text))
        )

        expected_batches = [
            [["a", "b", "c", "d"], ["e", "f", "g", "<|pad|>"]],
            [["h", "i", "j", "k"], ["l", "m", "n", "o"]],
            [["p", "<|pad|>", "<|pad|>", "<|pad|>"], ["q", "r", "s", "t"]],
        ]

        expected_padding_masks = [
            [[False, False, False, False], [False, False, False, True]],
            [[False, False, False, False], [False, False, False, False]],
            [[False, True, True, True], [False, False, False, False]],
        ]

        assert all(
            [
                actual == expected
                for actual, expected in zip(actual_batches, expected_batches)
            ]
        )
        assert all(
            [
                actual == expected
                for actual, expected in zip(
                    actual_padding_masks, expected_padding_masks
                )
            ]
        )

    def test_applies_sliding_window_correctly(self):
        input = [
            ["I", "am", "a", "fresh", "man", "knowing", "the", "key", "to", "life"],
        ]

        batch_manager = BatchManager(
            context_size=5,
            batch_size=2,
            sliding_window_size=2,
            padding_token="<|pad|>",
        )

        actual_batches, actual_batch_masks = zip(*batch_manager.to_batches(input))

        expected_batches = [
            [
                ["I", "am", "a", "fresh", "man"],
                ["fresh", "man", "knowing", "the", "key"],
            ],
            [
                ["the", "key", "to", "life", "<|pad|>"],
            ],
        ]

        # fmt: off
        expected_batch_masks = [
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
            [
                [False, False, False, False, True],
            ],
        ]
        # fmt: on

        assert all(
            [
                actual == expected
                for actual, expected in zip(actual_batches, expected_batches)
            ]
        )
        assert all(
            [
                actual == expected
                for actual, expected in zip(actual_batch_masks, expected_batch_masks)
            ]
        )

    def test_returns_partial_batch_if_sequence_does_not_provide_enough_contexts(self):
        input = [
            ["I", "am", "a", "fresh", "man", "knowing", "the", "key", "to", "life"]
        ]

        batch_manager = BatchManager(context_size=5, batch_size=5)

        actual_batches, actual_batch_masks = zip(*batch_manager.to_batches(input))

        expected_batches = [
            [
                ["I", "am", "a", "fresh", "man"],
                ["knowing", "the", "key", "to", "life"],
            ],
        ]

        # fmt: off
        expected_batch_masks = [
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
        ]
        # fmt: on

        assert all(
            [
                actual == expected
                for actual, expected in zip(actual_batches, expected_batches)
            ]
        )
        assert all(
            [
                actual == expected
                for actual, expected in zip(actual_batch_masks, expected_batch_masks)
            ]
        )
