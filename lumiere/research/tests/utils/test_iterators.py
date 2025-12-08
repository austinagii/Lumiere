import itertools
import random

import pytest

from lumiere.research.src.utils.iterators import merge_iterators


class TestMergeMode:
    """A suite of tests for the `MergeMode` class."""

    @pytest.mark.parametrize(
        "inputs,output",
        [
            # All empty.
            ([[], [], []], []),
            # Multiple empty.
            ([[1], [], []], [1]),
            ([[], [4], []], [4]),
            ([[], [], [7]], [7]),
            # Same length.
            ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            # Varying length.
            ([[1], [4, 5], [7, 8, 9]], [1, 4, 5, 7, 8, 9]),
            ([[1, 2], [4], [7, 8, 9]], [1, 2, 4, 7, 8, 9]),
            ([[1, 2], [4, 5, 6], [7]], [1, 2, 4, 5, 6, 7]),
        ],
    )
    def test_iteritem_uses_greedy_strategy_if_specified(self, inputs, output):
        iterators = [iter(x) for x in inputs]
        merged_iterator = merge_iterators(iterators, mode="greedy")
        assert list(merged_iterator) == output

    @pytest.mark.parametrize(
        "inputs,output",
        [
            # All empty.
            ([[], [], []], []),
            # Multiple empty.
            ([[1], [], []], [1]),
            ([[], [4], []], [4]),
            ([[], [], [7]], [7]),
            # Same length.
            ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 4, 7, 2, 5, 8, 3, 6, 9]),
            # Rotating exhausted iterators.
            ([[], [4, 5, 6], [7, 8, 9]], [4, 7, 5, 8, 6, 9]),
            ([[1, 2, 3], [], [7, 8, 9]], [1, 7, 2, 8, 3, 9]),
            ([[1, 2, 3], [4, 5, 6], []], [1, 4, 2, 5, 3, 6]),
            # Varying length.
            ([[1], [4, 5], [7, 8, 9]], [1, 4, 7, 5, 8, 9]),
            ([[1, 2], [4], [7, 8, 9]], [1, 4, 7, 2, 8, 9]),
            ([[1, 2, 3], [4, 5, 6], [7]], [1, 4, 7, 2, 5, 3, 6]),
        ],
    )
    def test_iteritem_uses_circular_strategy_if_specified(self, inputs, output):
        iterators = [iter(x) for x in inputs]
        merged_iterator = merge_iterators(iterators, mode="circular")
        assert list(merged_iterator) == output

    @pytest.mark.parametrize(
        "inputs,output",
        [
            # All empty.
            ([[], [], []], []),
            # Multiple empty.
            ([[1, 2, 3], [], []], [1, 2, 3]),
            ([[], [4, 5, 6], []], [4, 5, 6]),
            ([[], [], [7, 8, 9]], [7, 8, 9]),
        ],
    )
    def test_random_covers_edge_cases(self, inputs, output):
        iterators = [iter(x) for x in inputs]
        merged_iterator = merge_iterators(iterators, mode="random")
        assert list(merged_iterator) == output

    def test_random_handles_same_and_variable_length_inputs(self):
        data = [list(range(random.randint(0, 10))) for _ in range(20)]
        all_elements = sorted(itertools.chain.from_iterable(data))

        outputs = [
            merge_iterators([iter(x) for x in data], mode="random") for _ in range(10)
        ]

        assert all(sorted(output) == all_elements for output in outputs)

        # If an element is taken from a randomly chosen iterator at each step, then the
        # chance of getting a pair of identifcal sequences should be very small.
        all(a == b for a, b in itertools.combinations(outputs, 2))
