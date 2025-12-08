"""Utilities for manipulating iterators."""

import random
from collections.abc import Iterator, Sequence
from enum import StrEnum, auto
from itertools import chain
from typing import TypeVar


T = TypeVar("T")


class MergeMode(StrEnum):
    """Strategies for merging multiple iterators into a single iterator.

    Each mode represents a unique way of combining elements from multiple iterators into
    a single stream of elements. Each mode handles empty / exhausted and varing length
    iterators gracefully and continues until elements from all iterators have been
    consumed.

    Attributes:
        GREEDY: Exhausts each iterator completely before advancing to the next.
            Example: A1, A2, B1, B2, C1, C2

        CIRCULAR: Takes one element from each iterator in round-robin fashion, skipping
            exhausted iterators in subsequent rounds.
            Example: A1, B1, C1, A2, B2, C2

        RANDOM: Selects elements from a randomly chosen non-exhausted iterator, skipping
            exhausted iterators.
            Example: C1, B1, B2, A1, C2, A2

    """

    GREEDY = auto()
    CIRCULAR = auto()
    RANDOM = auto()


def _merge_greedy(iterators: Sequence[Iterator[T]]) -> Iterator[T]:
    yield from chain.from_iterable(iterators)


def _merge_circular(iterators: Sequence[Iterator[T]]) -> Iterator[T]:
    iterators = list(iterators)
    i = 0

    while iterators:
        try:
            yield next(iterators[i])
            i += 1
        except StopIteration:
            iterators.pop(i)

        if iterators:
            i = min(i, len(iterators)) % len(iterators)


def _merge_random(iterators: Sequence[Iterator[T]]) -> Iterator[T]:
    iterators = list(iterators)

    while iterators:
        i = random.randint(0, len(iterators) - 1) if len(iterators) > 1 else 0

        try:
            yield next(iterators[i])
        except StopIteration:
            iterators.pop(i)


merge_methods = {
    MergeMode.GREEDY: _merge_greedy,
    MergeMode.CIRCULAR: _merge_circular,
    MergeMode.RANDOM: _merge_random,
}


def merge_iterators(iterators: Sequence[Iterator[T]], mode="greedy") -> Iterator[T]:
    """Merge the specified iterators into a single iterator.

    Creates an iterator over a sequence of elements consumed from the provided
    iterators according to the mode specified.
    """
    if not iterators:
        return iter([])

    merge = merge_methods.get(mode)
    if merge is None:
        raise ValueError(f"'{mode}' is not a valid merge mode.")

    return merge(iterators)
