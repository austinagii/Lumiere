"""Data handling protocols and base classes.

This module provides the core abstractions for working with data in Lumière:
- Dataset: Protocol for accessing different data splits
- Pipeline: Protocol for processing data into batches
- Preprocessor: Base class for data preprocessing
"""

from collections.abc import Iterator
from typing import Any, Protocol, TypeVar


T = TypeVar("T")


class Dataset(Protocol):
    """Protocol defining the interface for dataset implementations.

    All dataset classes must implement the __getitem__ method to provide
    access to different data splits (e.g., train, validation, test).

    """

    def __getitem__(self, split_name: str) -> Iterator[T]:
        """Return an iterator over samples from the specified split.

        Args:
            split_name: Name of the split to access (e.g., "train", "valid", "test").

        Returns:
            An Iterator over samples from the specified split.

        Raises:
            KeyError: If the specified split is not present in the dataset.
        """
        ...


class Pipeline(Protocol):
    """Protocol defining the interface for pipeline implementations."""

    def batches(self, data): ...


class Preprocessor:
    """Base class for preprocessor implementations."""

    def __call__(self, *args, **kwargs) -> Any: ...
