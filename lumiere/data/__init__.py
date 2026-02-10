"""Classes and utilities for creating and accessing datasets.

Custom datasets can be created and (through the registration system) dynamically
discovered and loaded by name, enabling flexible data pipeline configurations.

Example:
    >>> # Define and register a custom dataset
    >>> @dataset("my-dataset")
    >>> class MyDataset:
    ...     def __init__(self, format: str):
    ...         self.format = format
    ...
    ...     def __getitem__(self, split_name: str) -> Iterator[str]:
    ...         # Yield samples for the requested split
    ...         yield from ["sample1", "sample2"]
    >>>
    >>> # Use the registered dataset via DataLoader
    >>> data = DataLoader(
    ...     datasets=[{"name": "my-dataset", "format": "text"}],
    ...     merge_mode="greedy"
    ... )
    >>> for sample in data["train"]:
    ...     print(sample)
"""

from .base import DataLoader, Dataset, Pipeline, Preprocessor

__all__ = [
    "DataLoader",
    "Dataset",
    "Pipeline",
    "Preprocessor",
]
