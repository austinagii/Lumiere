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

from .dataloader import DataLoader
from .dataset import dataset, get_dataset, register_dataset
from .pipeline import get_pipeline, pipeline, register_pipeline
from .preprocessor import get_preprocessor, preprocessor, register_preprocessor


__all__ = [
    "dataset",
    "register_dataset",
    "get_dataset",
    "pipeline",
    "register_pipeline",
    "get_pipeline",
    "preprocessor",
    "register_preprocessor",
    "get_preprocessor",
    "DataLoader",
]
