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

import contextlib
import importlib
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol, TypeVar

from lumiere.utils.iterators import MergeMode, merge_iterators


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


# A registry of datasets indexed by custom names.
_dataset_registry: dict[str, type[Dataset]] = {}


def dataset(dataset_name: str):
    """Decorator to register a dataset class in the global registry.

    Registered datasets can be retrieved by name using get_dataset().

    Args:
        dataset_name: Unique identifier for the dataset in the registry.

    """

    def decorator(cls):
        register_dataset(dataset_name, cls)
        return cls

    return decorator


def register_dataset(name: str, cls: type[Dataset]) -> None:
    _dataset_registry[name] = cls


def get_dataset(dataset_name: str) -> type[Dataset] | None:
    """Retrieve a dataset class from the registry by name.

    Args:
        dataset_name: Registered identifier of the dataset to retrieve.

    Returns:
        Dataset class if found in the registry, None otherwise.
    """
    if not _dataset_registry:  # Refresh the imports.
        datasets_dir = Path(__file__).parent / "datasets"
        module_files = datasets_dir.glob("*.py")
        module_files = [f for f in module_files if not f.stem.startswith("_")]

        # Import each module to trigger @dataset decorator registration
        for module_file in module_files:
            module_name = f"lumiere.data.datasets.{module_file.stem}"
            with contextlib.suppress(ImportError):
                importlib.import_module(module_name)

    return _dataset_registry.get(dataset_name)


class DataLoader:
    """Load and merge samples from a given split across multiple registered datasets.

    Each source dataset can be specified during initialization by passing a mapping
    containing the registered name of the dataset along with any arguments required for
    it's construction. Once initialized, these datasets can be accessed using the
    'datasets' property with elements from a given split across all source datasets
    being accessed using key-based notation with the name of the desired split.
    (e.g. dataloader["train"]).

    If there are multiple source datasets used by this dataloader, then a merge mode can
    be provided to specify how samples from the specified split across multiple datasets
    can be merged into one stream of outputs.

    If the specified split could not be found in one or more source datasets, then those
    datasets will be excluded from the resulting iterators. If no datasets contain the
    specified split then an error will be raised.

    Note that if only one dataset is specified, then the merge mode will devolve
    to greedy.

    Attributes:
        merge_mode: The method to be used when combining iterators over samples from
            multiple source datasets into a single iterator.
    """

    def __init__(
        self,
        datasets: Iterable[Mapping[str, Any]] | None = None,
        merge_mode: MergeMode | str = "greedy",
    ) -> None:
        """Initialize a DataLoader with multiple datasets.

        Each dataset configuration must contain a 'name' field with the registered
        dataset identifier, plus any additional keyword arguments required for
        that dataset's initialization.

        Args:
            datasets: Iterable of dataset configurations. Each config must contain:
                - 'name': Registered identifier of the dataset.
                - Additional key-value pairs for dataset-specific parameters.
            merge_mode: Strategy for merging iterators from multiple datasets.
                Options: "greedy" (sequential), "round_robin" (alternating),
                "sampling" (random). Defaults to "greedy".

        Raises:
            ValueError: If a dataset config is missing 'name', if the specified
                dataset is not registered, or if merge_mode is invalid.
            RuntimeError: If an error occurs during dataset initialization.

        """
        self.merge_mode = (
            MergeMode(merge_mode) if isinstance(merge_mode, str) else merge_mode
        )
        self._datasets: list[Dataset] = []
        if datasets is not None:
            self._datasets.extend([self._init_dataset(dataset) for dataset in datasets])

    # TODO: Consider making initialization with dataset instances the default.
    @classmethod
    def from_datasets(
        cls, datasets: list[Dataset], merge_mode: MergeMode | str = "greedy"
    ) -> "DataLoader":
        # TODO: Validate argument types.
        instance = cls()
        instance._datasets = datasets
        return instance

    @staticmethod
    def _init_dataset(dataset_config: Mapping[str, Any]) -> Dataset:
        """Initialize a single dataset from its configuration.

        Retrieves the dataset class from the registry and instantiates it
        with the provided parameters.

        Args:
            dataset_config: Configuration dictionary containing 'name' and
                dataset-specific initialization parameters.

        Returns:
            Initialized Dataset instance.

        Raises:
            ValueError: If 'name' is missing or dataset is not registered.
            RuntimeError: If dataset initialization fails.
        """
        if (dataset_name := dataset_config.get("name")) is None:
            raise ValueError("A dataset config must contain a dataset name.")

        dataset_cls = get_dataset(dataset_name)
        if dataset_cls is None:
            raise ValueError(
                f"The specified dataset '{dataset_name}' could not be found in the registry."  # noqa: E501
            )

        try:
            return dataset_cls(
                **{arg: argv for arg, argv in dataset_config.items() if arg != "name"}
            )
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while initializing dataset '{dataset_name}'"
            ) from e

    def __getitem__(self, split_name: str) -> Iterator[str]:
        """Return an iterator over the specified split across all datasets.

        Samples are yielded according to the configured merge_mode. Datasets
        that don't contain the requested split are automatically excluded.
        If only one dataset is available, samples are taken greedily regardless
        of merge_mode.

        Args:
            split_name: Name of the split to iterate (e.g., "train", "valid", "test").

        Returns:
            Iterator yielding samples from the specified split across all
            available datasets, merged according to merge_mode.

        Raises:
            KeyError: If no dataset contains the specified split.

        """
        split_iterators = []

        for dataset in self._datasets:
            try:
                split_iterators.append(dataset[split_name])
            except KeyError:
                continue

        return merge_iterators(split_iterators, mode=self.merge_mode)

    @property
    def datasets(self) -> list[Dataset]:
        """List of initialized Dataset instances managed by this dataloader.

        Returns:
            List of Dataset objects in the order they were configured.
        """
        return self._datasets
