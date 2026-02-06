from collections.abc import Iterator, Mapping
from typing import Any

from lumiere.utils.iterators import MergeMode, merge_iterators

from .dataset import Dataset, get_dataset


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
        datasets: list[Dataset],
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

        self.datasets = datasets

    @classmethod
    def from_config(
        cls, dataset_configs: list[Mapping[str, Any]], merge_mode
    ) -> "DataLoader":
        datasets = [cls._init_dataset(config) for config in dataset_configs]

        return cls(datasets, merge_mode=merge_mode)

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

        for dataset in self.datasets:
            try:
                split_iterators.append(dataset[split_name])
            except KeyError:
                continue

        return merge_iterators(split_iterators, mode=self.merge_mode)
