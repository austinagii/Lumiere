"""Dataset loading utilities for initializing datasets from configuration."""

from collections.abc import Mapping
from typing import Any

from lumiere.data.dataloader import DataLoader
from lumiere.data.dataset import Dataset, get_dataset
from lumiere.di import DependencyContainer


def load(
    config: Mapping[str, Any], container: DependencyContainer | None = None
) -> DataLoader:
    """Load and return a DataLoader instance from a configuration dictionary.

    The configuration must contain a 'datasets' field with a list of dataset
    configurations. Each dataset configuration must contain a 'name' field with
    the registered dataset identifier, plus any additional keyword arguments
    required for that dataset's initialization. Additional fields in the config
    (e.g., 'merge_mode') are passed to the DataLoader constructor.

    Dependencies can be injected via a DependencyContainer. Values in the config
    that start with "@" (e.g., "@preprocessor") will be resolved from the container.

    Args:
        config: Configuration dictionary containing:
            - 'datasets': List of dataset configurations. Each must contain:
                - 'name': Registered identifier of the dataset.
                - Additional key-value pairs for dataset-specific parameters.
                - Values starting with "@" will be resolved from the container.
            - Other DataLoader parameters (e.g., 'merge_mode').
        container: Optional DependencyContainer for resolving dependencies.

    Returns:
        Initialized DataLoader instance.

    Raises:
        ValueError: If config is missing 'datasets', if a dataset config is
            missing 'name', or if a specified dataset is not registered,
            or if a dependency cannot be resolved.
        RuntimeError: If an error occurs during dataset initialization.

    Example:
        >>> config = {
        ...     "datasets": [
        ...         {"name": "wikitext", "split": "50:50:50"},
        ...         {"name": "another_dataset", "param": "value"}
        ...     ],
        ...     "merge_mode": "round_robin"
        ... }
        >>> dataloader = load(config)
        >>>
        >>> # With dependency injection
        >>> container = DependencyContainer()
        >>> container.register("preprocessor", preprocessor_instance)
        >>> config = {
        ...     "datasets": [
        ...         {"name": "wikitext", "preprocessor": "@preprocessor"}
        ...     ],
        ...     "merge_mode": "greedy"
        ... }
        >>> dataloader = load(config, container)
    """
    if (dataset_configs := config.get("datasets")) is None:
        raise ValueError("Configuration must contain a 'datasets' field.")

    # Initialize datasets from their configurations
    datasets = [
        _init_dataset(dataset_config, container) for dataset_config in dataset_configs
    ]

    # Extract DataLoader parameters (everything except 'datasets')
    dataloader_params = {
        key: _resolve_value(value, container, "DataLoader", key)
        for key, value in config.items()
        if key != "datasets"
    }

    return DataLoader(datasets, **dataloader_params)


def _init_dataset(
    dataset_config: Mapping[str, Any], container: DependencyContainer | None
) -> Dataset:
    """Initialize a single dataset from its configuration.

    Retrieves the dataset class from the registry and instantiates it
    with the provided parameters. Supports dependency injection for config values.

    Args:
        dataset_config: Configuration dictionary containing 'name' and
            dataset-specific initialization parameters.
        container: Optional DependencyContainer for resolving dependencies.

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
        # Resolve dependencies in dataset config
        init_params = {
            arg: _resolve_value(argv, container, dataset_name, arg)
            for arg, argv in dataset_config.items()
            if arg != "name"
        }
        return dataset_cls(**init_params)
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while initializing dataset '{dataset_name}'"
        ) from e


def _resolve_value(
    value: Any, container: DependencyContainer | None, context: str, key: str
) -> Any:
    """Resolve a config value, handling dependency injection references.

    Args:
        value: The value to resolve. If it's a string starting with "@",
            it will be resolved from the container.
        container: Optional DependencyContainer for resolving dependencies.
        context: Context string for error messages (e.g., dataset name).
        key: The config key being resolved.

    Returns:
        The resolved value.

    Raises:
        ValueError: If a dependency reference cannot be resolved.
    """
    if isinstance(value, str) and value.startswith("@"):
        if container is None:
            raise ValueError(
                f"Dependency reference '{value}' found for '{key}' in '{context}', "
                f"but no DependencyContainer was provided."
            )

        dep_name = value[1:]  # Remove the "@" prefix
        resolved = container.get(dep_name)

        if resolved is None:
            raise ValueError(
                f"Dependency '{dep_name}' for '{key}' in '{context}' "
                f"could not be found in the container."
            )

        return resolved

    return value
