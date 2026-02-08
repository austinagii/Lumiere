from collections.abc import Iterator
from typing import Protocol, TypeVar

from lumiere.loading import Registry

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
_registry = Registry[type[Dataset]](
    name="dataset",
    base_module="lumiere.data.datasets",
    discovery_paths=["."],
)

# Expose existing API for backward compatibility
dataset = _registry.decorator
register_dataset = _registry.register
get_dataset = _registry.get

# Create loader (imported here to avoid circular imports)
def load(config: dict, container: Any = None):
    """Load and return a DataLoader instance from a configuration dictionary.

    The configuration must contain a 'datasets' field with a list of dataset
    configurations. Each dataset configuration must contain a 'name' field with
    the registered dataset identifier, plus any additional keyword arguments
    required for that dataset's initialization.

    Dependencies can be injected via a DependencyContainer. Values in the config
    that start with "@" (e.g., "@preprocessor") will be resolved from the container.

    Args:
        config: Configuration dictionary containing:
            - 'datasets': List of dataset configurations.
            - Other DataLoader parameters (e.g., 'merge_mode').
        container: Optional DependencyContainer for resolving dependencies.

    Returns:
        Initialized DataLoader instance.

    Example:
        >>> config = {
        ...     "datasets": [
        ...         {"name": "wikitext", "split": "50:50:50"}
        ...     ],
        ...     "merge_mode": "round_robin"
        ... }
        >>> dataloader = load(config)
    """
    from lumiere.data.dataloader import DataLoader
    from lumiere.loading import ConfigLoader, resolve_value

    # Create loader for individual datasets
    _dataset_loader = ConfigLoader[Dataset](_registry)

    if (dataset_configs := config.get("datasets")) is None:
        raise ValueError("Configuration must contain a 'datasets' field.")

    # Initialize datasets from their configurations
    datasets = [_dataset_loader.load(dict(dc), container=container) for dc in dataset_configs]

    # Extract DataLoader parameters (everything except 'datasets')
    dataloader_params = {
        key: resolve_value(value, container)
        for key, value in config.items()
        if key != "datasets"
    }

    return DataLoader(datasets, **dataloader_params)
