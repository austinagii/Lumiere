"""Dataset loading functionality."""

from typing import Any

from lumiere.data.dataloader import DataLoader
from lumiere.data.dataset import Dataset
from lumiere.discover import get_registry
from lumiere.loading.loader import ConfigLoader
from lumiere.loading.resolver import resolve_value

# Create loader for individual datasets
_dataset_loader = ConfigLoader[Dataset](get_registry(Dataset))


def load(config: dict[str, Any], container: Any = None) -> DataLoader:
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
