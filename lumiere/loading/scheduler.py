"""Scheduler loading functionality."""

from typing import Any

import torch

from lumiere.discover import get_registry
from lumiere.loading.loader import ConfigLoader

# Create loader
_loader = ConfigLoader[torch.optim.lr_scheduler.LRScheduler](
    get_registry(torch.optim.lr_scheduler.LRScheduler), required_params=["optimizer"]
)


def load(
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    container: Any = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Load and return a LRScheduler instance from a configuration dictionary.

    The configuration must contain a 'name' or 'type' field with the registered
    scheduler identifier, plus any additional keyword arguments required for
    that scheduler's initialization.

    Dependencies can be injected via a DependencyContainer. Values in the config
    that start with "@" (e.g., "@warmup_steps") will be resolved from the container.

    Args:
        config: Configuration dictionary containing:
            - 'name' or 'type': Registered identifier of the scheduler.
            - Additional key-value pairs for scheduler-specific parameters.
            - Values starting with "@" will be resolved from the container.
        optimizer: The optimizer instance to schedule.
        container: Optional DependencyContainer for resolving dependencies.

    Returns:
        Initialized LRScheduler instance.

    Example:
        >>> config = {"name": "cosine-annealing", "warmup_steps": 500}
        >>> scheduler = load(config, optimizer)
    """
    # Support both 'name' and 'type' for flexibility
    config_dict = dict(config)
    if "name" not in config_dict and "type" in config_dict:
        config_dict["name"] = config_dict.pop("type")

    return _loader.load(config_dict, optimizer, container=container)
