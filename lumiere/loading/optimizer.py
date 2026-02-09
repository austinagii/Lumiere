"""Optimizer loading functionality."""

from typing import Any

import torch

from lumiere.discover import get_registry
from lumiere.loading.loader import ConfigLoader

# Create loader
_loader = ConfigLoader[torch.optim.Optimizer](
    get_registry(torch.optim.Optimizer), required_params=["params"]
)


def load(
    config: dict[str, Any],
    params,
    container: Any = None,
) -> torch.optim.Optimizer:
    """Load and return an Optimizer instance from a configuration dictionary.

    The configuration must contain a 'name' or 'type' field with the registered
    optimizer identifier, plus any additional keyword arguments required for
    that optimizer's initialization.

    Dependencies can be injected via a DependencyContainer. Values in the config
    that start with "@" (e.g., "@learning_rate") will be resolved from the container.

    Args:
        config: Configuration dictionary containing:
            - 'name' or 'type': Registered identifier of the optimizer.
            - Additional key-value pairs for optimizer-specific parameters.
            - Values starting with "@" will be resolved from the container.
        params: Model parameters to optimize (typically model.parameters()).
        container: Optional DependencyContainer for resolving dependencies.

    Returns:
        Initialized Optimizer instance.

    Example:
        >>> config = {"name": "adamw", "lr": 0.001, "weight_decay": 0.01}
        >>> optimizer = load(config, model.parameters())
    """
    # Support both 'name' and 'type' for flexibility
    config_dict = dict(config)
    if "name" not in config_dict and "type" in config_dict:
        config_dict["name"] = config_dict.pop("type")

    return _loader.load(config_dict, params, container=container)
