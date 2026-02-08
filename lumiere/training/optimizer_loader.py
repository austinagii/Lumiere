"""Classes and utilities for creating and loading optimizers.

Custom optimizers can be created and (through the registration system) dynamically
discovered and loaded by name, enabling flexible training configurations.

Example:
    >>> # Define and register a custom optimizer
    >>> @optimizer("my-optimizer")
    >>> class MyOptimizer:
    ...     def __init__(self, params, lr: float, momentum: float):
    ...         self.params = params
    ...         self.lr = lr
    ...         self.momentum = momentum
    ...
    >>> # Use the registered optimizer via OptimizerLoader
    >>> opt = OptimizerLoader.load(
    ...     config={"name": "my-optimizer", "lr": 0.001, "momentum": 0.9},
    ...     params=model.parameters()
    ... )
"""

from collections.abc import Mapping
from typing import Any

import torch

from lumiere.loading import Registry

# A registry of optimizers indexed by custom names.
_registry = Registry[type[torch.optim.Optimizer]](
    name="optimizer",
    base_module="lumiere.training.optimizers",
    discovery_paths=["."],
)

# Expose existing API for backward compatibility
optimizer = _registry.decorator
register_optimizer = _registry.register
get_optimizer = _registry.get

# Create loader
from lumiere.loading import ConfigLoader

_loader = ConfigLoader[torch.optim.Optimizer](_registry, required_params=["params"])


def load(
    config: Mapping[str, Any],
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


class OptimizerLoader:
    """Load an optimizer from a configuration specification.

    The loader dynamically discovers and instantiates registered optimizers
    based on a configuration dictionary containing the optimizer name and
    its initialization parameters.
    """

    @classmethod
    def load(
        cls, config: Mapping[str, Any], params
    ) -> torch.optim.Optimizer:
        """Load an optimizer from a configuration specification.

        Args:
            config: Configuration dictionary containing:
                - 'name' or 'type': Registered identifier of the optimizer.
                - Additional key-value pairs for optimizer-specific parameters.
            params: Model parameters to optimize (typically model.parameters()).

        Returns:
            Initialized optimizer instance.

        Raises:
            ValueError: If 'name'/'type' is missing or optimizer is not registered.
            RuntimeError: If optimizer initialization fails.

        Example:
            >>> config = {"name": "adamw", "lr": 0.001, "weight_decay": 0.01}
            >>> optimizer = OptimizerLoader.load(config, model.parameters())
        """
        # Support both 'name' and 'type' for flexibility
        optimizer_name = config.get("name") or config.get("type")
        if optimizer_name is None:
            raise ValueError(
                "Optimizer config must contain either 'name' or 'type' field."
            )

        optimizer_cls = get_optimizer(optimizer_name)
        if optimizer_cls is None:
            raise ValueError(
                f"The specified optimizer '{optimizer_name}' could not be found in the registry."  # noqa: E501
            )

        try:
            # Extract all parameters except 'name' and 'type'
            optimizer_params = {
                arg: argv
                for arg, argv in config.items()
                if arg not in ("name", "type")
            }
            return optimizer_cls(params, **optimizer_params)
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while initializing optimizer '{optimizer_name}'"
            ) from e
