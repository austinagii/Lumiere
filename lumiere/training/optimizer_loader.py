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

        from lumiere.discover import get
        optimizer_cls = get(torch.optim.Optimizer, optimizer_name)
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
