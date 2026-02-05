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

import contextlib
import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


# A registry of optimizers indexed by custom names.
_optimizer_registry: dict[str, type[torch.optim.Optimizer]] = {}


def optimizer(optimizer_name: str):
    """Decorator to register an optimizer class in the global registry.

    Registered optimizers can be retrieved by name using get_optimizer().

    Args:
        optimizer_name: Unique identifier for the optimizer in the registry.

    """

    def decorator(cls):
        register_optimizer(optimizer_name, cls)
        return cls

    return decorator


def register_optimizer(name: str, cls: type[torch.optim.Optimizer]) -> None:
    _optimizer_registry[name] = cls


def get_optimizer(optimizer_name: str) -> type[torch.optim.Optimizer] | None:
    """Retrieve an optimizer class from the registry by name.

    Args:
        optimizer_name: Registered identifier of the optimizer to retrieve.

    Returns:
        Optimizer class if found in the registry, None otherwise.
    """
    if not _optimizer_registry:  # Refresh the imports.
        optimizers_dir = Path(__file__).parent / "optimizers"
        if not optimizers_dir.exists():
            return None

        module_files = optimizers_dir.glob("*.py")
        module_files = [f for f in module_files if not f.stem.startswith("_")]

        # Import each module to trigger @optimizer decorator registration
        for module_file in module_files:
            module_name = f"lumiere.training.optimizers.{module_file.stem}"
            with contextlib.suppress(ImportError):
                importlib.import_module(module_name)

    return _optimizer_registry.get(optimizer_name)


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
