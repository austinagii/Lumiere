"""Optimizer loading utilities for initializing optimizers from configuration."""

from collections.abc import Mapping
from typing import Any

import torch

from lumiere.di import DependencyContainer
from lumiere.training.optimizer_loader import get_optimizer


def load(
    config: Mapping[str, Any],
    params,
    container: DependencyContainer | None = None,
) -> torch.optim.Optimizer:
    """Load and return an Optimizer instance from a configuration dictionary.

    The configuration must contain a 'name' or 'type' field with the registered
    optimizer identifier, plus any additional keyword arguments required for
    that optimizer's initialization.

    Dependencies can be injected via a DependencyContainer. Values in the config
    that start with "@" (e.g., "@learning_rate") will be resolved from the container.
    This allows for a hybrid approach where some values come from config and
    others are injected as live objects.

    Args:
        config: Configuration dictionary containing:
            - 'name' or 'type': Registered identifier of the optimizer.
            - Additional key-value pairs for optimizer-specific parameters.
            - Values starting with "@" will be resolved from the container.
        params: Model parameters to optimize (typically model.parameters()).
        container: Optional DependencyContainer for resolving dependencies.
            If provided, config values like "@lr" will be resolved to
            the registered dependency.

    Returns:
        Initialized Optimizer instance.

    Raises:
        ValueError: If config is missing 'name'/'type', if the specified optimizer
            is not registered, or if a dependency cannot be resolved.
        RuntimeError: If an error occurs during optimizer initialization.

    Example:
        >>> # With direct values
        >>> config = {
        ...     "name": "adamw",
        ...     "lr": 0.001,
        ...     "weight_decay": 0.01
        ... }
        >>> optimizer = load(config, model.parameters())
        >>>
        >>> # With dependency injection
        >>> container = DependencyContainer()
        >>> container.register("lr", 0.001)
        >>> container.register("weight_decay", 0.01)
        >>> config = {
        ...     "name": "adamw",
        ...     "lr": "@lr",  # Will be injected
        ...     "weight_decay": "@weight_decay"  # Will be injected
        ... }
        >>> optimizer = load(config, model.parameters(), container)
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
        # Resolve dependencies in config
        optimizer_params = {
            arg: _resolve_value(argv, container, optimizer_name, arg)
            for arg, argv in config.items()
            if arg not in ("name", "type")
        }
        return optimizer_cls(params, **optimizer_params)
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while initializing optimizer '{optimizer_name}'"
        ) from e


def _resolve_value(
    value: Any, container: DependencyContainer | None, context: str, key: str
) -> Any:
    """Resolve a config value, handling dependency injection references.

    Args:
        value: The value to resolve. If it's a string starting with "@",
            it will be resolved from the container.
        container: Optional DependencyContainer for resolving dependencies.
        context: Context string for error messages (e.g., optimizer name).
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
