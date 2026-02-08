"""Scheduler loading utilities for initializing schedulers from configuration."""

from collections.abc import Mapping
from typing import Any

import torch

from lumiere.di import DependencyContainer
from lumiere.training.scheduler_loader import get_scheduler


def load(
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    container: DependencyContainer | None = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Load and return a LRScheduler instance from a configuration dictionary.

    The configuration must contain a 'name' or 'type' field with the registered
    scheduler identifier, plus any additional keyword arguments required for
    that scheduler's initialization.

    Dependencies can be injected via a DependencyContainer. Values in the config
    that start with "@" (e.g., "@warmup_steps") will be resolved from the container.
    This allows for a hybrid approach where some values come from config and
    others are injected as live objects.

    Args:
        config: Configuration dictionary containing:
            - 'name' or 'type': Registered identifier of the scheduler.
            - Additional key-value pairs for scheduler-specific parameters.
            - Values starting with "@" will be resolved from the container.
        optimizer: The optimizer instance to schedule.
        container: Optional DependencyContainer for resolving dependencies.
            If provided, config values like "@warmup_steps" will be resolved to
            the registered dependency.

    Returns:
        Initialized LRScheduler instance.

    Raises:
        ValueError: If config is missing 'name'/'type', if the specified scheduler
            is not registered, or if a dependency cannot be resolved.
        RuntimeError: If an error occurs during scheduler initialization.

    Example:
        >>> # With direct values
        >>> config = {
        ...     "name": "cosine-annealing",
        ...     "warmup_steps": 500,
        ...     "max_epochs": 100,
        ...     "epoch_steps": 2000
        ... }
        >>> scheduler = load(config, optimizer)
        >>>
        >>> # With dependency injection
        >>> container = DependencyContainer()
        >>> container.register("warmup_steps", 500)
        >>> container.register("max_epochs", 100)
        >>> config = {
        ...     "name": "cosine-annealing",
        ...     "warmup_steps": "@warmup_steps",  # Will be injected
        ...     "max_epochs": "@max_epochs",  # Will be injected
        ...     "epoch_steps": 2000
        ... }
        >>> scheduler = load(config, optimizer, container)
    """
    # Support both 'name' and 'type' for flexibility
    scheduler_name = config.get("name") or config.get("type")
    if scheduler_name is None:
        raise ValueError(
            "Scheduler config must contain either 'name' or 'type' field."
        )

    scheduler_cls = get_scheduler(scheduler_name)
    if scheduler_cls is None:
        raise ValueError(
            f"The specified scheduler '{scheduler_name}' could not be found in the registry."  # noqa: E501
        )

    try:
        # Resolve dependencies in config
        scheduler_params = {
            arg: _resolve_value(argv, container, scheduler_name, arg)
            for arg, argv in config.items()
            if arg not in ("name", "type")
        }
        return scheduler_cls(optimizer, **scheduler_params)
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while initializing scheduler '{scheduler_name}'"
        ) from e


def _resolve_value(
    value: Any, container: DependencyContainer | None, context: str, key: str
) -> Any:
    """Resolve a config value, handling dependency injection references.

    Args:
        value: The value to resolve. If it's a string starting with "@",
            it will be resolved from the container.
        container: Optional DependencyContainer for resolving dependencies.
        context: Context string for error messages (e.g., scheduler name).
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
