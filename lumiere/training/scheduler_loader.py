"""Classes and utilities for creating and loading learning rate schedulers.

Custom schedulers can be created and (through the registration system) dynamically
discovered and loaded by name, enabling flexible training configurations.

Example:
    >>> # Define and register a custom scheduler
    >>> @scheduler("my-scheduler")
    >>> class MyScheduler:
    ...     def __init__(self, optimizer, warmup_steps: int):
    ...         self.optimizer = optimizer
    ...         self.warmup_steps = warmup_steps
    ...
    ...     def __call__(self) -> torch.optim.lr_scheduler.LRScheduler:
    ...         # Return scheduler instance
    ...         return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100)
    ...
    >>> # Use the registered scheduler via SchedulerLoader
    >>> sched = SchedulerLoader.load(
    ...     config={"name": "my-scheduler", "warmup_steps": 500},
    ...     optimizer=optimizer
    ... )
"""

from collections.abc import Mapping
from typing import Any

import torch

from lumiere.loading import Registry

# A registry of schedulers indexed by custom names.
# Scheduler classes should be callable that accept an optimizer and return
# a torch.optim.lr_scheduler.LRScheduler instance
_registry = Registry[type](
    name="scheduler",
    base_module="lumiere.training.lr_schedulers",
    discovery_paths=["."],
)

# Expose existing API for backward compatibility
scheduler = _registry.decorator
register_scheduler = _registry.register
get_scheduler = _registry.get

# Create loader
from lumiere.loading import ConfigLoader

_loader = ConfigLoader[torch.optim.lr_scheduler.LRScheduler](
    _registry, required_params=["optimizer"]
)


def load(
    config: Mapping[str, Any],
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


class SchedulerLoader:
    """Load a learning rate scheduler from a configuration specification.

    The loader dynamically discovers and instantiates registered schedulers
    based on a configuration dictionary containing the scheduler name and
    its initialization parameters.
    """

    @classmethod
    def load(
        cls, config: Mapping[str, Any], optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Load a scheduler from a configuration specification.

        Args:
            config: Configuration dictionary containing:
                - 'name' or 'type': Registered identifier of the scheduler.
                - Additional key-value pairs for scheduler-specific parameters.
            optimizer: The optimizer instance to schedule.

        Returns:
            Initialized scheduler instance.

        Raises:
            ValueError: If 'name'/'type' is missing or scheduler is not registered.
            RuntimeError: If scheduler initialization fails.

        Example:
            >>> config = {
            ...     "name": "cosine-annealing",
            ...     "warmup_steps": 500,
            ...     "max_epochs": 100
            ... }
            >>> scheduler = SchedulerLoader.load(config, optimizer)
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
            # Extract all parameters except 'name' and 'type'
            scheduler_params = {
                arg: argv
                for arg, argv in config.items()
                if arg not in ("name", "type")
            }
            return scheduler_cls(optimizer, **scheduler_params)
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while initializing scheduler '{scheduler_name}'"
            ) from e
