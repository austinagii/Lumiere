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

        from lumiere.discover import get
        scheduler_cls = get(torch.optim.lr_scheduler.LRScheduler, scheduler_name)
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
