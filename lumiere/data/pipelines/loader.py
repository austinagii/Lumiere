"""Pipeline loading utilities for initializing pipelines from configuration."""

from collections.abc import Mapping
from typing import Any

from lumiere.data.pipeline import Pipeline, get_pipeline
from lumiere.data.preprocessor import Preprocessor, get_preprocessor
from lumiere.di import DependencyContainer


def load(
    config: Mapping[str, Any], container: DependencyContainer | None = None
) -> Pipeline:
    """Load and return a Pipeline instance from a configuration dictionary.

    The configuration must contain a 'name' field with the registered pipeline
    identifier, plus any additional keyword arguments required for that pipeline's
    initialization. For nested components like preprocessors, the configuration
    should include their initialization details.

    Dependencies can be injected via a DependencyContainer. Values in the config
    that start with "@" (e.g., "@tokenizer") will be resolved from the container.
    This allows for a hybrid approach where some values come from config and
    others are injected as live objects.

    Args:
        config: Configuration dictionary containing:
            - 'name': Registered identifier of the pipeline.
            - Additional key-value pairs for pipeline-specific parameters.
            - 'preprocessors': Optional list of preprocessor configurations,
              each containing 'name' and initialization parameters.
            - Values starting with "@" will be resolved from the container.
        container: Optional DependencyContainer for resolving dependencies.
            If provided, config values like "@tokenizer" will be resolved to
            the registered dependency.

    Returns:
        Initialized Pipeline instance.

    Raises:
        ValueError: If config is missing 'name', if the specified pipeline
            is not registered, or if a dependency cannot be resolved.
        RuntimeError: If an error occurs during pipeline initialization.

    Example:
        >>> # With direct values
        >>> config = {
        ...     "name": "text",
        ...     "tokenizer": tokenizer_instance,
        ...     "batch_size": 32,
        ...     "context_size": 512,
        ...     "pad_id": 0,
        ...     "sliding_window_size": 128,
        ...     "preprocessors": [
        ...         {"name": "autoregressive", "device": "cuda"}
        ...     ]
        ... }
        >>> pipeline = load(config)
        >>>
        >>> # With dependency injection
        >>> container = DependencyContainer()
        >>> container.register("tokenizer", tokenizer_instance)
        >>> config = {
        ...     "name": "text",
        ...     "tokenizer": "@tokenizer",  # Will be injected
        ...     "batch_size": 32,
        ...     "context_size": 512,
        ...     "pad_id": 0,
        ...     "sliding_window_size": 128
        ... }
        >>> pipeline = load(config, container)
    """
    if (pipeline_name := config.get("name")) is None:
        raise ValueError("A pipeline config must contain a pipeline name.")

    pipeline_cls = get_pipeline(pipeline_name)
    if pipeline_cls is None:
        raise ValueError(
            f"The specified pipeline '{pipeline_name}' could not be found in the registry."  # noqa: E501
        )

    # Process the configuration
    init_params = {}
    for key, value in config.items():
        if key == "name":
            continue
        elif key == "preprocessors" and value is not None:
            # Initialize preprocessors from their configurations
            init_params[key] = [
                _init_preprocessor(preprocessor_config, container)
                for preprocessor_config in value
            ]
        else:
            # Resolve dependency injection references
            init_params[key] = _resolve_value(value, container, pipeline_name, key)

    try:
        return pipeline_cls(**init_params)
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while initializing pipeline '{pipeline_name}'"
        ) from e


def _resolve_value(
    value: Any, container: DependencyContainer | None, context: str, key: str
) -> Any:
    """Resolve a config value, handling dependency injection references.

    Args:
        value: The value to resolve. If it's a string starting with "@",
            it will be resolved from the container.
        container: Optional DependencyContainer for resolving dependencies.
        context: Context string for error messages (e.g., pipeline name).
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


def _init_preprocessor(
    preprocessor_config: Mapping[str, Any], container: DependencyContainer | None
) -> Preprocessor:
    """Initialize a single preprocessor from its configuration.

    Retrieves the preprocessor class from the registry and instantiates it
    with the provided parameters. Supports dependency injection for config values.

    Args:
        preprocessor_config: Configuration dictionary containing 'name' and
            preprocessor-specific initialization parameters.
        container: Optional DependencyContainer for resolving dependencies.

    Returns:
        Initialized Preprocessor instance.

    Raises:
        ValueError: If 'name' is missing or preprocessor is not registered.
        RuntimeError: If preprocessor initialization fails.
    """
    if (preprocessor_name := preprocessor_config.get("name")) is None:
        raise ValueError("A preprocessor config must contain a preprocessor name.")

    preprocessor_cls = get_preprocessor(preprocessor_name)
    if preprocessor_cls is None:
        raise ValueError(
            f"The specified preprocessor '{preprocessor_name}' could not be found in the registry."  # noqa: E501
        )

    try:
        # Resolve dependencies in preprocessor config
        init_params = {
            arg: _resolve_value(argv, container, preprocessor_name, arg)
            for arg, argv in preprocessor_config.items()
            if arg != "name"
        }
        return preprocessor_cls(**init_params)
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while initializing preprocessor '{preprocessor_name}'"
        ) from e
