"""Model loading utilities for building models from configuration."""

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from lumiere.builder import ModelSpec, TransformerBuilder
from lumiere.di import DependencyContainer


def load(
    config: Mapping[str, Any], container: DependencyContainer | None = None
) -> nn.Module:
    """Load and return a Model instance from a hierarchical configuration.

    This loader uses the TransformerBuilder to construct models from hierarchical
    specifications with factory configurations. It supports dependency injection
    for all configuration values.

    Args:
        config: Hierarchical configuration dictionary. Factory fields (like
            'embedding_factory', 'block_factory') should contain 'type' and
            'name' fields plus component-specific parameters.
        container: Optional DependencyContainer for resolving dependencies.

    Returns:
        Initialized Model instance.

    Raises:
        ValueError: If a dependency cannot be resolved or component not found.
        RuntimeError: If an error occurs during model initialization.

    Example:
        >>> config = {
        ...     "vocab_size": 30000,
        ...     "context_size": 512,
        ...     "num_blocks": 12,
        ...     "embedding_factory": {
        ...         "type": "embedding",
        ...         "name": "sinusoidal",
        ...         "padding_id": 0
        ...     },
        ...     "block_factory": {
        ...         "type": "block",
        ...         "name": "standard",
        ...         "attention_factory": {
        ...             "type": "attention",
        ...             "name": "multihead",
        ...             "num_heads": 8
        ...         },
        ...         "feedforward_factory": {
        ...             "type": "feedforward",
        ...             "name": "linear",
        ...             "d_ff": 2048
        ...         },
        ...         "normalization_factory": {
        ...             "type": "normalization",
        ...             "name": "rms"
        ...         }
        ...     },
        ...     "normalization_factory": {
        ...         "type": "normalization",
        ...         "name": "rms"
        ...     }
        ... }
        >>> model = load(config)
    """
    try:
        # Resolve dependencies in the config
        resolved_config = _resolve_nested_config(config, container)

        # Build the model using TransformerBuilder
        spec = ModelSpec(resolved_config)
        return TransformerBuilder.build(spec, container=container)
    except Exception as e:
        raise RuntimeError(f"Error building model from spec: {e}") from e


def _resolve_nested_config(
    config: Mapping[str, Any], container: DependencyContainer | None
) -> dict[str, Any]:
    """Recursively resolve dependencies in a nested configuration.

    Args:
        config: Configuration dictionary that may contain nested dicts.
        container: Optional DependencyContainer for resolving dependencies.

    Returns:
        Configuration with all dependencies resolved.
    """
    resolved = {}
    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively resolve nested configs
            resolved[key] = _resolve_nested_config(value, container)
        elif isinstance(value, list):
            # Resolve list items
            resolved[key] = [
                _resolve_nested_config(item, container)
                if isinstance(item, dict)
                else _resolve_value(item, container, "config", key)
                for item in value
            ]
        else:
            # Resolve single values
            resolved[key] = _resolve_value(value, container, "config", key)
    return resolved


def _resolve_value(value: Any, container: DependencyContainer | None, context: str, key: str) -> Any:
    """Resolve a config value, handling dependency injection references.

    Args:
        value: The value to resolve. If it's a string starting with "@",
            it will be resolved from the container.
        container: Optional DependencyContainer for resolving dependencies.
        context: Context string for error messages.
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
