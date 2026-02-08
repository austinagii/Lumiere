"""
Dependency injection resolver for Lumière loaders.

Provides utilities for resolving configuration values with dependency injection
support using the @variable syntax.
"""

from typing import Any


def resolve_value(value: Any, container: Any = None) -> Any:
    """
    Resolve a configuration value with dependency injection support.

    This function handles the @variable syntax for dependency injection, allowing
    configuration values to reference variables from a DependencyContainer.

    Args:
        value: The value to resolve. Can be:
            - str starting with '@': Variable reference to resolve from container
            - dict: Recursively resolve all values in the dictionary
            - list: Recursively resolve all items in the list
            - Any other type: Return as-is
        container: Optional DependencyContainer to resolve variable references from

    Returns:
        The resolved value with all @variable references replaced by their actual values

    Examples:
        >>> resolve_value("@vocab_size", container)
        50000
        >>> resolve_value({"lr": 0.001, "betas": "@betas"}, container)
        {"lr": 0.001, "betas": (0.9, 0.999)}
        >>> resolve_value([1, 2, "@hidden_size"], container)
        [1, 2, 768]
    """
    if isinstance(value, str) and value.startswith("@"):
        if container is None:
            raise ValueError(
                f"Cannot resolve variable '{value}' without a dependency container"
            )
        var_name = value[1:]  # Remove '@' prefix
        if not hasattr(container, var_name):
            raise ValueError(
                f"Variable '{var_name}' not found in dependency container"
            )
        return getattr(container, var_name)
    elif isinstance(value, dict):
        return {k: resolve_value(v, container) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_value(item, container) for item in value]
    else:
        return value
