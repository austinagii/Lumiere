"""Component registry system for neural network modules."""

import contextlib
import importlib
from pathlib import Path
from typing import Any

from torch import nn
from torch.nn import LayerNorm, RMSNorm


# Registry for all neural network components
_component_registry: dict[str, type[nn.Module]] = {}


def component(component_type: str, component_name: str):
    """Decorator to register a component in the global registry.

    Args:
        component_type: The type/category of component (e.g., 'attention', 'feedforward').
        component_name: The specific name of this component (e.g., 'multihead', 'linear').

    Example:
        >>> @component('attention', 'multihead')
        >>> class MultiHeadAttention(nn.Module):
        ...     pass
    """

    def decorator(cls):
        key = f"{component_type}.{component_name}"
        register_component(key, cls)
        return cls

    return decorator


def register_component(key: str, cls: type[nn.Module]) -> None:
    """Register a component class in the registry.

    Args:
        key: The full key in format 'type.name' (e.g., 'attention.multihead').
        cls: The component class to register.
    """
    _component_registry[key] = cls


def get_component(component_type: str, component_name: str) -> type[nn.Module] | None:
    """Retrieve a component class from the registry.

    Args:
        component_type: The type/category of component.
        component_name: The specific name of the component.

    Returns:
        Component class if found, None otherwise.
    """
    if not _component_registry:
        _populate_registry()

    key = f"{component_type}.{component_name}"
    return _component_registry.get(key)


def _populate_registry():
    """Populate the registry by importing all component modules."""
    nn_dir = Path(__file__).parent

    # Component subdirectories to scan
    component_dirs = ["attention", "feedforward", "embedding", "blocks"]

    for component_dir in component_dirs:
        dir_path = nn_dir / component_dir
        if not dir_path.exists():
            continue

        # Import all Python files in the directory
        module_files = dir_path.glob("*.py")
        module_files = [f for f in module_files if not f.stem.startswith("_")]

        for module_file in module_files:
            module_name = f"lumiere.nn.{component_dir}.{module_file.stem}"
            with contextlib.suppress(ImportError):
                importlib.import_module(module_name)


def create_factory(config: dict[str, Any], container: Any = None):
    """Create a factory function from a component configuration.

    This function creates a factory (callable) that can be used by the
    TransformerBuilder. The factory, when called, will instantiate the
    component with the configured parameters.

    Args:
        config: Configuration dict with 'type' and 'name' fields plus parameters.
        container: Optional DependencyContainer for resolving dependencies.

    Returns:
        A callable factory function that creates the component.

    Example:
        >>> config = {
        ...     "type": "attention",
        ...     "name": "multihead",
        ...     "num_heads": 8,
        ...     "d_key": 64
        ... }
        >>> factory = create_factory(config)
        >>> attention_module = factory()
    """
    component_type = config.get("type")
    component_name = config.get("name")

    if component_type is None or component_name is None:
        raise ValueError(
            "Component config must contain 'type' and 'name' fields. "
            f"Got: {config}"
        )

    component_cls = get_component(component_type, component_name)
    if component_cls is None:
        raise ValueError(
            f"Component '{component_type}.{component_name}' not found in registry."
        )

    # Extract parameters (everything except 'type' and 'name')
    params = {k: v for k, v in config.items() if k not in ("type", "name")}

    # Resolve dependencies if container is provided
    if container is not None:
        params = {
            k: _resolve_value(v, container, f"{component_type}.{component_name}", k)
            for k, v in params.items()
        }

    # Return a factory function
    def factory(**override_params):
        # Merge override params with configured params
        final_params = {**params, **override_params}
        return component_cls(**final_params)

    return factory


def _resolve_value(value: Any, container: Any, context: str, key: str) -> Any:
    """Resolve a config value, handling dependency injection references."""
    if isinstance(value, str) and value.startswith("@"):
        if container is None:
            raise ValueError(
                f"Dependency reference '{value}' found for '{key}' in '{context}', "
                f"but no DependencyContainer was provided."
            )

        dep_name = value[1:]
        resolved = container.get(dep_name)

        if resolved is None:
            raise ValueError(
                f"Dependency '{dep_name}' for '{key}' in '{context}' "
                f"not found in container."
            )

        return resolved

    return value


# Register PyTorch normalization layers at module import time
register_component("normalization.layer", LayerNorm)
register_component("normalization.rms", RMSNorm)
